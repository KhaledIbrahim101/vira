from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class ModelRunner(ABC):
    @abstractmethod
    def generate_video(self, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_video_from_image(self, ref_image: str, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        raise NotImplementedError


class DummyRunner(ModelRunner):
    def __init__(self, output_root: str = "/tmp/vira"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def generate_video(self, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        out = self.output_root / f"shot_{seed}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={resolution}:rate={fps}",
            "-f", "lavfi", "-i", f"sine=frequency={300 + (seed % 300)}:sample_rate=44100",
            "-t", str(duration), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", str(out),
        ]
        subprocess.run(cmd, check=True)
        return str(out)

    def generate_video_from_image(self, ref_image: str, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        out = self.output_root / f"shot_ref_{seed}.mp4"
        vf = f"scale={resolution.split('x')[0]}:{resolution.split('x')[1]},zoompan=z='min(zoom+0.0015,1.2)':d=1:s={resolution}:fps={fps},format=yuv420p"
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", ref_image,
            "-f", "lavfi", "-i", f"sine=frequency={350 + (seed % 200)}:sample_rate=44100",
            "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(out),
        ]
        subprocess.run(cmd, check=True)
        return str(out)


class WanRunner(ModelRunner):
    """Wan2.x-backed runner (T2V + I2V) with OOM fallback and one-time model loading."""

    def __init__(
        self,
        model_path: str = "/models/wan2",
        device: str = "cuda",
        dtype: str = "float16",
        vram_mode: str = "safe",
        output_root: str = "/tmp/vira",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.vram_mode = vram_mode
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._torch = self._import("torch")
        self._imageio = self._import("imageio")
        self._np = self._import("numpy")
        self._PILImage = self._import("PIL.Image", attr="Image")
        # Use DiffusionPipeline so we work across diffusers versions (AutoPipelineForTextToVideo not in all releases)
        self._DiffusionPipeline = self._import("diffusers", attr="DiffusionPipeline")

        self._dtype = self._resolve_dtype(dtype)
        self._t2v = self._load_t2v_pipeline()
        self._i2v = self._load_i2v_pipeline()

    def _import(self, module_name: str, attr: str | None = None):
        try:
            mod = __import__(module_name, fromlist=[attr] if attr else [])
            return getattr(mod, attr) if attr else mod
        except Exception as exc:
            raise RuntimeError(
                f"Wan backend requested, but dependency import failed ({module_name}). "
                "Install GPU requirements and ensure model runtime deps are present."
            ) from exc

    def _resolve_dtype(self, dtype: str):
        mapping = {
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "float32": self._torch.float32,
        }
        return mapping.get(dtype, self._torch.float16)

    def _load_t2v_pipeline(self):
        try:
            pipe = self._DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self._dtype)
            pipe.to(self.device)
            self._apply_vram_mode(pipe)
            return pipe
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Wan T2V model from '{self.model_path}'. Mount weights and verify WAN_MODEL_PATH."
            ) from exc

    def _load_i2v_pipeline(self):
        # Reuse T2V pipeline; many Wan-style checkpoints support both prompt= and image= in __call__
        return self._t2v

    def _apply_vram_mode(self, pipe: Any):
        mode = self.vram_mode
        if mode == "safe":
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        elif mode == "balanced":
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
        # max mode leaves defaults

    def _parse_res(self, resolution: str) -> tuple[int, int]:
        w, h = resolution.lower().split("x")
        return int(w), int(h)

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text or "cuda out of memory" in text

    def _fallback_params(self, resolution: str, frames: int) -> tuple[str, int]:
        w, h = self._parse_res(resolution)
        w2 = max(640, int(w * 0.75))
        h2 = max(360, int(h * 0.75))
        f2 = max(8, int(frames * 0.75))
        return f"{w2}x{h2}", f2

    def _write_video(self, frames, fps: int, out_path: Path):
        writer = self._imageio.get_writer(str(out_path), fps=fps)
        try:
            for frame in frames:
                # frame may be PIL image or numpy array
                if hasattr(frame, "convert"):
                    frame = frame.convert("RGB")
                writer.append_data(self._np.asarray(frame))
        finally:
            writer.close()

    def _run_t2v_once(self, shot_prompt: str, negative_prompt: str, frames: int, resolution: str, seed: int):
        width, height = self._parse_res(resolution)
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        output = self._t2v(
            prompt=shot_prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            width=width,
            height=height,
            generator=generator,
        )
        return output.frames[0] if hasattr(output, "frames") else output[0]

    def _run_i2v_once(self, ref_image: str, shot_prompt: str, negative_prompt: str, frames: int, resolution: str, seed: int):
        if self._i2v is None:
            raise RuntimeError("Wan I2V pipeline unavailable in current model package.")
        width, height = self._parse_res(resolution)
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        image = self._PILImage.open(ref_image).convert("RGB").resize((width, height))
        output = self._i2v(
            image=image,
            prompt=shot_prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            width=width,
            height=height,
            generator=generator,
        )
        return output.frames[0] if hasattr(output, "frames") else output[0]

    def generate_video(self, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        frames = max(8, duration * fps)
        out = self.output_root / f"wan_t2v_{seed}.mp4"
        try:
            video_frames = self._run_t2v_once(shot_prompt, negative_prompt, frames, resolution, seed)
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            fallback_res, fallback_frames = self._fallback_params(resolution, frames)
            logger.warning("Wan T2V OOM, retrying once with fallback params", extra={"resolution": fallback_res, "frames": fallback_frames})
            try:
                video_frames = self._run_t2v_once(shot_prompt, negative_prompt, fallback_frames, fallback_res, seed)
            except Exception as second_exc:
                raise RuntimeError("Wan T2V failed after OOM fallback (reduced resolution/frames).") from second_exc
        self._write_video(video_frames, fps=fps, out_path=out)
        return str(out)

    def generate_video_from_image(self, ref_image: str, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        frames = max(8, duration * fps)
        out = self.output_root / f"wan_i2v_{seed}.mp4"
        try:
            video_frames = self._run_i2v_once(ref_image, shot_prompt, negative_prompt, frames, resolution, seed)
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            fallback_res, fallback_frames = self._fallback_params(resolution, frames)
            logger.warning("Wan I2V OOM, retrying once with fallback params", extra={"resolution": fallback_res, "frames": fallback_frames})
            try:
                video_frames = self._run_i2v_once(ref_image, shot_prompt, negative_prompt, fallback_frames, fallback_res, seed)
            except Exception as second_exc:
                raise RuntimeError("Wan I2V failed after OOM fallback (reduced resolution/frames).") from second_exc
        self._write_video(video_frames, fps=fps, out_path=out)
        return str(out)
