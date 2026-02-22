from __future__ import annotations

from abc import ABC, abstractmethod
import gc
import logging
from pathlib import Path
import subprocess
from typing import Any

# Load ftfy before any Wan pipeline runs; tokenizer may reference it by name.
try:
    import ftfy as _ftfy
    import builtins
    builtins.ftfy = _ftfy  # some tokenizers expect 'ftfy' in scope
except ImportError:
    _ftfy = None  # type: ignore[misc, assignment]

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
    """Wan2.x-backed runner (T2V + I2V) with OOM fallback. Model is loaded once on first shot and cached for all subsequent shots."""

    def __init__(
        self,
        model_path: str = "/models/wan2",
        device: str = "cuda",
        dtype: str = "float16",
        vram_mode: str = "safe",
        num_inference_steps: int = 25,
        output_root: str = "/tmp/vira",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.vram_mode = vram_mode
        self.num_inference_steps = num_inference_steps
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._torch = self._import("torch")
        self._imageio = self._import("imageio")
        self._np = self._import("numpy")
        self._PILImage = self._import("PIL", attr="Image")  # PIL.Image class (has .open)
        # Use DiffusionPipeline so we work across diffusers versions (AutoPipelineForTextToVideo not in all releases)
        self._DiffusionPipeline = self._import("diffusers", attr="DiffusionPipeline")

        self._dtype = self._resolve_dtype(dtype)
        self._model_root = self._resolve_model_root(Path(model_path))
        # Lazy-load pipelines on first use; reuse for all shots (no reload per shot)
        self._t2v: Any = None
        self._i2v: Any = None

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

    def _resolve_model_root(self, path: Path) -> Path:
        """Return the directory that contains model_index.json (may be path or a single subdir)."""
        if (path / "model_index.json").exists():
            return path
        for sub in path.iterdir():
            if sub.is_dir() and (sub / "model_index.json").exists():
                return sub
        return path

    def _load_t2v_pipeline(self):
        try:
            if _ftfy is None:
                raise RuntimeError(
                    "Wan tokenizer requires 'ftfy'. Install with: pip install ftfy"
                )
            load_path = str(self._model_root)
            pipe = self._DiffusionPipeline.from_pretrained(load_path, torch_dtype=self._dtype)
            # In safe mode use CPU offload so we never put the full model on GPU (avoids OOM on 16GB).
            if self.vram_mode == "safe" and hasattr(pipe, "enable_model_cpu_offload"):
                self._apply_vram_mode(pipe)
                # Do not call pipe.to(device); offload keeps weights on CPU and moves to GPU per layer during inference.
            else:
                pipe.to(self.device)
                self._apply_vram_mode(pipe)
            return pipe
        except Exception as exc:
            try:
                listing = list(Path(self.model_path).iterdir()) if Path(self.model_path).exists() else []
            except Exception:
                listing = []
            raise RuntimeError(
                f"Failed to load Wan T2V model from '{self.model_path}' (resolved to '{self._model_root}'). "
                f"Ensure the volume mount points to a diffusers model dir containing model_index.json. "
                f"Contents of WAN_MODEL_PATH: {listing}"
            ) from exc

    def _get_t2v(self):
        """Load T2V pipeline once on first use; return cached pipeline for all subsequent shots."""
        if self._t2v is None:
            self._t2v = self._load_t2v_pipeline()
            logger.info("Wan T2V model loaded (cached for all shots in this worker)")
        return self._t2v

    def _get_i2v(self):
        """Reuse T2V pipeline for I2V; many Wan checkpoints support both prompt= and image= in __call__."""
        return self._get_t2v()

    def _apply_vram_mode(self, pipe: Any):
        mode = self.vram_mode
        if mode == "safe":
            # Sequential offload uses less peak VRAM than model_cpu_offload (one submodule at a time)
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
            elif hasattr(pipe, "enable_model_cpu_offload"):
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
    def _round_to_multiple_16(w: int, h: int) -> tuple[int, int]:
        """Wan requires width and height divisible by 16."""
        return (w // 16) * 16, (h // 16) * 16

    def _clear_gpu_memory(self):
        """Free GPU cache and run GC. Avoid moving float16 pipeline to CPU (unsupported); just clear cache."""
        if hasattr(self._torch.cuda, "empty_cache"):
            self._torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text or "cuda out of memory" in text

    def _fallback_params(self, resolution: str, frames: int, level: int = 1) -> tuple[str, int]:
        """Return (resolution, frames) for OOM fallback. level 1=75%, 2=640x352, 3=512x320 min frames."""
        w, h = self._parse_res(resolution)
        if level == 1:
            w2 = max(640, int(w * 0.75))
            h2 = max(360, int(h * 0.75))
            f2 = max(8, int(frames * 0.75))
        elif level == 2:
            w2, h2 = 640, 352  # divisible by 16
            f2 = min(33, max(9, int(frames * 0.5)))
        else:
            # Level 3: minimum that still produces a short clip (512x320, 17 frames)
            w2, h2 = 512, 320
            f2 = min(17, max(9, int(frames * 0.25)))
        w2, h2 = self._round_to_multiple_16(w2, h2)
        w2, h2 = max(16, w2), max(16, h2)
        f2 = ((f2 - 1) // 4) * 4 + 1
        f2 = max(9, f2)
        return f"{w2}x{h2}", f2

    def _frames_to_list(self, video_frames) -> list:
        """Normalize pipeline output to a list of 3D frames (each C,H,W or H,W,C). Handles 5D/4D tensors and lists."""
        if hasattr(video_frames, "shape"):
            arr = video_frames.cpu().numpy() if hasattr(video_frames, "cpu") else self._np.asarray(video_frames)
            nd = arr.ndim
            if nd == 5:
                arr = arr[0]
                nd = 4
            if nd == 4:
                return [arr[i] for i in range(arr.shape[0])]
        if isinstance(video_frames, (list, tuple)):
            return list(video_frames)
        return [video_frames]

    def _frame_to_float_hwc(self, frame) -> Any:
        """Convert pipeline frame to (H, W, C) float in [0,1]. Handles PIL (output_type=pil), tensor, numpy."""
        if hasattr(frame, "convert"):
            frame = frame.convert("RGB")
        if hasattr(frame, "cpu") and callable(getattr(frame, "cpu", None)):
            arr = frame.cpu().numpy()
        else:
            arr = self._np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = arr.transpose(1, 2, 0)
        if self._np.issubdtype(arr.dtype, self._np.floating):
            if float(arr.min()) < -0.1:
                arr = ((arr + 1.0) * 0.5).clip(0, 1)
            else:
                arr = arr.clip(0, 1)
            return arr.astype(self._np.float32) if arr.dtype != self._np.float32 else arr
        # PIL/numpy uint8: 0-255 -> 0-1
        return (arr.astype(self._np.float32) / 255.0).clip(0, 1)

    def _materialize_frames_and_free_gpu(self, video_frames) -> list:
        """Copy frames to CPU numpy (H,W,C uint8) with contrast normalized to full 0-255. Frees GPU before return."""
        video_frames = self._frames_to_list(video_frames)
        if not video_frames:
            if hasattr(self._torch.cuda, "empty_cache"):
                self._torch.cuda.empty_cache()
            gc.collect()
            return []
        # Convert to float [0,1] and compute global min/max so we use full 0-255 range (fixes dim/gray output)
        float_frames = [self._frame_to_float_hwc(f) for f in video_frames]
        f0 = float_frames[0]
        gmin, gmax = float(self._np.min(f0)), float(self._np.max(f0))
        for a in float_frames[1:]:
            gmin = min(gmin, float(self._np.min(a)))
            gmax = max(gmax, float(self._np.max(a)))
        logger.info(
            "Wan pipeline frames: count=%s first_frame shape=%s dtype=float32 global_min=%.4f global_max=%.4f",
            len(float_frames), getattr(f0, "shape", None), gmin, gmax,
        )
        span = max(gmax - gmin, 1e-6)
        out = [((a - gmin) / span * 255).clip(0, 255).astype(self._np.uint8) for a in float_frames]
        if hasattr(self._torch.cuda, "empty_cache"):
            self._torch.cuda.empty_cache()
        gc.collect()
        return out

    def _write_video(self, frames, fps: int, out_path: Path):
        """Write (H,W,C) uint8 frames to MP4 via ffmpeg for correct duration and player compatibility."""
        if not frames:
            raise ValueError("No frames to write")
        h, w = frames[0].shape[0], frames[0].shape[1]
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for arr in frames:
                if arr.shape[0] != h or arr.shape[1] != w or arr.ndim != 3:
                    arr = self._np.asarray(arr)
                    if arr.ndim == 3 and arr.shape[0] in (3, 4):
                        arr = arr.transpose(1, 2, 0)
                    if arr.shape[0] != h or arr.shape[1] != w:
                        raise ValueError(f"Frame shape {arr.shape} != {h}x{w}")
                proc.stdin.write(self._np.ascontiguousarray(arr).tobytes())
        finally:
            proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")

    def _run_t2v_once(self, shot_prompt: str, negative_prompt: str, frames: int, resolution: str, seed: int):
        width, height = self._parse_res(resolution)
        width, height = self._round_to_multiple_16(width, height)
        width, height = max(16, width), max(16, height)
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        pipe = self._get_t2v()
        kwargs = dict(
            prompt=shot_prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            width=width,
            height=height,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )
        try:
            output = pipe(**kwargs, output_type="pil")
        except TypeError:
            output = pipe(**kwargs)
        return output.frames[0] if hasattr(output, "frames") else output[0]

    def _run_i2v_once(self, ref_image: str, shot_prompt: str, negative_prompt: str, frames: int, resolution: str, seed: int):
        width, height = self._parse_res(resolution)
        width, height = self._round_to_multiple_16(width, height)
        width, height = max(16, width), max(16, height)
        generator = self._torch.Generator(device=self.device).manual_seed(seed)
        pipe = self._get_i2v()
        # Wan 1.3B is T2V-only (WanPipeline); I2V needs WanImageToVideoPipeline. Fall back to T2V if no image arg.
        kwargs_i2v = dict(
            prompt=shot_prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            width=width,
            height=height,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )
        try:
            image = self._PILImage.open(ref_image).convert("RGB").resize((width, height))
            try:
                output = pipe(image=image, **kwargs_i2v, output_type="pil")
            except TypeError:
                output = pipe(image=image, **kwargs_i2v)
        except TypeError as e:
            if "unexpected keyword argument 'image'" in str(e) or "unexpected keyword argument" in str(e):
                logger.info("Pipeline does not support image input (T2V-only), using prompt only for I2V shot")
                try:
                    output = pipe(**kwargs_i2v, output_type="pil")
                except TypeError:
                    output = pipe(**kwargs_i2v)
            else:
                raise
        return output.frames[0] if hasattr(output, "frames") else output[0]

    def generate_video(self, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        frames = max(8, duration * fps)
        # Wan: (num_frames - 1) divisible by 4
        frames = ((frames - 1) // 4) * 4 + 1
        frames = max(9, frames)
        out = self.output_root / f"wan_t2v_{seed}.mp4"
        self._clear_gpu_memory()
        # In safe/balanced mode full res often OOMs on 22GB; start at 640x352 so we don't fill GPU and block fallbacks
        if self.vram_mode in ("safe", "balanced"):
            resolution, frames = self._fallback_params(resolution, frames, level=2)
            logger.info("Wan T2V using capped resolution in %s mode", self.vram_mode, extra={"resolution": resolution, "frames": frames})
        try:
            video_frames = self._run_t2v_once(shot_prompt, negative_prompt, frames, resolution, seed)
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            logger.warning("Wan T2V error (treating as OOM), first attempt: %s", exc, exc_info=False)
            self._clear_gpu_memory()
            fallback_res, fallback_frames = self._fallback_params(resolution, frames, level=1)
            logger.warning("Wan T2V OOM, retrying with fallback (75%%)", extra={"resolution": fallback_res, "frames": fallback_frames})
            try:
                video_frames = self._run_t2v_once(shot_prompt, negative_prompt, fallback_frames, fallback_res, seed)
            except Exception as exc2:
                if not self._is_oom(exc2):
                    raise
                logger.warning("Wan T2V error, fallback 75%% failed: %s", exc2, exc_info=False)
                self._clear_gpu_memory()
                fallback_res2, fallback_frames2 = self._fallback_params(resolution, frames, level=2)
                logger.warning("Wan T2V OOM, retrying with fallback level 2 (640x352)", extra={"resolution": fallback_res2, "frames": fallback_frames2})
                try:
                    video_frames = self._run_t2v_once(shot_prompt, negative_prompt, fallback_frames2, fallback_res2, seed)
                except Exception as exc3:
                    if not self._is_oom(exc3):
                        raise
                    logger.warning("Wan T2V error, fallback level 2 failed: %s", exc3, exc_info=False)
                    self._clear_gpu_memory()
                    fallback_res3, fallback_frames3 = self._fallback_params(resolution, frames, level=3)
                    logger.warning("Wan T2V OOM, retrying with fallback level 3 (512x320)", extra={"resolution": fallback_res3, "frames": fallback_frames3})
                    try:
                        video_frames = self._run_t2v_once(shot_prompt, negative_prompt, fallback_frames3, fallback_res3, seed)
                    except Exception as exc4:
                        logger.warning("Wan T2V error, fallback level 3 failed: %s", exc4, exc_info=False)
                        raise RuntimeError("Wan T2V failed after OOM fallbacks (reduced resolution/frames).") from exc4
        materialized = self._materialize_frames_and_free_gpu(video_frames)
        self._write_video(materialized, fps=fps, out_path=out)
        return str(out)

    def generate_video_from_image(self, ref_image: str, shot_prompt: str, negative_prompt: str, duration: int, resolution: str, fps: int, seed: int) -> str:
        frames = max(8, duration * fps)
        frames = ((frames - 1) // 4) * 4 + 1
        frames = max(9, frames)
        out = self.output_root / f"wan_i2v_{seed}.mp4"
        self._clear_gpu_memory()
        if self.vram_mode in ("safe", "balanced"):
            resolution, frames = self._fallback_params(resolution, frames, level=2)
            logger.info("Wan I2V using capped resolution in %s mode", self.vram_mode, extra={"resolution": resolution, "frames": frames})
        try:
            video_frames = self._run_i2v_once(ref_image, shot_prompt, negative_prompt, frames, resolution, seed)
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            self._clear_gpu_memory()
            fallback_res, fallback_frames = self._fallback_params(resolution, frames, level=1)
            logger.warning("Wan I2V OOM, retrying with fallback (75%%)", extra={"resolution": fallback_res, "frames": fallback_frames})
            try:
                video_frames = self._run_i2v_once(ref_image, shot_prompt, negative_prompt, fallback_frames, fallback_res, seed)
            except Exception as exc2:
                if not self._is_oom(exc2):
                    raise
                self._clear_gpu_memory()
                fallback_res2, fallback_frames2 = self._fallback_params(resolution, frames, level=2)
                logger.warning("Wan I2V OOM, retrying with fallback level 2 (640x352)", extra={"resolution": fallback_res2, "frames": fallback_frames2})
                try:
                    video_frames = self._run_i2v_once(ref_image, shot_prompt, negative_prompt, fallback_frames2, fallback_res2, seed)
                except Exception as exc3:
                    if not self._is_oom(exc3):
                        raise
                    self._clear_gpu_memory()
                    fallback_res3, fallback_frames3 = self._fallback_params(resolution, frames, level=3)
                    logger.warning("Wan I2V OOM, retrying with fallback level 3 (512x320)", extra={"resolution": fallback_res3, "frames": fallback_frames3})
                    try:
                        video_frames = self._run_i2v_once(ref_image, shot_prompt, negative_prompt, fallback_frames3, fallback_res3, seed)
                    except Exception as exc4:
                        raise RuntimeError("Wan I2V failed after OOM fallbacks (reduced resolution/frames).") from exc4
        materialized = self._materialize_frames_and_free_gpu(video_frames)
        self._write_video(materialized, fps=fps, out_path=out)
        return str(out)
