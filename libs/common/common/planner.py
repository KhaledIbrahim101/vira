import random
from common.schemas import JobPlan, ShotPlan

CAMERA_MOTIONS = ["dolly-in", "orbit", "pan-right", "handheld subtle", "crane-up", "push-in"]


def _character_profile(seed_key: str) -> dict:
    rng = random.Random(seed_key)
    hair = ["silver short hair", "long black hair", "pink twin tails", "white bob cut"]
    eyes = ["amber eyes", "teal eyes", "violet eyes", "crimson eyes"]
    outfit = ["school uniform", "samurai armor", "streetwear jacket", "futuristic pilot suit"]
    archetype = ["stoic swordswoman", "reckless hero", "calm tactician", "mischievous mage"]
    return {
        "name": f"Char-{rng.randint(100,999)}",
        "archetype": rng.choice(archetype),
        "hair": rng.choice(hair),
        "eyes": rng.choice(eyes),
        "outfit": rng.choice(outfit),
    }


def make_plan(prompt: str, duration_sec: int, aspect_ratio: str = "16:9") -> JobPlan:
    seed_key = f"{prompt}:{duration_sec}:{aspect_ratio}"
    rng = random.Random(seed_key)
    if duration_sec >= 25:
        shot_count = max(3, min(6, duration_sec // 5))
    else:
        shot_count = max(2, min(4, duration_sec // 6 + 1))

    base_duration = duration_sec // shot_count
    remaining = duration_sec % shot_count
    character = _character_profile(seed_key)
    style_block = "anime cinematic, high detail, clean line art, expressive shading, no text"

    actions = ["sprints forward", "locks stance", "turns dramatically", "dodges attack", "final heroic pose", "quiet aftermath"]
    envs = ["neon city street", "misty shrine", "sunset rooftop", "rainy alley", "train platform", "moonlit bridge"]
    lightings = ["soft rim light", "neon bounce lighting", "golden hour glow", "blue moonlight", "storm flashes"]
    moods = ["tense", "heroic", "melancholic", "determined", "mysterious"]

    resolution = "1280x720" if aspect_ratio == "16:9" else "960x540"
    shots = []
    for i in range(shot_count):
        dur = base_duration + (1 if i < remaining else 0)
        camera_motion = CAMERA_MOTIONS[i % len(CAMERA_MOTIONS)]
        prompt_sections = [
            f"STYLE: {style_block}",
            f"CHARACTER: {character['name']}, {character['archetype']}, {character['hair']}, {character['eyes']}, {character['outfit']}",
            f"ACTION: {actions[i % len(actions)]}",
            f"CAMERA: {camera_motion}",
            f"SCENE: {envs[i % len(envs)]}",
            f"LIGHTING: {lightings[i % len(lightings)]}",
            f"MOOD: {moods[i % len(moods)]}",
            f"USER_INTENT: {prompt}",
        ]
        shots.append(
            ShotPlan(
                duration_sec=dur,
                shot_prompt=" | ".join(prompt_sections),
                camera=camera_motion,
                action=actions[i % len(actions)],
                environment=envs[i % len(envs)],
                seed=rng.randint(1, 2_000_000_000),
                negative_prompt="text, watermark, logo, subtitles, blurry, distorted face",
                resolution=resolution,
                fps_internal=24,
                continuity_mode="none" if i == 0 else "last_frame",
            )
        )
    return JobPlan(style_block=style_block, character=character, shots=shots)
