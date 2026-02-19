from common.schemas import ShotPlan


def test_shot_plan_accepts_continuity_fields():
    shot = ShotPlan(
        duration_sec=5,
        shot_prompt="STYLE: anime | CHARACTER: x | ACTION: y | CAMERA: dolly-in | SCENE: z | LIGHTING: moon | MOOD: tense",
        camera="dolly-in",
        action="run",
        environment="street",
        seed=123,
        negative_prompt="text",
        resolution="1280x720",
        continuity_mode="last_frame",
        input_ref_image_path="/tmp/ref.png",
    )
    assert shot.continuity_mode == "last_frame"
