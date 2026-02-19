from common.planner import make_plan


def test_planner_rich_prompt_and_shot_count_short():
    plan = make_plan("anime rooftop duel", 10)
    assert 2 <= len(plan.shots) <= 4
    assert "STYLE:" in plan.shots[0].shot_prompt
    assert "CHARACTER:" in plan.shots[0].shot_prompt
    assert plan.shots[0].continuity_mode == "none"
    if len(plan.shots) > 1:
        assert plan.shots[1].continuity_mode == "last_frame"


def test_planner_long_duration_more_shots():
    plan = make_plan("anime city chase", 30)
    assert 3 <= len(plan.shots) <= 6
