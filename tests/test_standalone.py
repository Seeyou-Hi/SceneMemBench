import json
import sys
import numpy as np
import pytest
from scenemembench.demo import create
from scenemembench.cli import main
from scenemembench.continuation.dataset import load_context_bundle
from scenemembench.continuation.runner import run_context_continuation
from scenemembench.continuation.contracts import ActionCommand, PrefillResult
from scenemembench.models.toy import ToyAdapter
from scenemembench.metrics.evaluate import pose_matrix, geometry_pair, evaluate
from scenemembench.metrics.selection import (
    PoseToleranceConfig,
    terminal_pose_candidates,
)
from scenemembench.judges import vqa


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    create(root)
    return root


@pytest.fixture
def video(demo, tmp_path):
    out = tmp_path / "run"
    main(
        [
            "run",
            "--model",
            "toy",
            "--case",
            str(demo / "reversed_rotation"),
            "--output",
            str(out),
            "--device",
            "cpu",
        ]
    )
    report = json.loads((out / "run_report.json").read_text())
    assert report["continuation_verified"]
    assert report["continuation_frame_count"] == 16
    return out / "continuation.mp4"


def test_no_old_project_imports():
    assert not any(
        n.split(".")[0] in {"worldfoundry", "PlayWorld", "playworld"}
        for n in sys.modules
    )


def test_distributed_worker_leaves_rank_zero_outputs_untouched(demo, tmp_path):
    from argparse import Namespace
    from scenemembench.cli import _run

    output = tmp_path / "shared"
    output.mkdir()
    sentinel = output / "rank-zero-report.json"
    sentinel.write_text("rank zero owns these files")

    class Collective:
        def broadcast_object_list(self, payload, src):
            # Rank zero checked the output before creating its first report.
            payload[0] = False

    _run(
        Namespace(case=str(demo / "reversed_rotation"), config=None,
                  output=str(output), model="toy", device="cpu",
                  prompt=None, seed=42),
        rank=1, distributed=Collective(),
    )
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "rank zero owns these files"


@pytest.mark.parametrize("axis", [3, 4, 5])
def test_thirty_degree_window_includes_boundary_and_excludes_earlier_loop(axis):
    rows = []
    for angle in [0, 60, 31, 30, 10, 0]:
        raw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        raw[axis] = angle
        rows.append(pose_matrix({"primary_pose": raw}))
    candidates = terminal_pose_candidates(
        rows, eval_start_source_frame=100, config=PoseToleranceConfig(max_rotation_deg=30)
    )
    assert [c.source_frame_id for c in candidates] == [103, 104, 105]


def test_geometry_scale_and_invalid_pixels():
    ref = np.ones((20, 20), dtype=float) * 2
    gen = ref * 3
    k = np.array([[20, 0, 10], [0, 20, 10], [0, 0, 1.0]])
    score = geometry_pair(ref, gen, k)
    assert score["endpoint_si_absrel"] == pytest.approx(0)
    assert score["camera_aware_normal_cosine"] == pytest.approx(1)
    gen[:10] = np.nan
    assert geometry_pair(ref, gen, k)["endpoint_si_absrel"] == pytest.approx(0.5)


def test_query_pixels_not_required_for_continuation(demo, tmp_path):
    import shutil

    case = tmp_path / "case"
    shutil.copytree(demo / "reversed_rotation", case)
    rows = json.loads((case / "pose/poses.json").read_text())
    for row in rows[33:]:
        (case / row["rgb"]).unlink()
    (case / "eval.mp4").unlink()
    main(
        [
            "run",
            "--model",
            "toy",
            "--case",
            str(case),
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ]
    )
    assert (tmp_path / "out/continuation.mp4").exists()


def test_rejected_prefill_does_not_generate_and_resets(demo):
    class Rejected(ToyAdapter):
        ended = False

        def prefill_context(self, context):
            return PrefillResult("rejected", ())

        def step(self, action):
            raise AssertionError("must not generate")

        def end_case(self):
            self.ended = True

    adapter = Rejected()
    with pytest.raises(ValueError, match="rejected"):
        run_context_continuation(
            adapter,
            load_context_bundle(demo / "reversed_rotation"),
            [ActionCommand(0, "turn_right")],
        )
    assert adapter.ended


def test_seen_and_novel_share_fixed_generated_endpoint(demo, video, tmp_path):
    seen = evaluate(
        demo / "reversed_rotation", video, tmp_path / "seen.json", task="seen"
    )
    novel = evaluate(
        demo / "new_viewpoint", video, tmp_path / "novel.json", task="novel"
    )
    assert seen["generated_endpoint"] == novel["generated_endpoint"] == 15
    assert novel["reference_source_frame"] == 48
    assert seen["selection"]["max_rotation_deg"] is None
    assert seen["selection"]["rule"] == "seen-query-search-v1"
    assert min(seen["selection"]["candidate_source_ids"]) == 33
    assert max(seen["selection"]["candidate_source_ids"]) == 48
    assert seen["selection"]["selected_rotation_deg"] >= 0
    bounded = evaluate(
        demo / "reversed_rotation", video, tmp_path / "bounded.json",
        task="seen", max_rotation_deg=30,
    )
    assert min(bounded["selection"]["candidate_source_ids"]) > 33
    assert bounded["selection"]["rule"] == "seen-terminal-angular-v1"


def test_stale_straight_query_is_rejected(demo, video, tmp_path):
    import shutil

    case = tmp_path / "case"
    shutil.copytree(demo / "forward_look_back", case)
    m = json.loads((case / "manifest.json").read_text())
    m["split"]["eval"] = [32, 49]
    (case / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="eval_turnaround"):
        evaluate(case, video, tmp_path / "score.json", task="novel")


def test_vqa_separate_motion_and_independent_dimensions(video, tmp_path):
    calls = []

    def client(prompt, paths):
        calls.append((prompt, paths))
        answer = (
            {"status": "passed", "reason": "camera moves"}
            if len(paths) == 12
            else {"score": 0.6, "reason": "partial correspondence"}
        )
        return answer, {"answer": answer}

    result = vqa.evaluate(
        video,
        tmp_path / "vqa",
        revisit="reversed_rotation",
        model="mock",
        client=client,
    )
    assert result["status"] == "scored"
    assert [len(paths) for _, paths in calls] == [12, 3, 3, 3]
    assert all("camera moves" not in prompt for prompt, _ in calls[1:])
    assert set(result["scores"]) == {"subject", "appearance", "geometry"}


def test_vqa_rejected_gate_blocks_all_scores(video, tmp_path):
    calls = []

    def client(prompt, paths):
        calls.append(prompt)
        return {"status": "rejected", "reason": "static"}, {}

    result = vqa.evaluate(
        video, tmp_path / "vqa", revisit="full_rotation", model="mock", client=client
    )
    assert len(calls) == 1 and result["scores"] is None


def test_vqa_dry_run_makes_no_requests(video, tmp_path):
    def client(*args):
        raise AssertionError("unexpected API call")

    result = vqa.evaluate(
        video, tmp_path / "dry", revisit="new_viewpoint", dry_run=True, client=client
    )
    assert result["status"] == "prepared_not_scored"


def test_video_mini_prepares_only_context_and_rejects_truncation(demo, tmp_path):
    import shutil
    from scenemembench.prepare import prepare

    case = tmp_path / 'video-case'
    shutil.copytree(demo / 'reversed_rotation', case)
    shutil.rmtree(case / 'rgb')
    manifest = json.loads((case / 'manifest.json').read_text())
    manifest['rgb_storage'] = {'mode': 'decode_context_video'}
    (case / 'manifest.json').write_text(json.dumps(manifest))
    assert prepare(case)['status'] == 'prepared'
    assert load_context_bundle(case).context_frame_count == 33
    assert len(list((case / 'rgb').glob('*.png'))) == 33
    assert not (case / 'rgb/000033.png').exists()
    assert prepare(case)['status'] == 'already_prepared'

    broken = tmp_path / 'truncated-case'
    shutil.copytree(case, broken, ignore=shutil.ignore_patterns('rgb', 'context_prepared.json'))
    manifest['split']['context'] = [0, 34]
    (broken / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Expected 34 Context frames'):
        prepare(broken)
    assert not (broken / 'rgb').exists()
    assert not list(broken.glob('.context-rgb-*'))
