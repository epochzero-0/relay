"""Tests for polybatch.core.tracker: state transitions, decide(), persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polybatch.core import tracker as trk
from polybatch.core.tracker import Tracker


def _load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_status_absent_key_is_not_submitted(tmp_path):
    t = Tracker(tmp_path / "t.json")
    assert t.status("nope") == trk.NOT_SUBMITTED


def test_mark_submitted_reflected_in_status_and_job_id(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_abc")
    assert t.status("k1") == trk.SUBMITTED
    assert t.job_id("k1") == "job_abc"


def test_mark_done_reflected_in_status(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_abc")
    t.mark_done("k1", succeeded=5, failed=0)
    assert t.status("k1") == trk.DONE
    # job_id carried forward from the prior submitted entry.
    assert t.job_id("k1") == "job_abc"


def test_mark_partial_failed_reflected_in_status(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_abc")
    t.mark_partial_failed("k1", succeeded=3, expected=5)
    assert t.status("k1") == trk.PARTIAL_FAILED


def test_mark_failed_reflected_in_status(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_abc")
    t.mark_failed("k1", "expired")
    assert t.status("k1") == trk.FAILED


def test_mark_submit_failed_reflected_in_status_and_clears_job_id(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_abc")
    t.mark_submit_failed("k1", "boom")
    assert t.status("k1") == trk.SUBMIT_FAILED
    assert t.job_id("k1") is None


@pytest.mark.parametrize(
    "state, expected",
    [
        (None, ("submit", None)),
        (trk.SUBMITTED, ("resume", "job_x")),
        (trk.DONE, ("skip", None)),
        (trk.PARTIAL_FAILED, ("submit", None)),
        (trk.FAILED, ("submit", None)),
        (trk.SUBMIT_FAILED, ("submit", None)),
    ],
)
def test_decide_for_each_state(tmp_path, state, expected):
    t = Tracker(tmp_path / "t.json")
    if state is None:
        pass  # key absent
    elif state == trk.SUBMITTED:
        t.mark_submitted("k1", "job_x")
    elif state == trk.DONE:
        t.mark_submitted("k1", "job_x")
        t.mark_done("k1", succeeded=1, failed=0)
    elif state == trk.PARTIAL_FAILED:
        t.mark_submitted("k1", "job_x")
        t.mark_partial_failed("k1", succeeded=1, expected=2)
    elif state == trk.FAILED:
        t.mark_submitted("k1", "job_x")
        t.mark_failed("k1", "some reason")
    elif state == trk.SUBMIT_FAILED:
        t.mark_submit_failed("k1", "some reason")

    assert t.decide("k1") == expected


def test_decide_submitted_without_job_id_falls_back_to_submit(tmp_path):
    # Reach SUBMITTED with a null job_id only via direct state manipulation
    # (mark_submitted always sets a job_id); simulate via internal dict since
    # decide() must handle this defensively per its docstring.
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_x")
    t._chunks["k1"]["job_id"] = None
    assert t.decide("k1") == ("submit", None)


def test_file_on_disk_is_valid_json_with_documented_shape_after_each_mark(tmp_path):
    path = tmp_path / "t.json"
    t = Tracker(path)

    t.mark_submitted("k1", "job_1")
    raw = _load_raw(path)
    assert raw["version"] == 1
    assert raw["chunks"]["k1"]["status"] == trk.SUBMITTED
    assert raw["chunks"]["k1"]["job_id"] == "job_1"

    t.mark_done("k1", succeeded=2, failed=1)
    raw = _load_raw(path)
    assert raw["chunks"]["k1"]["status"] == trk.DONE
    assert raw["chunks"]["k1"]["succeeded"] == 2
    assert raw["chunks"]["k1"]["failed"] == 1

    t.mark_partial_failed("k2", succeeded=1, expected=3)
    raw = _load_raw(path)
    assert raw["chunks"]["k2"]["status"] == trk.PARTIAL_FAILED
    assert raw["chunks"]["k2"]["succeeded"] == 1
    assert raw["chunks"]["k2"]["expected"] == 3

    t.mark_failed("k3", "expired")
    raw = _load_raw(path)
    assert raw["chunks"]["k3"]["status"] == trk.FAILED
    assert raw["chunks"]["k3"]["reason"] == "expired"

    t.mark_submit_failed("k4", "too big")
    raw = _load_raw(path)
    assert raw["chunks"]["k4"]["status"] == trk.SUBMIT_FAILED
    assert raw["chunks"]["k4"]["job_id"] is None
    assert raw["chunks"]["k4"]["reason"] == "too big"


def test_reload_from_disk_preserves_state(tmp_path):
    path = tmp_path / "t.json"
    t1 = Tracker(path)
    t1.mark_submitted("k1", "job_1")
    t1.mark_done("k1", succeeded=4, failed=0)

    t2 = Tracker(path)
    assert t2.status("k1") == trk.DONE
    assert t2.job_id("k1") == "job_1"


def test_reset_forgets_key(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_1")
    assert t.status("k1") == trk.SUBMITTED
    t.reset("k1")
    assert t.status("k1") == trk.NOT_SUBMITTED
    assert t.job_id("k1") is None


def test_reset_missing_key_is_a_noop(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.reset("does-not-exist")  # should not raise
    assert t.status("does-not-exist") == trk.NOT_SUBMITTED


def test_summary_reports_all_known_keys(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("k1", "job_1")
    t.mark_submitted("k2", "job_2")
    t.mark_done("k2", succeeded=1, failed=0)
    assert t.summary() == {"k1": trk.SUBMITTED, "k2": trk.DONE}


def test_submitted_keys_filters_by_prefix_and_status(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("run1_p1_chunk0", "job_1")
    t.mark_submitted("run1_p1_chunk1", "job_2")
    t.mark_done("run1_p1_chunk1", succeeded=1, failed=0)
    t.mark_submitted("run2_p1_chunk0", "job_3")

    keys = t.submitted_keys("run1_")
    assert keys == ["run1_p1_chunk0"]


def test_submitted_keys_requires_job_id(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submit_failed("run1_p1_chunk0", "boom")  # SUBMIT_FAILED, job_id None
    assert t.submitted_keys("run1_") == []


def test_next_pass_empty_tracker_returns_one(tmp_path):
    t = Tracker(tmp_path / "t.json")
    assert t.next_pass("run1_") == 1


def test_next_pass_with_existing_passes_returns_max_plus_one(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("run1_p1_chunk0", "job_1")
    t.mark_submitted("run1_p2_chunk0", "job_2")
    assert t.next_pass("run1_") == 3


def test_next_pass_ignores_keys_under_a_different_prefix(tmp_path):
    t = Tracker(tmp_path / "t.json")
    t.mark_submitted("run1_p1_chunk0", "job_1")
    t.mark_submitted("run2_p5_chunk0", "job_2")
    assert t.next_pass("run1_") == 2
