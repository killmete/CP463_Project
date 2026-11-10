import pytest

from drowsiness.logic.signals import Condition, EventTimes, MedianSmoother, TimeWindow


def test_median_smoother_removes_single_spike_and_skips_none():
    s = MedianSmoother(0.1)
    assert s.update(0.0, 1.0) == 1.0
    assert s.update(1 / 30, 1.0) == 1.0
    assert s.update(2 / 30, 0.0) == 1.0  # one-frame spike
    assert s.update(3 / 30, None) is None
    assert s.update(1.0, 0.5) == 0.5  # old samples expired


def test_condition_holds_short_gaps_and_carries_adverse():
    c = Condition(max_gap_s=0.3, adverse=True)
    assert c.update(0.0, True) is True
    assert c.update(0.2, None) is True  # hold
    assert c.update(1.0, None) is True  # carry
    assert c.duration(1.0) == pytest.approx(1.0)
    assert c.update(1.1, False) is False
    assert c.duration(1.1) == 0.0


def test_condition_non_adverse_becomes_unknown_after_gap():
    c = Condition(max_gap_s=0.3, adverse=False)
    c.update(0.0, True)
    assert c.update(0.2, None) is True
    assert c.update(0.5, None) is None
    assert c.duration(0.5) == 0.0


def test_time_window_clips_and_skips_unknown():
    w = TimeWindow(10.0)
    w.add(0.0, 4.0, True)
    w.add(4.0, 6.0, None)
    w.add(6.0, 12.0, False)
    assert w.totals(12.0) == pytest.approx((2.0, 8.0))  # window [2, 12]


def test_event_times_window():
    e = EventTimes(5.0)
    for t in (0.0, 3.0, 6.0):
        e.add(t)
    assert e.count(6.0) == 2
