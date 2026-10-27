import csv

import pytest

from drowsiness.feature_log import (
    BASE_COLUMNS,
    BLENDSHAPE_NAMES,
    FeatureCsvWriter,
    read_feature_csv,
)
from drowsiness.types import FrameFeatures, PerceptionOutput


def test_blendshape_list_is_complete_and_unique():
    assert len(BLENDSHAPE_NAMES) == 52 == len(set(BLENDSHAPE_NAMES))


FEATURES = [
    FrameFeatures(timestamp=0.0, face_found=False),
    FrameFeatures(
        timestamp=1 / 30,
        face_found=True,
        ear_left=0.312345678901234,
        ear_right=0.29,
        mar=0.05,
        pitch=-3.25,
        yaw=12.0,
        roll=None,  # solvePnP failure keeps the rest
        blendshapes={"eyeBlinkLeft": 0.125, "jawOpen": 0.75},
    ),
]


def test_round_trip_is_exact(tmp_path):
    path = tmp_path / "features.csv"
    with FeatureCsvWriter(path) as writer:
        for i, f in enumerate(FEATURES):
            writer.write(PerceptionOutput(f, None, 4.5), frame_index=i)
    assert read_feature_csv(path) == FEATURES


def test_header_and_cells(tmp_path):
    path = tmp_path / "features.csv"
    with FeatureCsvWriter(path, blendshapes=False) as writer:
        writer.write(PerceptionOutput(FEATURES[1], None, 4.5), frame_index=7)
    with path.open(encoding="utf-8", newline="") as f:
        header, row = list(csv.reader(f))
    assert tuple(header) == BASE_COLUMNS
    cells = dict(zip(header, row, strict=True))
    assert cells["frame_index"] == "7" and cells["face_found"] == "1"
    assert cells["roll"] == "" and cells["gaze_h"] == ""
    assert read_feature_csv(path)[0].blendshapes is None


def test_reader_rejects_other_csvs(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature CSV"):
        read_feature_csv(path)


def test_write_before_open_raises(tmp_path):
    with pytest.raises(RuntimeError):
        FeatureCsvWriter(tmp_path / "x.csv").write(PerceptionOutput(FEATURES[0], None, 0.0), 0)
