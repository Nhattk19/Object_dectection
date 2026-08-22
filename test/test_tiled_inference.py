from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tiled_inference import iter_tile_coordinates, tile_starts


def test_tile_starts_cover_final_edge_without_duplicate():
    assert tile_starts(1_500, 640, 0.20) == [0, 512, 860]


def test_small_image_uses_one_tile():
    assert list(iter_tile_coordinates(480, 600, 640, 0.20)) == [
        (0, 0, 600, 480)
    ]


def test_tiles_cover_all_corners():
    coordinates = list(iter_tile_coordinates(1_080, 1_920, 640, 0.20))
    assert coordinates[0] == (0, 0, 640, 640)
    assert coordinates[-1] == (1_280, 440, 1_920, 1_080)


@pytest.mark.parametrize("overlap", [-0.1, 1.0])
def test_invalid_overlap_is_rejected(overlap):
    with pytest.raises(ValueError):
        tile_starts(1_000, 640, overlap)
