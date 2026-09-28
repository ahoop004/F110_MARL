"""Grouped controllers share only identical, live occupancy distance fields."""
import gc
import weakref

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import distance_transform_edt

from agents.mpc.racing import _map_distance_field


def test_distance_fields_share_immutable_geometry_and_release_unused_maps(tmp_path):
    path = tmp_path / 'map.png'
    pixels = np.array([[0, 128, 255], [255, 255, 0]], dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    meta = {'resolution': .1, 'free_thresh': .196}
    field = _map_distance_field(path, meta)
    free = 1 - np.flipud(pixels.astype(np.float64)) / 255. < .196
    expected = (distance_transform_edt(free) - distance_transform_edt(~free)) * .1
    np.testing.assert_array_equal(field, expected)
    assert _map_distance_field(str(path), {**meta, 'origin': [1., 2., .5]}) is field
    with pytest.raises(ValueError):
        field[0, 0] = 0.

    for changed in [{'resolution': .2}, {'free_thresh': .6}, {'negate': 1}]:
        other = _map_distance_field(path, {**meta, **changed})
        assert other is not field
        assert not np.array_equal(other, field)

    ref = weakref.ref(field)
    del field
    gc.collect()
    assert ref() is None


def test_distance_field_reloads_replaced_map_without_changing_live_field(tmp_path):
    path = tmp_path / 'map.png'
    Image.fromarray(np.array([[0, 255], [255, 255]], dtype=np.uint8)).save(path)
    meta = {'resolution': .1}
    original = _map_distance_field(path, meta)
    before = original.copy()
    replacement = tmp_path / 'replacement.png'
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(replacement)
    replacement.replace(path)
    updated = _map_distance_field(path, meta)
    assert updated is not original
    assert not np.array_equal(updated, original)
    np.testing.assert_array_equal(original, before)
