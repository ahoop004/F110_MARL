import numpy as np
import pytest

from utils.track_preview import _distance_to_wall, _track_width, TrackPreviewGeometry
from utils.map_loader import MapLoader


def test_wall_distance_uses_segment_interior_and_handles_duplicate_vertices():
    wall = np.array([[0., 0.], [10., 0.], [10., 0.], [10., 10.], [0., 10.]])
    np.testing.assert_allclose(_distance_to_wall(np.array([[5., 1.], [9., 5.]]), wall), [1., 1.])


@pytest.mark.parametrize('vertices', [1, 2, 5, 1000])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_compiled_wall_distance_matches_vectorized_reference(vertices, dtype):
    rng = np.random.default_rng(62)
    wall = rng.normal(size=(vertices, 2)).astype(dtype)
    if vertices > 2:
        wall[1] = wall[0]
    points = rng.normal(size=(20, 2)).astype(dtype)
    points[0] = wall[0]
    starts = np.asarray(wall, dtype=np.float64)
    vectors = np.roll(starts, -1, axis=0) - starts
    lengths = np.sum(vectors * vectors, axis=1)
    expected = []
    for point in points:
        relative = point - starts
        t = np.divide(np.sum(relative * vectors, axis=1), lengths,
                      out=np.zeros(len(starts)), where=lengths > 0)
        residual = relative - np.clip(t, 0., 1.)[:, None] * vectors
        expected.append(np.sqrt(np.min(np.sum(residual * residual, axis=1))))
    np.testing.assert_array_equal(_distance_to_wall(points, wall), np.asarray(expected, dtype=np.float32))


def test_corner_width_does_not_hit_remote_boundary():
    # The inner corner sits just above the horizontal cross-section. A normal
    # ray misses it and spans the outer enclosure's entire 12 m width.
    outer = np.array([[-2., -5.], [10., -5.], [10., 5.], [-2., 5.]])
    inner = np.array([[1., .1], [8., 3.], [8., 4.]])
    points = np.array([[0., -1.], [0., 0.], [0., 1.]])
    widths = _track_width(points, {0: outer, 1: inner}, closed=True)
    assert widths[1] == pytest.approx(2. + np.hypot(1., .1))
    # Global translation and rigid rotation must not change physical width.
    rotation = np.array([[0., -1.], [1., 0.]])
    transformed = _track_width(points @ rotation + 100,
        {0: outer @ rotation + 100, 1: inner @ rotation + 100}, closed=True)
    np.testing.assert_allclose(widths, transformed, atol=1e-5)


def test_wide_track_is_not_arbitrarily_clipped():
    angles = np.linspace(0., 2 * np.pi, 2048, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    width = _track_width(circle[::64] * 30., {0: circle * 20., 1: circle * 40.}, True)
    np.testing.assert_allclose(width, 20., atol=1e-3)


@pytest.mark.parametrize('name,low,high', [
    ('circle_map', 2.65, 2.75), ('L_map', .95, 1.05),
    ('Shanghai_map', 2., 5.), ('Spielberg_map', 2., 4.), ('Spa_map', 2., 4.),
])
def test_bundled_map_width_regressions(name, low, high):
    data = MapLoader().load(dict(map_dir='maps', map_bundle=name,
                                centerline_autoload=True, walls_autoload=True))
    geometry = TrackPreviewGeometry.build(data.centerline, data.walls)
    assert np.isfinite(geometry.width).all()
    assert geometry.width.min() > low
    assert geometry.width.max() < high
