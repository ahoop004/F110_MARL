"""Track grip metadata stays explicit, validated, and separate from dynamics."""
from copy import deepcopy
import numpy as np
import pytest
import yaml
from PIL import Image

from env.map_config import resolve_map_runtime_config
from utils.map_loader import MapLoader, parse_surface_metadata


@pytest.fixture
def surface():
    return {
        'version': 1, 'id': 'test_surface', 'material': 'synthetic', 'condition': 'dry',
        'friction': {
            'mu': .9, 'reference_tire': 'test_tire', 'relative_std': .02,
            'calibration_id': 'synthetic-test-v1', 'calibration_status': 'uncalibrated',
            'source': 'Unit-test assumption, not a measured grip value',
        },
    }


def write_map(tmp_path, surface):
    Image.new('L', (16, 16), 255).save(tmp_path / 'test.png')
    metadata = {'image': 'test.png', 'resolution': .1, 'origin': [0, 0, 0]}
    if surface is not None:
        metadata['surface'] = surface
    (tmp_path / 'test.yaml').write_text(yaml.safe_dump(metadata))
    return {'map_dir': str(tmp_path), 'map': 'test.yaml', 'map_yaml': 'test.yaml',
            'centerline_autoload': False, 'walls_autoload': False}


def test_old_bundles_and_unknown_grip_need_no_added_data(tmp_path):
    assert parse_surface_metadata(None) is None
    assert parse_surface_metadata({'version': 1, 'id': 'unmeasured'}) == {
        'version': 1, 'id': 'unmeasured'}
    cfg = write_map(tmp_path, None)
    data = MapLoader().load(cfg)
    assert 'surface' not in data.metadata
    assert 'surface' not in resolve_map_runtime_config(cfg).metadata


def test_surface_metadata_survives_both_load_paths_without_changing_geometry(tmp_path, surface):
    config = write_map(tmp_path, surface)
    loader = MapLoader()
    first = loader.load(config)
    reference_mask = first.track_mask.copy()
    first.metadata['surface']['friction']['mu'] = .1
    second = loader.load(config)
    assert second.metadata['surface'] == surface  # cached nested data is detached
    assert second.track_mask is first.track_mask
    np.testing.assert_array_equal(second.track_mask, reference_mask)
    for data in (None, second):
        runtime = resolve_map_runtime_config(config, data)
        assert runtime.metadata['surface'] == surface
        runtime.metadata['surface']['friction']['mu'] = .2
        assert second.metadata['surface']['friction']['mu'] == .9
    # Surface measurements are metadata, not vehicle parameter overrides.
    assert 'vehicle_params' not in config


@pytest.mark.parametrize('key,value', [
    ('mu', -1), ('mu', np.nan), ('mu', np.inf), ('mu', True), ('mu', '.9'),
    ('mu', 10**400),
    ('relative_std', -1), ('relative_std', None), ('reference_tire', ''),
    ('calibration_id', ''), ('calibration_status', 'verified'), ('source', ''),
    ('zones', []),
])
def test_invalid_friction_is_rejected(surface, key, value):
    surface['friction'][key] = value
    with pytest.raises(ValueError, match='surface.friction'):
        parse_surface_metadata(surface)


@pytest.mark.parametrize('value', [[], {}, {'version': True, 'id': 'test'},
    {'version': 2, 'id': 'test'}, {'version': 1, 'id': 'test', 'zones': []}])
def test_invalid_surface_schema_is_rejected(value):
    with pytest.raises(ValueError, match='surface'):
        parse_surface_metadata(value)


def test_grip_requires_reference_tire_and_calibration_provenance(tmp_path, surface):
    del surface['friction']['reference_tire']
    cfg = write_map(tmp_path, surface)
    for load in (lambda: MapLoader().load(cfg), lambda: resolve_map_runtime_config(cfg)):
        with pytest.raises(ValueError, match='reference_tire'):
            load()


def test_parser_does_not_mutate_inputs(surface):
    original = deepcopy(surface)
    parsed = parse_surface_metadata(surface)
    parsed['friction']['mu'] = 1.2
    assert surface == original
