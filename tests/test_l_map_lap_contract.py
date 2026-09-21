"""L-map progress reward must coexist with working measured-lap tracking."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.scenario import load_and_expand_scenario
from core.setup import create_training_setup
from env.centerline_state import LapTracker, validate_finish_line
from env.collision_state import RaceLifecycle
from env.types import AgentRaceStatus

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'maps/L_map'


@pytest.mark.parametrize('finish_on_laps', [False, True])
def test_l_map_full_circuits_count_and_time_laps(finish_on_laps):
    metadata = yaml.safe_load((MAP / 'L_map.yaml').read_text())
    points = np.loadtxt(MAP / 'L_map_centerline.csv', delimiter=',', skiprows=1, usecols=(0, 1))
    line = validate_finish_line(metadata['annotations']['finish_line'], centerline=points)
    assert line['segment_length'] == pytest.approx(1.)
    midpoint = (line['start'] + line['end']) / 2
    start = np.argmin(np.linalg.norm(points - (midpoint - .3 * line['direction']), axis=1))
    points = np.roll(points, -start, axis=0)
    lifecycle = RaceLifecycle(['car_0'], 3, finish_on_laps=finish_on_laps)
    tracker = LapTracker(['car_0'], line, lifecycle, count_initial_crossing_as_lap=False)
    tracker.reset(points[:1, 0], points[:1, 1])
    # Initial partial circuit starts the clock; following circuits count laps.
    counts, times = [], []
    for step, point in enumerate(np.tile(points, (4, 1)), 1):
        crossed = tracker.update(point[:1], point[1:2], np.ones(1), np.zeros(1), step=step)
        record = lifecycle.records['car_0']
        if crossed['car_0']:
            counts.append(record.lap_count)
            times.append(record.lap_time_steps)
    assert counts == [1, 2, 3]
    assert times == [len(points)] * 3
    assert record.status == (AgentRaceStatus.FINISHED if finish_on_laps else AgentRaceStatus.ACTIVE)


@pytest.mark.parametrize('mode', ['train', 'eval'])
def test_real_pretraining_setup_has_lap_tracker_and_rejects_missing_gate(mode):
    path = ROOT / 'scenarios/ppo_lap_completion_pretrain.yaml'
    scenario = load_and_expand_scenario(str(path))
    env, _, _ = create_training_setup(scenario, mode=mode, scenario_dir=path.parent)
    try:
        assert env._lap_tracker is not None
        assert env._require_finish_line
        assert env.lifecycle.finish_on_laps == (mode == 'eval')
        _, infos = env.reset(seed=42)
        assert infos['car_0']['finish_line_version'] == 1
        missing = deepcopy(env.map_meta)
        missing['annotations'].pop('finish_line')
        with pytest.raises(ValueError, match='requires a finish_line'):
            env._configure_lap_tracker(missing)
    finally:
        env.close()
