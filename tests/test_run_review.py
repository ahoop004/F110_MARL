import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.run_review import (
    load_run, combine, summarize_races, summarize_agents, latest_evaluations,
    aggregate_training_seeds, filter_clips, read_jsonl, discover_runs,
)
from analysis.plots import (
    learning_curves, outcome_plots, finish_time_plots, export_figures,
    evaluation_curves, seed_variation_plots,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def race(ep='ep0', *, success=True, mode='finite', times=(10., 20.)):
    return dict(episode_id=ep, race_mode=mode, map_id='circle_map',
        both_finished=success if mode == 'finite' else None, first_place=success if mode == 'finite' else None,
        sweep=False if mode == 'finite' else None, any_learner_collision_dnf=not success,
        mean_net_progress_laps=.2, duration_s=5., training_return=1.,
        reported_at_environment_steps=100,
        agents={f'car_{i}': dict(team='trainable', finished=success if mode == 'finite' else None,
            collision_dnf=not success, boundary_dnf=False, timeout=False,
            clean_finish_time_s=time if mode == 'finite' else None) for i, time in enumerate(times)})


def snapshot(path, seed=1):
    write_json(path/'config_snapshot.json', dict(provenance=dict(run_id='reused_id', seed=seed)))


def test_counts_missing_values_and_finisher_weighting(tmp_path):
    snapshot(tmp_path)
    rows = [race(times=(10., 20.)), race('ep1', times=(40., None)),
            race('ep2', mode='continuous')]
    write_lines(tmp_path/'race_metrics.jsonl', rows)
    run = load_run(tmp_path)
    summary = summarize_races(run.races)
    finite = summary.loc[summary.race_mode == 'finite'].set_index('metric')
    assert finite.loc['both_finished', 'denominator'] == 2
    assert finite.loc['clean_finish_time_s', 'value'] == pytest.approx(70/3)
    assert finite.loc['clean_finish_time_s', 'denominator'] == 3
    continuous = summary.loc[summary.race_mode == 'continuous'].set_index('metric')
    assert np.isnan(continuous.loc['both_finished', 'value'])
    assert continuous.loc['both_finished', 'denominator'] == 0
    assert summarize_agents(run.agents).loc[lambda x: x.race_mode == 'continuous'].query(
        "metric == 'finished'").value.isna().all()


def test_selection_snapshots_stay_separate_from_final_and_training_seed(tmp_path):
    snapshot(tmp_path, seed=7)
    history = [dict(environment_steps=step, policy_version=index,
        evaluation_protocol=dict(name='selection', seeds=[100], max_steps=8),
        episode_results=[{**race(f'ep{index}'), 'seed': 100, 'reported_at_environment_steps': step}])
        for index, step in enumerate([100, 200])]
    write_lines(tmp_path/'evaluation_history.jsonl', history)
    write_json(tmp_path/'evaluation_report.json', dict(protocol='final', seeds=[100],
        evaluation_provenance=dict(seed=100), checkpoint_provenance=dict(seed=7),
        episode_results=[dict(seed=100, race_record=race('final'))]))
    run = load_run(tmp_path)
    assert set(run.races.training_seed) == {7}
    chosen = latest_evaluations(run.races)
    assert set(chosen.evaluation_id) == {'selection_000001', 'standalone'}
    assert len(chosen) == 2
    assert run.evaluations.iloc[0].protocol_id == run.evaluations.iloc[1].protocol_id
    assert run.evaluations.iloc[2].protocol_id != run.evaluations.iloc[1].protocol_id
    assert len(evaluation_curves(run.races)) == 1
    plt.close('all')


def test_seed_aggregation_equal_weights_and_duplicate_rejection(tmp_path):
    runs = []
    for name, seed, count, success in [('a', 1, 1, True), ('b', 2, 9, False)]:
        path = tmp_path/name
        snapshot(path, seed)
        write_lines(path/'race_metrics.jsonl', [race(f'ep{i}', success=success) for i in range(count)])
        runs.append(load_run(path))
    summary = summarize_races(combine(runs, 'races'))
    groups = {r.metadata['run_key']: 'base' for r in runs}
    aggregated = aggregate_training_seeds(summary, groups).set_index('metric')
    assert aggregated.loc['both_finished', 'mean'] == .5  # Not 1/10 pooled episodes.
    assert aggregated.loc['both_finished', 'seed_std'] == pytest.approx(2**-.5)
    assert len(seed_variation_plots(aggregated.reset_index())) == 1
    plt.close('all')
    single = aggregate_training_seeds(summary, {runs[0].metadata['run_key']: 'base'})
    assert single.seed_std.isna().all()
    summary['training_seed'] = 1
    with pytest.raises(ValueError, match='one run/checkpoint'):
        aggregate_training_seeds(summary, groups)


def test_clip_outcomes_join_only_their_source_folder(tmp_path):
    runs = []
    for name, success in [('a', True), ('b', False)]:
        path = tmp_path/name
        snapshot(path)
        write_lines(path/'race_metrics.jsonl', [race(success=success)])
        write_lines(path/'behavior'/'clips.jsonl', [dict(clip_id='same', episode_id='ep0',
            kind='representative_race', retention_reasons=[], agent_ids=['car_0'], complete=True)])
        runs.append(load_run(path))
    clips = combine(runs, 'clips')
    assert len(clips) == 2
    assert filter_clips(clips, outcome='both_finished').run.tolist() == ['a']
    assert filter_clips(clips, outcome='any_learner_collision_dnf').run.tolist() == ['b']
    assert discover_runs(tmp_path) == [tmp_path/'a', tmp_path/'b']


def test_incomplete_log_tail_warns_but_corruption_is_not_hidden(tmp_path):
    path = tmp_path/'log.jsonl'
    path.write_text('{"a": 1}\n{"a":')
    with pytest.warns(UserWarning, match='incomplete last'):
        assert read_jsonl(path) == [{'a': 1}]
    path.write_text('{"a": invalid}\n{"a": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_plots_preserve_modes_and_export_without_measured_finish_times(tmp_path):
    snapshot(tmp_path)
    write_lines(tmp_path/'race_metrics.jsonl', [race(mode='continuous'),
        race('ep1', success=False, times=(None, None))])
    run = load_run(tmp_path)
    figures = learning_curves(run.races)
    assert set(figures) == {'learning_finite', 'learning_continuous'}
    figures.update(outcome_plots(summarize_races(run.races)))
    figures.update(finish_time_plots(run.agents))
    export_figures(figures, tmp_path/'figures')
    assert len(list((tmp_path/'figures').glob('*.pdf'))) == len(figures)
    plt.close('all')
