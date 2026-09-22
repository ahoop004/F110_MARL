"""Read P0 metrics and P2 clip indices without loading models or trajectories.

Folder paths are identities: run_id is user supplied and may be reused. Unknown
facts remain missing. Evaluation snapshots are never pooled across checkpoints.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys
import warnings

import numpy as np
import pandas as pd


ARTIFACTS = ('config_snapshot.json', 'race_metrics.jsonl', 'evaluation_report.json',
             'evaluation_history.jsonl', 'update_metrics.csv')
RATE_METRICS = ('both_finished', 'first_place', 'sweep', 'at_least_one_finished',
                'any_learner_collision_dnf')
VALUE_METRICS = ('rank_score', 'mean_net_progress_laps', 'mean_learner_laps',
                 'duration_s', 'training_return', 'own_collision_dnf_count',
                 'opponent_collision_dnf_count', 'own_boundary_dnf_count',
                 'opponent_boundary_dnf_count')
RACE_GROUPS = ['run_key', 'run', 'training_seed', 'phase', 'protocol', 'protocol_id',
               'evaluation_id', 'race_mode']


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def read_jsonl(path):
    """Ignore only an unfinished last line from an actively written log."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not line.endswith('\n') and not stream.read():
                    warnings.warn(f'{path}:{number}: incomplete last record skipped', stacklevel=2)
                    break
                raise
    return rows


def discover_runs(root):
    root = Path(root).resolve()
    directories = {p.parent for name in ARTIFACTS for p in root.rglob(name)}
    return sorted(directories)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _protocol(report, provenance, *, history=False):
    """Fingerprint recorded evaluation conditions, including ordered map/seed pairs."""
    protocol = report.get('evaluation_protocol', {}) if history else report
    physics = dict(provenance.get('physics_contract', {}))
    friction = physics.get('friction_protocol', {})
    if friction:
        physics['friction_protocol'] = {k: v for k, v in friction.items() if k != 'train'}
    assignments = []
    for entry in report.get('episode_results', []):
        race = entry.get('race_record', entry)
        assignments.append((entry.get('seed', race.get('environment_seed')),
                            race.get('map_id', entry.get('map_bundle'))))
    return _digest(dict(name=protocol.get('name', report.get('protocol', 'unknown')),
        seeds=protocol.get('seeds'), map_seeds=assignments,
        max_steps=protocol.get('max_steps', provenance.get('max_steps')),
        target_laps=protocol.get('target_laps', provenance.get('target_laps')),
        timestep_s=protocol.get('timestep_s'), action_repeat=protocol.get('action_repeat'),
        physics=physics, maps=provenance.get('map_protocols'),
        track_limits=provenance.get('track_limits'), behaviors=provenance.get('behavior_contracts')))


@dataclass
class RunData:
    path: Path
    metadata: dict
    races: pd.DataFrame
    agents: pd.DataFrame
    updates: pd.DataFrame
    evaluations: pd.DataFrame
    clips: pd.DataFrame


def load_run(directory, *, label=None, dataset_dirs=()):
    path = Path(directory).resolve()
    snapshot = read_json(path/'config_snapshot.json')
    report = read_json(path/'evaluation_report.json')
    provenance = snapshot.get('provenance') or report.get('checkpoint_provenance') or {}
    config = snapshot.get('config', {})
    identity = dict(run_key=str(path), run=label or path.name,
                    training_seed=provenance.get('seed', config.get('experiment', {}).get('seed')))
    meta = dict(**identity, run_id=provenance.get('run_id'),
        scenario=provenance.get('scenario_name', config.get('experiment', {}).get('name')),
        algorithm=provenance.get('algorithm'),
        train_maps=provenance.get('map_split', {}).get('train'),
        eval_maps=provenance.get('map_split', {}).get('eval'),
        pretrained_checkpoint=provenance.get('pretrained_actor', {}).get('path'),
        pretrained_sha256=provenance.get('pretrained_actor', {}).get('sha256'),
        checkpoint=report.get('checkpoint'), checkpoint_sha256=report.get('checkpoint_sha256'),
        resolved_config_sha256=provenance.get('resolved_config_sha256'),
        protocol=report.get('protocol'), provenance_mismatches=report.get('provenance_mismatches'),
        race_records_available=(path/'race_metrics.jsonl').exists())
    race_rows, agent_rows, evaluation_rows = [], [], []

    def add_race(row, context):
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat.update(identity, **context)
        flat.setdefault('race_mode', 'unknown')
        flat.setdefault('map_id', 'unknown')
        # Do not invent a global completion timestamp from a worker-local count.
        flat['environment_steps'] = (context.get('environment_steps') if context['phase'] == 'evaluation'
                                     else row.get('reported_at_environment_steps'))
        flat['episode_id'] = row.get('episode_id')
        own = [a for a in row.get('agents', {}).values() if a.get('team') == 'trainable']
        times = [a['clean_finish_time_s'] for a in own if a.get('clean_finish_time_s') is not None]
        flat['finish_time_sum_s'], flat['finish_time_samples'] = sum(times), len(times)
        race_rows.append(flat)
        for aid, agent in row.get('agents', {}).items():
            agent_rows.append({**flat, **{k: v for k, v in agent.items() if not isinstance(v, (dict, list))},
                               'agent_id': aid})

    for row in read_jsonl(path/'race_metrics.jsonl'):
        add_race(row, dict(phase='training', protocol='training', protocol_id='training',
                          evaluation_id='training'))
    history_rows = read_jsonl(path/'evaluation_history.jsonl')
    for index, entry in enumerate(history_rows):
        p = entry.get('evaluation_protocol', {})
        context = dict(phase='evaluation', protocol=p.get('name', 'selection'),
            protocol_id=_protocol(entry, provenance, history=True),
            evaluation_id=f'selection_{index:06d}', environment_steps=entry.get('environment_steps'))
        evaluation_rows.append({**identity, **context, 'checkpoint': None,
            'policy_version': entry.get('policy_version'), 'is_best': entry.get('is_best'),
            'seeds': p.get('seeds'), 'reported_race_count': entry.get('race_count', entry.get('episodes'))})
        for row in entry.get('episode_results', []):
            add_race(row, context)
    if report:
        context = dict(phase='evaluation', protocol=report.get('protocol', 'unknown'),
            protocol_id=_protocol(report, report.get('evaluation_provenance', {})),
            evaluation_id='standalone', environment_steps=None)
        evaluation_rows.append({**identity, **context, 'checkpoint': report.get('checkpoint'),
            'checkpoint_sha256': report.get('checkpoint_sha256'), 'seeds': report.get('seeds'),
            'reported_race_count': report.get('summary', {}).get('race_count')})
        for entry in report.get('episode_results', []):
            if 'race_record' in entry:
                add_race(entry['race_record'], context)
    updates_path = path/'update_metrics.csv'
    try:
        updates = pd.read_csv(updates_path) if updates_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        updates = pd.DataFrame()
    for key, value in identity.items():
        updates[key] = value
    # Read small clip indices only; trajectories remain on disk until requested.
    clip_rows = []
    for dataset in sorted({(path/'behavior').resolve(), *(Path(d).resolve() for d in dataset_dirs)}):
        latest = {}
        for clip in read_jsonl(dataset/'clips.jsonl'):
            latest[clip['clip_id']] = clip
        for clip in latest.values():
            clip_rows.append({**clip, **identity, 'dataset_dir': str(dataset)})
    clips = pd.DataFrame(clip_rows)
    races = pd.DataFrame(race_rows)
    if not clips.empty and not races.empty:
        facts = races.loc[races.phase == 'training', ['episode_id', 'race_mode', *[
            k for k in ('both_finished', 'first_place', 'sweep', 'any_learner_collision_dnf') if k in races]]]
        if facts.episode_id.duplicated().any():
            raise ValueError(f'Duplicate training episode IDs in {path}')
        clips = clips.merge(facts, on='episode_id', how='left', validate='many_to_one')
    meta.update(training_races=sum(r.get('phase') == 'training' for r in race_rows),
                evaluation_snapshots=len(evaluation_rows), recorded_clips=len(clip_rows))
    return RunData(path, meta, races, pd.DataFrame(agent_rows), updates,
                   pd.DataFrame(evaluation_rows), clips)


def combine(runs, table):
    frames = [getattr(run, table) for run in runs if not getattr(run, table).empty]
    if not frames:
        return pd.DataFrame()
    # Keep unavailable fields, without inferring dtypes from all-null columns.
    columns = list(dict.fromkeys(c for frame in frames for c in frame.columns))
    measured = [frame.dropna(axis=1, how='all') for frame in frames]
    return pd.concat(measured, ignore_index=True, sort=False).reindex(columns=columns)


def latest_evaluations(races):
    """Latest selection checkpoint and standalone report, kept as separate snapshots."""
    if races.empty:
        return races.copy()
    rows = races.loc[races.phase == 'evaluation'].copy()
    if rows.empty:
        return rows
    latest = rows.groupby(['run_key', 'protocol'], dropna=False).evaluation_id.transform('max')
    return rows.loc[rows.evaluation_id == latest].copy()


def summarize_races(races, *, by_map=True):
    """Long table with measured denominators. Finisher times weight cars, not races."""
    if races.empty:
        return pd.DataFrame()
    keys = RACE_GROUPS + (['map_id'] if by_map else [])
    records = []
    for values, rows in races.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, values))
        if not by_map:
            base['map_id'] = 'ALL'
        for metric in (*RATE_METRICS, *VALUE_METRICS):
            values = pd.to_numeric(rows.get(metric, pd.Series(dtype=float)), errors='coerce').dropna()
            records.append({**base, 'metric': metric, 'value': values.mean(),
                'numerator': values.sum() if len(values) else np.nan,
                'denominator': len(values), 'race_count': len(rows),
                'statistic': 'rate' if metric in RATE_METRICS else 'mean'})
        n = rows.finish_time_samples.sum()
        total = rows.finish_time_sum_s.sum()
        records.append({**base, 'metric': 'clean_finish_time_s', 'value': total/n if n else np.nan,
            'numerator': total if n else np.nan, 'denominator': n, 'race_count': len(rows),
            'statistic': 'mean_over_learner_finishers'})
    return pd.DataFrame(records)


def summarize_agents(agents):
    if agents.empty:
        return pd.DataFrame()
    keys = RACE_GROUPS + ['map_id', 'team', 'agent_id']
    records = []
    for values, rows in agents.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, values))
        for metric in ('finished', 'collision_dnf', 'boundary_dnf', 'timeout', 'clean_finish_time_s'):
            data = pd.to_numeric(rows.get(metric, pd.Series(dtype=float)), errors='coerce').dropna()
            records.append({**base, 'metric': metric, 'value': data.mean(),
                'numerator': data.sum() if len(data) else np.nan, 'denominator': len(data),
                'race_count': len(rows)})
    return pd.DataFrame(records)


def aggregate_training_seeds(summary, comparison_groups):
    """Explicit arm assignment; one matched snapshot per training seed and arm.

    An equal-weight mean and sample SD across independent training seeds, never
    an episode-pooled estimate or a confidence interval. Unknown seeds rejected.
    Protocol fingerprints separate recorded evaluation conditions, but do not
    prove that opponent policies or unrecorded conditions match.
    """
    if summary.empty or not comparison_groups:
        return pd.DataFrame()
    rows = summary.loc[summary.run_key.isin(comparison_groups)].copy()
    rows['comparison_group'] = rows.run_key.map(comparison_groups)
    if rows.training_seed.isna().any():
        raise ValueError('Training seed is missing; cannot estimate seed variation.')
    keys = ['comparison_group', 'phase', 'protocol', 'protocol_id', 'race_mode', 'map_id', 'metric']
    if rows.duplicated(keys + ['training_seed']).any():
        raise ValueError('Choose one run/checkpoint per training seed, group, map, and protocol.')
    records = []
    for values, part in rows.groupby(keys, dropna=False, sort=False):
        measured = part.value.dropna()
        records.append({**dict(zip(keys, values)), 'mean': measured.mean(),
            'seed_std': measured.std(ddof=1), 'seed_count': len(measured),
            'available_seed_count': len(part), 'race_count': part.race_count.sum()})
    return pd.DataFrame(records)


def filter_clips(clips, *, run=None, map_id=None, kind=None, event=None, agent=None,
                 outcome=None, checkpoint=None, policy_version=None):
    rows = clips.copy()
    if rows.empty:
        return rows
    for column, value in [('run', run), ('map_id', map_id), ('kind', kind)]:
        if value is not None:
            rows = rows.loc[rows[column] == value]
    if event is not None:
        rows = rows.loc[rows.retention_reasons.map(lambda reasons: event in reasons)]
    if agent is not None:
        rows = rows.loc[rows.agent_ids.map(lambda ids: agent in ids)]
    if checkpoint is not None:
        # Do not assign a run's final checkpoint to earlier training clips.
        match = pd.Series(False, index=rows.index)
        for column in ('checkpoint', 'checkpoint_sha256'):
            if column in rows:
                match |= rows[column].eq(str(checkpoint))
        rows = rows.loc[match]
    if policy_version is not None:
        start = pd.to_numeric(rows.get('policy_version_start', pd.Series(index=rows.index, dtype=float)), errors='coerce')
        end = pd.to_numeric(rows.get('policy_version_end', pd.Series(index=rows.index, dtype=float)), errors='coerce')
        rows = rows.loc[start.le(policy_version) & end.ge(policy_version)]
    if outcome is not None:
        if outcome not in RATE_METRICS:
            raise ValueError(f'Choose an outcome from {RATE_METRICS}')
        rows = rows.loc[rows.get(outcome, pd.Series(False, index=rows.index)).eq(True)]
    return rows


def replay_command(clip, repo_root, *, speed=1):
    return shlex.join([sys.executable, str(Path(repo_root)/'replay.py'), str(clip['dataset_dir']),
                       '--clip', clip['clip_id'], '--speed', str(speed)])
