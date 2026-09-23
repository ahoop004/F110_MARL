"""Read P0 metrics and P2 clip indices without loading models or trajectories.

Folder paths are identities: run_id is user supplied and may be reused. Unknown
facts remain missing. Evaluation snapshots are never pooled across checkpoints.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shlex
import sys
import warnings

import numpy as np
import pandas as pd


ARTIFACTS = ('config_snapshot.json', 'race_metrics.jsonl', 'evaluation_report.json',
             'evaluation_history.jsonl', 'update_metrics.csv', 'team_metrics.jsonl', 'updates.jsonl', 'evaluation_races.jsonl')
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
    physics = dict(provenance.get('physics_contract') or {})
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
    team_races: pd.DataFrame = field(default_factory=pd.DataFrame)
    recording_windows: pd.DataFrame = field(default_factory=pd.DataFrame)


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
            evaluation_id=entry.get('evaluation_id', f'selection_{index:06d}'), environment_steps=entry.get('environment_steps'))
        evaluation_rows.append({**identity, **context, 'checkpoint': entry.get('checkpoint'),
            'checkpoint_sha256': entry.get('checkpoint_sha256'),
            'policy_version': entry.get('policy_version'), 'is_best': entry.get('is_best'),
            'seeds': p.get('seeds'), 'reported_race_count': entry.get('race_count', entry.get('episodes'))})
        for row in entry.get('episode_results', []):
            add_race(row, context)
    if report:
        context = dict(phase='evaluation', protocol=report.get('protocol', 'unknown'),
            protocol_id=_protocol(report, report.get('evaluation_provenance', {})),
            evaluation_id='standalone', environment_steps=report.get('environment_steps'))
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
    team_rows = []
    teams = sorted(set(config.get('environment', {}).get('agent_teams', {}).values()))
    if config.get('two_team', {}).get('enabled'):
        updates = pd.DataFrame(read_jsonl(path/'updates.jsonl'))
        for key, value in identity.items():
            updates[key] = value
        def add_team_race(row, phase, **context):
            for team in teams:
                team_rows.append({**identity, 'team': team, 'phase': phase,
                    'training_seed': row.get('checkpoint_training_seed', identity['training_seed']),
                    **{k: row.get(k) for k in ('episode_id', 'map_id', 'environment_id', 'environment_episode',
                        'environment_steps', 'completed', 'budget_cut', 'policy_version_start', 'policy_version_end',
                        'checkpoint', 'checkpoint_sha256', 'seed', 'team_policy_versions')},
                    **context,
                    **{k.removeprefix(team+'/'): v for k, v in row.items() if k.startswith(team+'/')}})
        for record in read_jsonl(path/'team_metrics.jsonl'):
            add_team_race({k.removeprefix('selfplay/'): v for k, v in record.items()}, 'training')
        # Old self-play logs have rounds/maps/seeds, but no checkpoint identity.
        # Leave those fields missing instead of assigning a later saved pair.
        rounds = {}
        summaries = {r.get('selfplay_eval/evaluation_id', f"eval_{r['selfplay_eval/round']:06d}"): r
                     for r in read_jsonl(path/'evaluation_metrics.jsonl')}
        for row in read_jsonl(path/'evaluation_races.jsonl'):
            evaluation_id = row.get('evaluation_id', f"eval_{row['round']:06d}")
            rounds.setdefault(evaluation_id, []).append(row)
        for evaluation_id, records in rounds.items():
            first = records[0]
            protocol = first.get('protocol', 'unknown')
            physics = dict(provenance.get('physics_contract') or {})
            if 'friction_protocol' in physics:
                physics['friction_protocol'] = {k: v for k, v in physics['friction_protocol'].items() if k != 'train'}
            pid = _digest(dict(protocol=first.get('evaluation_protocol'), physics=physics,
                maps=provenance.get('map_protocols'),
                assignments=[(r.get('seed'), r.get('map_id', r.get('map'))) for r in records]))
            context = dict(evaluation_id=evaluation_id, protocol=protocol, protocol_id=pid)
            evaluation_rows.append({**identity, **context, 'phase': 'evaluation', 'algorithm': 'mappo_two_team',
                'training_seed': first.get('checkpoint_training_seed', identity['training_seed']),
                'is_best': summaries.get(evaluation_id, {}).get('selfplay_eval/is_best'),
                'checkpoint': first.get('checkpoint'), 'checkpoint_sha256': first.get('checkpoint_sha256'),
                'environment_steps': first.get('environment_steps'), 'reported_race_count': len(records),
                'seeds': [r.get('seed') for r in records], 'team_policy_versions': first.get('team_policy_versions')})
            for row in records:
                add_team_race(dict(row, map_id=row.get('map_id', row.get('map'))), 'evaluation', **context)
    team_races = pd.DataFrame(team_rows)
    # Read small clip indices only; trajectories remain on disk until requested.
    clip_rows, window_rows = [], []
    for dataset in sorted({(path/'behavior').resolve(), (path/'evaluation_behavior').resolve(),
                           *(Path(d).resolve() for d in dataset_dirs),
                           *((Path(d)/'evaluation').resolve() for d in dataset_dirs)}):
        dataset_metadata = read_json(dataset/'metadata.json')
        for window in dataset_metadata.get('recording_windows', []):
            window_rows.append({**identity, 'dataset_dir': str(dataset),
                'phase': dataset_metadata.get('phase', 'training'), **window})
        latest = {}
        for clip in read_jsonl(dataset/'clips.jsonl'):
            latest[clip['clip_id']] = clip
        for clip in latest.values():
            clip_rows.append({'phase': 'training', **clip, **identity, 'dataset_dir': str(dataset)})
    clips = pd.DataFrame(clip_rows)
    races = pd.DataFrame(race_rows)
    if not clips.empty and not races.empty:
        facts = races.loc[races.episode_id.notna(), ['episode_id', 'race_mode', *[
            k for k in ('both_finished', 'first_place', 'sweep', 'any_learner_collision_dnf') if k in races]]]
        if facts.episode_id.duplicated().any():
            raise ValueError(f'Duplicate episode IDs in {path}')
        clips = clips.merge(facts, on='episode_id', how='left', validate='many_to_one')
    if not clips.empty and not team_races.empty:
        # Keep each team's outcome distinct; win here means completion/progress,
        # not first place. Partial races have no completed-race outcome.
        for team in teams:
            facts = team_races.loc[(team_races.team == team) & team_races.completed.eq(1)].dropna(subset=['episode_id'])
            columns = [c for c in ('win', 'both_finished', 'any_crash', 'finish_count') if c in facts]
            facts = facts[['episode_id', *columns]].rename(columns={c: team+'/'+c for c in columns})
            clips = clips.merge(facts, on='episode_id', how='left', validate='many_to_one')
    meta.update(training_races=sum(r.get('phase') == 'training' for r in race_rows),
                evaluation_snapshots=len(evaluation_rows), recorded_clips=len(clip_rows))
    if team_rows:
        meta.update(training_races=int(team_races.loc[(team_races.team == teams[0]) & team_races.phase.eq('training'), 'completed'].eq(1).sum()),
                    race_records_available=True, teams=teams)
    return RunData(path, meta, races, pd.DataFrame(agent_rows), updates,
                   pd.DataFrame(evaluation_rows), clips, team_races, pd.DataFrame(window_rows))


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
                 outcome=None, checkpoint=None, policy_version=None, team=None, phase=None, protocol=None,
                 window_index=None):
    rows = clips.copy()
    if rows.empty:
        return rows
    for column, value in [('run', run), ('map_id', map_id), ('kind', kind), ('phase', phase), ('protocol', protocol),
                          ('recording_window_index', window_index)]:
        if value is not None:
            rows = rows.loc[rows.get(column, pd.Series(index=rows.index, dtype=object)) == value]
    if event is not None:
        rows = rows.loc[rows.retention_reasons.map(lambda reasons: event in reasons)]
    if agent is not None:
        rows = rows.loc[rows.agent_ids.map(lambda ids: agent in ids)]
    if team is not None:
        mapping = rows.get('agent_teams', pd.Series(index=rows.index, dtype=object))
        rows = rows.loc[mapping.map(lambda m: isinstance(m, dict) and team in m.values())]
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
        if team is not None:
            aliases = {'any_learner_collision_dnf': 'any_crash', 'both_finished': 'both_finished', 'win': 'win'}
            if outcome not in aliases:
                raise ValueError('Team outcomes: both_finished, any_learner_collision_dnf, or win (completion/progress)')
            outcome = team+'/'+aliases[outcome]
        elif outcome not in RATE_METRICS:
            raise ValueError(f'Choose an outcome from {RATE_METRICS}')
        rows = rows.loc[rows.get(outcome, pd.Series(False, index=rows.index)).eq(True)]
    return rows


def replay_command(clip, repo_root, *, speed=1):
    return shlex.join([sys.executable, str(Path(repo_root)/'replay.py'), str(clip['dataset_dir']),
                       '--clip', clip['clip_id'], '--speed', str(speed)])


def summarize_selfplay_evaluations(team_races):
    """One row per team/map/evaluation; never pool different checkpoint pairs."""
    if team_races.empty:
        return pd.DataFrame()
    rows = team_races.loc[team_races.phase.eq('evaluation') & team_races.completed.eq(1)]
    if rows.empty:
        return pd.DataFrame()
    groups = ['run_key', 'run', 'training_seed', 'evaluation_id', 'protocol', 'protocol_id',
              'checkpoint', 'checkpoint_sha256', 'environment_steps', 'team', 'map_id']
    summaries = []
    for keys, part in rows.groupby(groups, dropna=False):
        summary = dict(zip(groups, keys))
        summary.update(race_count=len(part), seeds=part.seed.tolist())
        for metric in ('both_finished', 'win', 'draw', 'any_crash', 'finish_rate', 'progress_laps', 'reward'):
            if metric in part:
                measured = pd.to_numeric(part[metric], errors='coerce').dropna()
                summary.update({metric: measured.mean(), metric+'_count': len(measured),
                                metric+'_sum': measured.sum() if len(measured) else np.nan})
        summaries.append(summary)
    return pd.DataFrame(summaries)
