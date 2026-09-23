"""Exportable matplotlib figures for run review; missing values stay missing."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_review import summarize_races


LABELS = {
    'mean_net_progress_laps': 'Earned progress (laps per learner)',
    'any_learner_collision_dnf': 'Races with learner collision DNF',
    'training_return': 'Mean learner episode return', 'duration_s': 'Episode duration (s)',
    'both_finished': 'Both learners finished', 'first_place': 'Team first place',
    'sweep': 'Team sweep', 'clean_finish_time_s': 'Clean learner finish time (s)',
    'train/entropy': 'Policy entropy', 'train/approx_kl': 'Approximate KL',
    'train/policy_loss': 'Policy loss', 'train/value_loss': 'Value loss',
    'perf/round_env_steps_per_second': 'Joint environment steps / s',
    'train/clip_fraction': 'Clipping fraction',
}


def _finish_axis(ax, title, xlabel=None):
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    else:
        ax.text(.5, .5, 'No measured data', ha='center', va='center', transform=ax.transAxes)


def learning_curves(races, *, window=1):
    """One figure per race mode. Smooth over reporting barriers, weighted by races."""
    if window < 1:
        raise ValueError('window must be positive')
    figures = {}
    if races.empty:
        return figures
    training = races.loc[races.phase == 'training']
    metrics = ['mean_net_progress_laps', 'any_learner_collision_dnf', 'training_return', 'duration_s']
    for mode, part in training.groupby('race_mode', dropna=False):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
        for ax, metric in zip(axes.flat, metrics):
            for _, run in part.groupby('run_key', sort=False):
                if metric not in run:
                    continue
                points = run[['environment_steps', metric]].apply(pd.to_numeric, errors='coerce').dropna()
                if points.empty:
                    continue
                bins = points.groupby('environment_steps')[metric].agg(['sum', 'count'])
                smoothed = bins['sum'].rolling(window, min_periods=1).sum()/bins['count'].rolling(window, min_periods=1).sum()
                ax.plot(bins.index, smoothed, marker='.', label=str(run.run.iloc[0]))
            _finish_axis(ax, LABELS[metric], 'Joint environment decisions at reporting barrier')
        fig.suptitle(f'Training · {mode} · rolling {window} reporting points (completed episodes)')
        figures[f'learning_{mode}'] = fig
    return figures


def optimizer_curves(updates):
    if updates.empty or 'train/environment_steps' not in updates:
        return {}
    metrics = ['train/entropy', 'train/approx_kl', 'train/policy_loss', 'train/value_loss',
               'perf/round_env_steps_per_second', 'train/clip_fraction']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, metric in zip(axes.flat, metrics):
        for _, rows in updates.groupby('run_key', sort=False):
            if metric not in rows:
                continue
            points = rows[['train/environment_steps', metric]].apply(pd.to_numeric, errors='coerce').dropna()
            if len(points):
                points = points.sort_values('train/environment_steps')
                ax.plot(points.iloc[:, 0], points.iloc[:, 1], marker='.', label=str(rows.run.iloc[0]))
        _finish_axis(ax, LABELS[metric], 'Joint environment decisions')
    fig.suptitle('Optimizer diagnostics and throughput')
    return {'optimizer': fig}


def evaluation_curves(races):
    if races.empty:
        return {}
    evaluation = races.loc[(races.phase == 'evaluation') & races.environment_steps.notna()]
    figures = {}
    # Protocol fingerprints also separate changed evaluation conditions over time.
    for (protocol, protocol_id), rows in evaluation.groupby(['protocol', 'protocol_id']):
        summary = summarize_races(rows)
        steps = rows[['run_key', 'evaluation_id', 'environment_steps']].drop_duplicates()
        summary = summary.merge(steps, on=['run_key', 'evaluation_id'], validate='many_to_one')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
        for ax, metric in zip(axes, ['both_finished', 'first_place', 'sweep']):
            for (_, map_id), part in summary.loc[summary.metric == metric].groupby(['run_key', 'map_id']):
                part = part.sort_values('environment_steps').dropna(subset=['value'])
                if len(part):
                    ax.plot(part.environment_steps, part.value, marker='o',
                            label=f'{part.run.iloc[0]} / {map_id}')
            ax.set_ylim(-.03, 1.03)
            _finish_axis(ax, LABELS[metric], 'Joint environment decisions')
        fig.suptitle(f'Evaluation · {protocol} · conditions {protocol_id} (counts in tables)')
        figures[f'evaluation_curve_{protocol}_{protocol_id}'] = fig
    return figures


def outcome_plots(summary):
    """Separate panels by phase, mode, and evaluation conditions; annotate n/N."""
    if summary.empty:
        return {}
    figures = {}
    for (phase, mode, protocol, pid, map_id), rows in summary.groupby(
            ['phase', 'race_mode', 'protocol', 'protocol_id', 'map_id'], dropna=False):
        metrics = ['both_finished', 'first_place', 'sweep', 'any_learner_collision_dnf']
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), layout='constrained')
        for ax, metric in zip(axes, metrics):
            data = rows.loc[rows.metric == metric].dropna(subset=['value'])
            x = np.arange(len(data))
            bars = ax.bar(x, data.value, color='#346b9a')
            ax.set_xticks(x, data.run, rotation=35, ha='right', fontsize=7)
            ax.set_ylim(0, 1.17)
            for bar, (_, row) in zip(bars, data.iterrows()):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                        f'{row.numerator:g}/{row.denominator:g}', ha='center', fontsize=8)
            ax.set_title(LABELS[metric], fontsize=10)
            if data.empty:
                ax.text(.5, .5, 'Unavailable', transform=ax.transAxes, ha='center')
        fig.suptitle(f'{phase} · {mode} · {protocol} · {map_id}\nConditions: {pid}')
        figures[f'outcomes_{phase}_{mode}_{protocol}_{pid}_{map_id}'] = fig
    return figures


def finish_time_plots(agents):
    """Individual clean finisher distributions, with all learner outcomes as context."""
    if agents.empty:
        return {}
    figures = {}
    own = agents.loc[(agents.team == 'trainable') & (agents.race_mode == 'finite')]
    keys = ['phase', 'protocol', 'protocol_id', 'map_id']
    for values, rows in own.groupby(keys, dropna=False):
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        ticks, labels = [], []
        for index, (_, run) in enumerate(rows.groupby('run_key', sort=False)):
            times = pd.to_numeric(run.clean_finish_time_s, errors='coerce').dropna()
            finished = pd.to_numeric(run.finished, errors='coerce').dropna()
            ticks.append(index)
            labels.append(f'{run.run.iloc[0]}\nfinished {finished.sum():g}/{len(finished)}, clean times n={len(times)}')
            if len(times):
                ax.boxplot([times.to_numpy()], positions=[index], widths=.4, manage_ticks=False)
                ax.scatter(np.full(len(times), index), times, s=12, alpha=.6)
        ax.set_xticks(ticks, labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Clean learner finish time (s)')
        ax.set_title(' · '.join(str(v) for v in values))
        if rows.clean_finish_time_s.notna().sum() == 0:
            ax.text(.5, .5, 'No clean finish times recorded', ha='center', transform=ax.transAxes)
        figures['finish_times_'+'_'.join(str(v) for v in values)] = fig
    return figures


def seed_variation_plots(summary):
    if summary.empty:
        return {}
    figures = {}
    keys = ['phase', 'race_mode', 'protocol', 'protocol_id', 'map_id']
    for values, rows in summary.groupby(keys, dropna=False):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for ax, metric in zip(axes, ['both_finished', 'first_place', 'sweep']):
            data = rows.loc[rows.metric == metric].dropna(subset=['mean'])
            for index, (_, row) in enumerate(data.iterrows()):
                ax.plot(index, row['mean'], 'o', color='#346b9a')
                if pd.notna(row.seed_std):
                    ax.errorbar(index, row['mean'], yerr=row.seed_std, color='#346b9a', capsize=4)
            ax.set_xticks(range(len(data)), [f'{r.comparison_group}\nn={r.seed_count} seeds'
                                          for r in data.itertuples()], rotation=20, ha='right')
            ax.set_title(LABELS[metric])
            ax.set_ylabel('Equal-weight seed mean ± sample SD')
        fig.suptitle(' · '.join(str(v) for v in values))
        figures['seed_variation_'+'_'.join(str(v) for v in values)] = fig
    return figures


def export_figures(figures, directory):
    """Standalone PNG and PDF artifacts; called explicitly by the notebook."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
        for suffix in ('png', 'pdf'):
            fig.savefig(directory/f'{safe}.{suffix}', dpi=180, bbox_inches='tight')


def selfplay_curves(team_races, updates):
    """Symmetric team views; completion/progress wins are not first-place wins."""
    figures = {}
    training = team_races.loc[team_races.phase.eq('training')] if not team_races.empty else team_races
    if not training.empty:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
        complete = training.loc[training.completed.eq(1)]
        for ax, metric, label in zip(axes.flat,
                ['both_finished', 'progress_laps', 'any_crash', 'reward'],
                ['Both teammates finished', 'Team earned progress (laps)', 'Any team collision', 'Joint team episode return']):
            for (_, team), rows in complete.groupby(['run_key', 'team']):
                if metric not in rows:
                    continue
                points = rows.groupby('environment_steps')[metric].mean().dropna()
                ax.plot(points.index, points.values, marker='.', label=f'{rows.run.iloc[0]} / {team}')
            if metric in ('both_finished', 'any_crash'):
                ax.set_ylim(-.03, 1.03)
            _finish_axis(ax, label, 'Joint environment decisions at reporting barrier')
        fig.suptitle('Self-play training · completed races only · both policies change during training')
        figures['selfplay_training'] = fig
    if not updates.empty:
        metrics = [c for c in updates if c.startswith('train/') and c.endswith('/policy_loss')]
        if metrics:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
            for ax, suffix in zip(axes, ('policy_loss', 'value_loss')):
                for _, rows in updates.groupby('run_key'):
                    for key in [c for c in metrics if c.endswith('/policy_loss')]:
                        key = key.rsplit('/', 1)[0]+'/'+suffix
                        if key in rows:
                            data = rows[['train/environment_steps', key]].dropna()
                            if not data.empty:
                                ax.plot(data.iloc[:, 0], data.iloc[:, 1], marker='.', label=f'{rows.run.iloc[0]} / {key.split("/")[1]}')
                _finish_axis(ax, suffix.replace('_', ' '), 'Joint environment decisions')
            figures['selfplay_optimizer'] = fig
    return figures


def selfplay_evaluation_curves(summary):
    """Plot matched evaluation conditions separately, with counts in the table."""
    figures = {}
    if summary.empty:
        return figures
    for (protocol, pid), rows in summary.groupby(['protocol', 'protocol_id']):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
        for ax, metric, label in zip(axes, ('both_finished', 'win', 'any_crash'),
                ('Both teammates finished', 'Completion/progress win', 'Any team collision')):
            if metric in rows:
                for (_, team, map_id), part in rows.groupby(['run_key', 'team', 'map_id']):
                    part = part.dropna(subset=['environment_steps', metric]).sort_values('environment_steps')
                    if not part.empty:
                        ax.plot(part.environment_steps, part[metric], marker='o',
                            label=f'{part.run.iloc[0]} / {team} / {map_id}')
            ax.set_ylim(-.03, 1.03)
            _finish_axis(ax, label, 'Checkpoint training environment steps')
        fig.suptitle(f'Self-play evaluation · {protocol} · conditions {pid} (counts in table)')
        figures[f'selfplay_evaluation_{protocol}_{pid}'] = fig
    return figures
