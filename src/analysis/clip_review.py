"""Bounded P2 clip loading and synchronized scene/telemetry drawing."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from src.replay.race_reader import iter_race_frames


@dataclass
class ClipWindow:
    dataset: Path
    clip: dict
    metadata: dict
    frames: list

    @property
    def times(self):
        return np.array([self.frames[0]['simulation_time_s']] +
                        [f['simulation_time_end_s'] for f in self.frames])

    @property
    def boundaries(self):
        return [self.frames[0]['physics_index']] + [f['physics_index_end'] for f in self.frames]

    def state(self, index):
        return self.frames[index-1]['post_state'] if index else self.frames[0]['pre_state']

    def delay(self, index, speed=1.):
        if speed <= 0:
            raise ValueError('Playback speed must be positive')
        return (self.times[index+1] - self.times[index])/speed


def load_window(dataset, clip, *, start=None, end=None, max_frames=2000):
    """Indices select physics intervals, inclusive; never silently downsample."""
    dataset = Path(dataset).resolve()
    if max_frames < 1:
        raise ValueError('max_frames must be positive')
    start = clip['start_physics_index'] if start is None else int(start)
    clip_end = clip.get('end_physics_index')
    if clip_end is not None and not np.isfinite(clip_end):
        clip_end = None
    end = clip_end if end is None else int(end)
    if start < clip['start_physics_index'] or (end is not None and end < start):
        raise ValueError('Choose a nonempty interval inside this clip')
    if clip_end is not None and (start > clip_end or end > clip_end):
        raise ValueError('Requested interval exceeds the retained clip')
    if end is not None and end-start+1 > max_frames:
        raise ValueError(f'Select at most {max_frames} physics intervals with REVIEW_START/REVIEW_END')
    frames = []
    for f in iter_race_frames(dataset, episode_id=clip['episode_id'], start=start, end=end):
        if len(frames) >= max_frames:
            raise ValueError(f'Open/long clip exceeds {max_frames} frames; choose an explicit interval')
        if f.get('environment_id') != clip.get('environment_id') or f.get('run_id') != clip.get('run_id'):
            raise ValueError('Frame source does not match clip identity')
        if f['physics_index_end'] != f['physics_index']+1 or f['simulation_time_end_s'] <= f['simulation_time_s']:
            raise ValueError('Invalid recorded physics interval')
        if frames and (f['physics_index'] != frames[-1]['physics_index_end'] or
                       not np.isclose(f['simulation_time_s'], frames[-1]['simulation_time_end_s'], atol=1e-9, rtol=0)):
            raise ValueError('Missing or discontinuous frames; choose a contiguous retained interval')
        if set(f['pre_state']) != set(clip['agent_ids']) or set(f['post_state']) != set(clip['agent_ids']):
            raise ValueError('Frame does not contain the clip agents')
        frames.append(f)
    if not frames:
        raise ValueError('No retained frames in this interval')
    if frames[0]['physics_index'] != start or (end is not None and frames[-1]['physics_index'] != end):
        raise ValueError('Requested interval is not fully retained; select available frames')
    return ClipWindow(dataset, dict(clip), json.loads((dataset/'metadata.json').read_text()), frames)


def telemetry(window, reference):
    """State at boundaries; commands apply over intervals starting at times[:-1]."""
    ids = window.clip['agent_ids']
    if reference not in ids:
        raise ValueError('Unknown reference car')
    times = window.times
    states = [window.state(i) for i in range(len(times))]
    contract = window.metadata.get('action_contract', {})
    units = contract.get('units', ['unknown', 'unknown'])
    length = window.clip.get('track_length_m')
    result = {}
    for aid in ids:
        car = [s[aid] for s in states]
        applied = [f.get('commands', {}).get(aid, {}).get('applied') for f in window.frames]
        controls = [s.get('controls') or {} for s in car]
        speed_control = contract.get('speed_control', '')
        rate_key = 'wheel_speed_reference_rate' if speed_control.startswith('wheel') else 'speed_reference_rate'
        reward_keys = sorted({k for f in window.frames for k in f.get('learners', {}).get(aid, {}).get('reward_components', {})})
        rewards = {k: [f.get('learners', {}).get(aid, {}).get('reward_components', {}).get(k, np.nan)
                       for f in window.frames] for k in reward_keys}
        result[aid] = dict(
            speed=[np.linalg.norm(s['velocity_body']) for s in car],
            steering=[c.get('steering_angle', np.nan) for c in controls],
            steering_command=[c[0] if c is not None else np.nan for c in applied],
            drive_command=[c[1] if c is not None else np.nan for c in applied],
            rate_command=[c.get(rate_key, np.nan) for c in controls[1:]],
            rewards=rewards, gap=[], lateral_gap=[])
        for index, s in enumerate(states):
            center = s[aid].get('centerline') or {}
            ref = s[reference].get('centerline') or {}
            if length and center.get('s') is not None and ref.get('s') is not None:
                if index == 0:
                    lapdiff = (s[aid].get('lap_count') or 0) - (s[reference].get('lap_count') or 0)
                    gap = center['s']-ref['s']
                    gap = gap+lapdiff*length if lapdiff else (gap+length/2)%length-length/2
                else:
                    a_delta, b_delta = center.get('progress_delta'), ref.get('progress_delta')
                    gap = gap+(a_delta-b_delta)*length if a_delta is not None and b_delta is not None else np.nan
            else:
                gap = np.nan
            result[aid]['gap'].append(gap if s[aid]['active'] and s[reference]['active'] else np.nan)
            d, r = center.get('d'), ref.get('d')
            result[aid]['lateral_gap'].append(d-r if d is not None and r is not None else np.nan)
    return result, units, contract.get('rate_units', 'unknown')


def draw_review(window, *, maps_dir=None, reference=None):
    """Return a figure plus a boundary-index updater, usable with Agg or ipympl."""
    ids = window.clip['agent_ids']
    reference = reference or ids[0]
    data, units, rate_units = telemetry(window, reference)
    times = window.times
    colors = dict(zip(ids, ['#0072B2', '#56B4E9', '#D55E00', '#E69F00']))
    fig = Figure(figsize=(15, 9), layout='constrained')
    grid = fig.add_gridspec(3, 3, width_ratios=[1.15, 1, 1])
    scene = fig.add_subplot(grid[:, 0])
    axes = [fig.add_subplot(grid[r, c]) for r in range(3) for c in (1, 2)]
    if maps_dir:
        import yaml
        from PIL import Image
        map_file = Path(maps_dir)/window.clip['map_id']/f"{window.clip['map_id']}.yaml"
        if map_file.exists():
            meta = yaml.safe_load(map_file.read_text())
            pixels = np.asarray(Image.open(map_file.parent/meta['image']))
            height, width = pixels.shape[:2]
            x, y, yaw = meta['origin']
            transform = Affine2D().rotate(yaw).translate(x, y)+scene.transData
            scene.imshow(np.flipud(pixels), origin='lower', cmap='gray',
                         extent=[0, width*meta['resolution'], 0, height*meta['resolution']],
                         transform=transform, alpha=.7)
    points = np.array([s['pose'][:2] for i in range(len(times)) for s in window.state(i).values()])
    low, high = points.min(axis=0)-3, points.max(axis=0)+3
    scene.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect='equal', xlabel='x (m)', ylabel='y (m)')
    patches, texts = {}, {}
    params = (window.metadata.get('physics_contract') or {}).get('vehicle_params', {})
    half_l, half_w = params.get('length', .58)/2, params.get('width', .31)/2
    corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
    for aid in ids:
        team = window.clip.get('agent_teams', {}).get(aid,
            'learner' if aid in window.clip['trainable_ids'] else 'opponent')
        color = colors[aid]
        patches[aid] = Polygon(corners, closed=True, facecolor=color, label=f'{aid} ({team})')
        scene.add_patch(patches[aid])
        texts[aid] = scene.text(0, 0, aid, color=color, fontsize=8)
        series = data[aid]
        axes[0].plot(times, series['speed'], color=color, label=aid)
        axes[1].plot(times, series['steering'], color=color, label=aid+' observed')
        axes[1].step(times, series['steering_command']+[series['steering_command'][-1]], where='post', color=color, ls='--')
        axes[2].step(times, series['drive_command']+[series['drive_command'][-1]], where='post', color=color)
        axes[3].step(times, series['rate_command']+[series['rate_command'][-1]], where='post', color=color)
        if aid != reference:
            axes[4].plot(times, series['gap'], color=color, label=aid+' longitudinal')
            axes[4].plot(times, series['lateral_gap'], color=color, ls=':', label=aid+' lateral')
        for key, values in series['rewards'].items():
            axes[5].plot(times[1:], values, color=color, label=f'{aid}: {key}',
                         ls=['-', '--', ':'][list(series['rewards']).index(key)%3])
    team_colors = {}
    for aid, team in window.clip.get('agent_teams', {}).items():
        team_colors.setdefault(team, colors[aid])
    team_keys = sorted({k for f in window.frames for k in f.get('team_reward_components', {})})
    for index, key in enumerate(team_keys):
        axes[5].plot(times[1:], [f.get('team_reward_components', {}).get(key, np.nan) for f in window.frames],
                     color=team_colors.get(key.split('/')[0], 'black'),
                     ls=['--', ':', '-.'][index % 3], label='team: '+key)
    titles = ['Observed speed (m/s)', 'Steering (rad): observed / dashed command',
              f'Applied drive reference ({units[1]})', f'Recorded reference rate ({rate_units})',
              f'Gaps to {reference} (m): longitudinal / dotted lateral', 'Reward components per physics interval']
    for frame in window.frames:
        if frame.get('events'):
            axes[5].axvline(frame['simulation_time_end_s'], color='gray', alpha=.3, lw=.7)
    cursors = []
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel='Recorded simulation time (s)', xlim=(times[0], times[-1]))
        ax.grid(alpha=.2)
        cursors.append(ax.axvline(times[0], color='black', lw=1))
    for ax in (axes[0], axes[4], axes[5]):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, loc='best')
    scene.legend(fontsize=8)

    def update(index):
        state = window.state(index)
        for aid in ids:
            x, y, yaw = state[aid]['pose']
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            patches[aid].set_xy(corners@rotation.T+[x, y])
            patches[aid].set_alpha(1 if state[aid]['active'] else .35)
            patches[aid].set_visible(state[aid].get('present', True))
            texts[aid].set_visible(state[aid].get('present', True))
            texts[aid].set_position((x+.4, y+.4))
        for cursor in cursors:
            cursor.set_xdata([times[index], times[index]])
        scene.set_title(f"{window.clip['map_id']}\nt={times[index]:.3f}s · boundary {window.boundaries[index]}")
        fig.canvas.draw_idle()
    update(0)
    return fig, update
