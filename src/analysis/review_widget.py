"""Notebook controls for synchronized P2 playback and persistent P3 labeling."""
import asyncio
from copy import deepcopy
from html import escape
import json

import ipywidgets as W
from ipympl.backend_nbagg import new_figure_manager_given_figure

from .annotations import (
    AnnotationStore, segment, INDIVIDUAL_LABELS, COMBINED_LABELS, OUTCOMES,
)
from .clip_review import draw_review


class ClipReviewer:
    def __init__(self, window, *, annotation_path, maps_dir=None, reference=None):
        self.window = window
        self.store = AnnotationStore(annotation_path)
        self._task = None
        self._editing_id = None
        self._role_changes = []
        self.figure, self._draw = draw_review(window, maps_dir=maps_dir, reference=reference)
        self.manager = new_figure_manager_given_figure(id(self), self.figure)
        self.canvas = self.manager.canvas
        self.canvas.toolbar_visible = True
        self.canvas.header_visible = False
        self.canvas.layout.width = '100%'
        self.cursor = W.IntSlider(value=0, min=0, max=len(window.frames), description='Boundary',
                                  continuous_update=False, layout=W.Layout(width='70%'))
        self.play = W.ToggleButton(description='Play', icon='play')
        self.speed = W.Dropdown(options=[.25, .5, 1., 2., 4.], value=1., description='Speed ×')
        self.loop = W.Checkbox(value=False, description='Loop window')
        self.clock = W.HTML()
        self.facts = W.HTML()
        self.message = W.HTML()
        self.cursor.observe(self._seek, names='value')
        self.play.observe(self._play_changed, names='value')
        event_options = [('Choose event', None)]
        for i, frame in enumerate(window.frames):
            for event in frame.get('events', []):
                text = f"{frame['simulation_time_end_s']:.3f}s · {event['kind']} · {','.join(event.get('participants', []))} ({event.get('source', 'unknown')})"
                event_options.append((text, i+1))
        self.events = W.Dropdown(options=event_options, description='Events', layout=W.Layout(width='95%'))
        self.events.observe(lambda change: self.seek(change['new']) if change['new'] is not None else None, names='value')
        self._build_editor()
        clip = window.clip
        title = W.HTML(f"<b>{escape(clip['clip_id'])}</b> · {escape(clip['kind'])} · "
            f"complete={clip.get('complete')} · {escape(str(clip.get('end_reason')))}<br>"
            'State is shown at the cursor boundary; commands apply over physics intervals. '
            'Dashed steering is commanded; faded cars are terminal. '
            'Gaps integrate signed progress, initialized from lap counts/nearest same-lap spacing. '
            'Missing reference rates remain blank. Playback may run slower than requested if rendering falls behind.')
        self.widget = W.VBox([title, W.HBox([self.play, self.speed, self.loop]), self.cursor,
            self.clock, self.canvas, self.events, self.facts, self.editor, self.message])
        self._seek({'new': 0})

    def seek(self, index):
        self.play.value = False
        self.cursor.value = index

    def _seek(self, change):
        i = change['new']
        self._draw(i)
        self.clock.value = f'<b>{self.window.times[i]:.3f} s</b> · physics boundary {self.window.boundaries[i]}'
        state = self.window.state(i)
        cars = '; '.join(f"{aid}: {s.get('terminal_reason') or 'active'} (lap {s.get('lap_count')})" for aid, s in state.items())
        frame = self.window.frames[max(0, i-1)]
        events = frame.get('events', []) if i else []
        self.facts.value = '<pre>'+escape(cars+'\n'+json.dumps(events, indent=2))+'</pre>'

    def _play_changed(self, change):
        if not change['new']:
            if self._task is not None and not self._task.done():
                self._task.cancel()
            self.play.description = 'Play'
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.play.value = False
            self.message.value = 'Playback requires a running notebook kernel event loop.'
            return
        self.play.description = 'Pause'
        self._task = loop.create_task(self._animate())

    async def _animate(self):
        try:
            while self.play.value:
                i = self.cursor.value
                if i == len(self.window.frames):
                    if self.loop.value:
                        self.cursor.value = 0
                        i = 0
                    else:
                        break
                await asyncio.sleep(self.window.delay(i, self.speed.value))
                if self.play.value and self.cursor.value == i:
                    self.cursor.value = i+1
        except asyncio.CancelledError:
            pass
        finally:
            # A paused task must not stop a newly started playback task.
            if self._task is asyncio.current_task():
                self._task = None
                self.play.value = False

    def _build_editor(self):
        w = self.window
        ids = w.clip['agent_ids']
        self.saved = W.Dropdown(description='Saved', layout=W.Layout(width='95%'))
        self.saved.observe(self._load_selected, names='value')
        self.scope = W.Dropdown(options=['individual', 'combined'], description='Scope')
        self.label = W.Combobox(options=INDIVIDUAL_LABELS, value='uncertain', ensure_option=False,
                               description='Tactic', layout=W.Layout(width='65%'))
        self.scope.observe(lambda c: setattr(self.label, 'options', INDIVIDUAL_LABELS if c['new']=='individual' else COMBINED_LABELS), names='value')
        self.start = W.BoundedIntText(value=w.boundaries[0], min=w.boundaries[0], max=w.boundaries[-1], description='Start')
        self.end = W.BoundedIntText(value=w.boundaries[-1], min=w.boundaries[0], max=w.boundaries[-1], description='End')
        start_button = W.Button(description='Set start at cursor')
        end_button = W.Button(description='Set end at cursor')
        start_button.on_click(lambda _: setattr(self.start, 'value', w.boundaries[self.cursor.value]))
        end_button.on_click(lambda _: setattr(self.end, 'value', w.boundaries[self.cursor.value]))
        self.participants = W.SelectMultiple(options=ids, value=(ids[0],), description='Actors')
        self.targets = W.SelectMultiple(options=ids, description='Targets')
        self.roles = {aid: W.Text(description=aid+' role', placeholder='e.g. passer, yielding teammate') for aid in ids}
        self.car_outcomes = {aid: W.Dropdown(description=aid+' result', options=OUTCOMES) for aid in ids}
        self.outcome = W.Dropdown(options=OUTCOMES, description='Outcome')
        self.confidence = W.FloatSlider(value=.5, min=0, max=1, step=.05, description='Confidence')
        self.notes = W.Textarea(description='Notes', layout=W.Layout(width='95%'))
        self.constituents = W.SelectMultiple(description='Segments', layout=W.Layout(width='95%', height='100px'))
        self.team_outcome = W.Dropdown(options=['unknown', 'beneficial', 'neutral', 'harmful'], description='Team result')
        self.interpretation = W.Dropdown(options=[('Observed pattern', 'observed_pattern'),
            ('Inferred coordination', 'inferred_coordination')], description='Interpretation', layout=W.Layout(width='60%'))
        self.evidence = W.Textarea(description='Evidence', placeholder='Evidence for team benefit or inferred intent, separate from the maneuver label.', layout=W.Layout(width='95%'))
        self.role_change_list = W.Select(description='Role changes', layout=W.Layout(width='95%'))
        add_role = W.Button(description='Record roles at cursor', layout=W.Layout(width='200px'))
        remove_role = W.Button(description='Remove role change', layout=W.Layout(width='190px'))
        add_role.on_click(self._add_role_change)
        remove_role.on_click(self._remove_role_change)
        save = W.Button(description='Save segment', button_style='success')
        save.on_click(self._save_clicked)
        reload_button = W.Button(description='Reload saved file')
        reload_button.on_click(self._reload_clicked)
        self.editor = W.VBox([W.HTML('<h3>Segment annotation</h3>Intervals are [start, end) physics boundaries. '
            'Individual scope has one acting car; other cars can be targets. Combined scope links at least two individual segments. '
            'Overlapping and custom labels are allowed. Use Ctrl/Cmd to select multiple items.'),
            W.HBox([self.saved, reload_button]), W.HBox([self.scope, self.label]),
            W.HBox([self.start, start_button, self.end, end_button]),
            W.HBox([self.participants, self.targets]),
            *[W.HBox([self.roles[aid], self.car_outcomes[aid]]) for aid in ids],
            W.HBox([self.outcome, self.confidence]), self.notes, self.constituents,
            W.HBox([self.team_outcome, self.interpretation]), self.evidence,
            W.HBox([add_role, remove_role]), self.role_change_list, save,
            W.HTML('Saved separately to <code>'+escape(str(self.store.path))+'</code>')])
        self._refresh_saved()

    def _same_race(self, row):
        s, c = row['source'], self.window.clip
        return s['dataset_dir'] == str(self.window.dataset) and all(s.get(k) == c.get(k) for k in ('run_id', 'environment_id', 'episode_id'))

    def _refresh_saved(self, selected=None):
        rows = [r for r in self.store.records if self._same_race(r)]
        describe = lambda r: f"{r['scope']} · {r['label']} · {r['start']}–{r['end']} · {','.join(r['participants'])} · {r['annotation_id'][:8]}"
        self.saved.options = [('New segment', None)]+[(describe(r), r['annotation_id']) for r in rows]
        self.constituents.options = [(describe(r), r['annotation_id']) for r in rows if r['scope']=='individual']
        self.saved.value = selected

    def _load_selected(self, change):
        annotation_id = change['new']
        self._editing_id = None
        self._role_changes = []
        if annotation_id is None:
            self.scope.value = 'individual'
            self.label.value = 'uncertain'
            self.start.value, self.end.value = self.window.boundaries[0], self.window.boundaries[-1]
            self.participants.value = (self.window.clip['agent_ids'][0],)
            self.targets.value = ()
            self.outcome.value, self.confidence.value = 'uncertain', .5
            self.notes.value = self.evidence.value = ''
            self.interpretation.value, self.team_outcome.value = 'observed_pattern', 'unknown'
            self.constituents.value = ()
            for aid in self.roles:
                self.roles[aid].value, self.car_outcomes[aid].value = '', 'uncertain'
        else:
            row = next(r for r in self.store.records if r['annotation_id']==annotation_id)
            if (row['source']['clip_id'] != self.window.clip['clip_id'] or
                    row['start'] < self.window.boundaries[0] or row['end'] > self.window.boundaries[-1]):
                self.message.value = escape(f"Open source clip {row['source']['clip_id']} with interval {row['start']}–{row['end']-1} to edit this segment.")
                # Refuse saving a duplicate under the wrong source from stale form values.
                self._editing_id = 'unavailable'
                return
            self._editing_id = annotation_id
            for key in ('scope', 'label', 'start', 'end', 'outcome', 'confidence', 'notes', 'evidence', 'interpretation', 'team_outcome'):
                getattr(self, key).value = row[key]
            self.participants.value, self.targets.value = tuple(row['participants']), tuple(row['targets'])
            self.constituents.value = tuple(row['constituent_ids'])
            for aid in self.roles:
                self.roles[aid].value = row['roles'].get(aid, '')
                self.car_outcomes[aid].value = row['participant_outcomes'].get(aid, 'uncertain')
            self._role_changes = deepcopy(row['role_changes'])
        self._show_role_changes()

    def _show_role_changes(self):
        self.role_change_list.options = [(f"{r['physics_index']}: {r['roles']}", i) for i, r in enumerate(self._role_changes)]

    def _add_role_change(self, _):
        self._role_changes.append(dict(physics_index=self.window.boundaries[self.cursor.value],
            roles={a: self.roles[a].value for a in self.participants.value}))
        self._show_role_changes()

    def _remove_role_change(self, _):
        if self.role_change_list.value is not None:
            self._role_changes.pop(self.role_change_list.value)
            self._show_role_changes()

    def save_annotation(self):
        if self._editing_id == 'unavailable':
            raise ValueError('Open the source clip or choose New segment before saving')
        row = segment(self.window, start=self.start.value, end=self.end.value, scope=self.scope.value,
            participants=self.participants.value, targets=self.targets.value, label=self.label.value,
            roles={a: self.roles[a].value for a in self.participants.value},
            participant_outcomes={a: self.car_outcomes[a].value for a in self.participants.value},
            outcome=self.outcome.value, confidence=self.confidence.value, notes=self.notes.value,
            constituent_ids=self.constituents.value, team_outcome=self.team_outcome.value,
            evidence=self.evidence.value, interpretation=self.interpretation.value,
            role_changes=self._role_changes, annotation_id=self._editing_id)
        saved = self.store.save(row)
        self._refresh_saved(saved['annotation_id'])
        return saved

    def _save_clicked(self, _):
        try:
            row = self.save_annotation()
            self.message.value = 'Saved '+escape(row['annotation_id'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.message.value = '<b>Not saved:</b> '+escape(str(exc))

    def _reload_clicked(self, _):
        try:
            self.store.reload()
            self._refresh_saved()
            self._load_selected({'new': None})
            self.message.value = 'Reloaded saved annotations; editor reset to a new segment.'
        except (ValueError, OSError) as exc:
            self.message.value = escape(str(exc))

    def close(self):
        self.play.value = False
        self.canvas.close()
        self.widget.close()
