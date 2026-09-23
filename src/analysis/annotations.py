"""Editable, versioned review annotations stored separately from P2 recordings."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


SCHEMA_VERSION = '1.0'
LABEL_VERSION = 'tactics_v1'
INDIVIDUAL_LABELS = ['free driving', 'following/pressure', 'pass attempt', 'position defense',
                     'yielding', 'avoidance', 'recovery', 'uncertain']
COMBINED_LABELS = ['sequential teammate passes', 'teammate yielding',
                   'contest while teammate passes', 'joint position defense', 'uncertain']
OUTCOMES = ['uncertain', 'success', 'failure', 'aborted', 'ongoing']


def source_reference(window):
    clip = window.clip
    return {**{k: clip.get(k) for k in ('run_id', 'environment_id', 'episode_id', 'clip_id', 'map_id',
        'policy_version_start', 'policy_version_end', 'team_policy_versions_start',
        'team_policy_versions_end', 'agent_teams', 'detector_version', 'kind', 'complete',
        'phase', 'evaluation_id', 'protocol', 'evaluation_protocol', 'checkpoint', 'checkpoint_sha256',
        'checkpoint_files', 'environment_seed', 'recording_window_index', 'recording_window',
        'recording_progress_start', 'recording_progress_clock', 'coverage_scope')},
        'dataset_dir': str(window.dataset), 'agent_ids': clip['agent_ids'],
        'available_start': window.boundaries[0], 'available_end': window.boundaries[-1]}


def segment(window, *, start, end, scope, participants, label, roles=None, targets=None,
            outcome='uncertain', confidence=.5, notes='', constituent_ids=None,
            participant_outcomes=None, team_outcome='unknown', evidence='',
            interpretation='observed_pattern', role_changes=None, annotation_id=None):
    """start/end are physics boundaries [start, end); no frame-index ambiguity."""
    if start not in window.boundaries or end not in window.boundaries or end <= start:
        raise ValueError('Choose start < end among the loaded physics boundaries')
    a, b = window.boundaries.index(start), window.boundaries.index(end)
    return dict(annotation_id=annotation_id or str(uuid.uuid4()), schema_version=SCHEMA_VERSION,
        label_version=LABEL_VERSION, source=source_reference(window), start=start, end=end,
        start_time_s=float(window.times[a]), end_time_s=float(window.times[b]), scope=scope,
        participants=list(participants), roles=roles or {}, targets=list(targets or []), label=label,
        outcome=outcome, confidence=confidence, notes=notes, constituent_ids=list(constituent_ids or []),
        participant_outcomes=participant_outcomes or {}, team_outcome=team_outcome,
        evidence=evidence, interpretation=interpretation, role_changes=role_changes or [])


def _race_key(row):
    s = row['source']
    return tuple(s.get(k) for k in ('dataset_dir', 'run_id', 'environment_id', 'episode_id'))


def validate_annotations(records):
    """Validate the whole graph, including parents affected by individual edits."""
    ids = [r['annotation_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate annotation IDs')
    by_id = dict(zip(ids, records))
    for row in records:
        if row.get('schema_version') != SCHEMA_VERSION or row.get('label_version') != LABEL_VERSION:
            raise ValueError('Unsupported annotation/label version')
        source = row['source']
        if not source['available_start'] <= row['start'] < row['end'] <= source['available_end']:
            raise ValueError('Segment exceeds its retained interval')
        if not row['start_time_s'] < row['end_time_s']:
            raise ValueError('Segment times must increase')
        participants = set(row['participants'])
        agents = set(source['agent_ids'])
        if not participants or len(participants) != len(row['participants']) or not participants <= agents:
            raise ValueError('Choose distinct participating cars from this race')
        if not set(row['targets']) <= agents or not set(row['roles']) <= participants:
            raise ValueError('Targets/roles must refer to valid cars/participants')
        if not set(row['participant_outcomes']) <= participants:
            raise ValueError('Individual outcomes must refer to participants')
        if any(v not in OUTCOMES for v in row['participant_outcomes'].values()) or row['outcome'] not in OUTCOMES:
            raise ValueError('Invalid outcome')
        if not isinstance(row['label'], str) or not row['label'].strip():
            raise ValueError('Provide a tactic label (custom labels are allowed)')
        if not isinstance(row['confidence'], (int, float)) or not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        if row['interpretation'] not in ('observed_pattern', 'inferred_coordination'):
            raise ValueError('Unknown interpretation')
        if row['team_outcome'] not in ('unknown', 'beneficial', 'neutral', 'harmful'):
            raise ValueError('Unknown team outcome')
        if (row['team_outcome'] != 'unknown' or row['interpretation'] == 'inferred_coordination') and not row['evidence'].strip():
            raise ValueError('Provide separate evidence for team outcome or inferred coordination')
        for change in row['role_changes']:
            if not row['start'] <= change['physics_index'] < row['end'] or not set(change['roles']) <= participants:
                raise ValueError('Role changes must lie within the segment and use its participants')
        links = row['constituent_ids']
        if row['scope'] == 'individual':
            if len(participants) != 1 or links:
                raise ValueError('Individual tactics have one acting car and no constituents; use targets for other cars')
            row['temporal_relations'] = []
        elif row['scope'] == 'combined':
            if len(participants) < 2 or len(set(links)) != len(links) or len(links) < 2:
                raise ValueError('Combined tactics require two participants and at least two distinct individual segments')
            children = []
            for link in links:
                child = by_id.get(link)
                if child is None or child['scope'] != 'individual' or _race_key(child) != _race_key(row):
                    raise ValueError('Constituents must be individual segments from the same recorded race')
                if child['start'] < row['start'] or child['end'] > row['end']:
                    raise ValueError('Combined segment must contain every constituent interval')
                if not set(child['participants']) <= participants:
                    raise ValueError('Combined participants must include all constituent actors')
                children.append(child)
            children.sort(key=lambda r: (r['start'], r['end'], r['annotation_id']))
            row['temporal_relations'] = [dict(first=a['annotation_id'], second=b['annotation_id'],
                relation='before' if a['end'] <= b['start'] else 'overlap')
                for i, a in enumerate(children) for b in children[i+1:]]
        else:
            raise ValueError('Scope must be individual or combined')


class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.reload()

    def reload(self):
        raw = self.path.read_bytes() if self.path.exists() else b''
        data = json.loads(raw) if raw else {'schema_version': SCHEMA_VERSION, 'annotations': []}
        if data.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported annotation file version')
        self.records = data['annotations']
        validate_annotations(self.records)
        self.revision = hashlib.sha256(raw).hexdigest()
        return deepcopy(self.records)

    def save(self, row):
        dataset = Path(row['source']['dataset_dir']).resolve()
        if self.path == dataset or dataset in self.path.parents:
            raise ValueError('Save annotations outside the raw recording directory')
        records = deepcopy(self.records)
        old = next((r for r in records if r['annotation_id'] == row['annotation_id']), None)
        row = deepcopy(row)
        if old and (_race_key(old) != _race_key(row) or old['source']['clip_id'] != row['source']['clip_id']):
            raise ValueError('An existing annotation cannot be reassigned to a different source clip')
        now = datetime.now(timezone.utc).isoformat()
        row['created_at'] = old['created_at'] if old else now
        row['updated_at'] = now
        records = [r for r in records if r['annotation_id'] != row['annotation_id']] + [row]
        validate_annotations(records)
        payload = json.dumps(dict(schema_version=SCHEMA_VERSION, annotations=records), indent=2, allow_nan=False)+'\n'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize read/check/replace so concurrent notebook kernels cannot lose edits.
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            raw = self.path.read_bytes() if self.path.exists() else b''
            if hashlib.sha256(raw).hexdigest() != self.revision:
                raise ValueError('Annotations changed in another reviewer. Reload before saving.')
            name = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as stream:
                    name = stream.name
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
            finally:
                if name and Path(name).exists():
                    Path(name).unlink()
        self.records = records
        self.revision = hashlib.sha256(payload.encode()).hexdigest()
        return deepcopy(next(r for r in records if r['annotation_id'] == row['annotation_id']))
