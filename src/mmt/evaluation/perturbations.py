'''E03 - saved scenarios. build_schedule draws randomness ONCE and records the outcome;
apply_schedule is deterministic and never touches an RNG, so a reloaded schedule replays exactly.'''
import copy, hashlib, itertools, json
import numpy as np
from .io_utils import canonical_hash, canonical_json, to_jsonable

SCHEDULE_SEED = 1701
FIELDS = ('observation_ids', 'values', 'valid_mask', 'event_start', 'event_end', 'available_at')

def rng_for(protocol_id, scenario_id, seed, sequence_id, source):
    blob = canonical_json(dict(protocol=protocol_id, scenario=scenario_id, seed=seed,
                               sequence=sequence_id, source=source))
    first8 = int.from_bytes(hashlib.sha256(blob).digest()[:8], 'big')   # never Python hash()
    return np.random.Generator(np.random.PCG64(first8))

# ---------------------------------------------------------------- catalogue
def all_subsets(sources):
    '''2^M presence subsets including empty and full (16 for PAMAP2, 8 for MOSI).'''
    out = []
    for r in range(len(sources) + 1):
        out += [tuple(c) for c in itertools.combinations(sources, r)]
    return out

def scenario_catalogue(sources, R, profile='window'):
    '''One case per individual source plus one affecting all sources. Identical configs are deduplicated.
    R = reference duration: 5 s for PAMAP2, segment duration for MOSI.'''
    targets = [[s] for s in sources] + [list(sources)]
    raw = [dict(kind='clean', sources=[], param=None)]
    for t in targets:
        for p in (0.1, 0.3, 0.5): raw.append(dict(kind='removal', sources=t, param=p))
        for r in (0.1, 0.3, 0.5): raw.append(dict(kind='outage', sources=t, param=r))
        for d in (0.1, 0.25, 0.5): raw.append(dict(kind='delay', sources=t, param=d))
        for d in (0.1, 0.25, 0.5): raw.append(dict(kind='out_of_order', sources=t, param=d))
        raw.append(dict(kind='combined', sources=t, param=dict(outage=0.3, delay=0.25)))   # held-out regime
    for sub in all_subsets(sources):
        raw.append(dict(kind='subset', sources=list(sub), param=None))   # sources = those PRESENT
    seen, out = set(), []
    for sc in raw:
        if sc['kind'] == 'subset' and len(sc['sources']) == len(sources):
            sc = dict(kind='clean', sources=[], param=None)          # full subset == clean
        sc = dict(sc, R=R, profile=profile)
        h = canonical_hash(sc)
        if h in seen: continue
        seen.add(h)
        tag = ','.join(sc['sources']) if sc['sources'] else 'none'
        sc['id'] = f"{sc['kind']}|{tag}|{canonical_json(sc['param']).decode()}"
        out.append(sc)
    return out

# ---------------------------------------------------------------- schedule
def _observation_digest(sources):
    h = hashlib.sha256()
    for key in sorted(sources):
        s = sources[key]
        h.update(repr(key).encode()); h.update(np.asarray(s.observation_ids, np.int64).tobytes())
        h.update(np.asarray(s.available_at, np.float64).tobytes())
    return h.hexdigest()

def _seq_bounds(sources):
    b = {}
    for (seq, _), s in sources.items():
        if len(s.event_end):
            lo, hi = float(s.event_start.min()), float(s.event_end.max())
            b[seq] = (min(lo, b[seq][0]), max(hi, b[seq][1])) if seq in b else (lo, hi)
    return b

def build_schedule(sources, scenario, seed=SCHEDULE_SEED, protocol_id='protocol'):
    '''sources: {(sequence_id, source_id): SourceSequence-like}. Returns a JSON-serialisable schedule.'''
    bounds = _seq_bounds(sources); kind = scenario['kind']; R = scenario['R']
    targeted = set(scenario['sources']); entries = []
    for (seq, sid), s in sorted(sources.items()):
        ids = np.asarray(s.observation_ids)
        order = np.argsort(ids, kind='stable')                       # draw in observation-ID order
        sid_ids = ids[order]
        e = dict(sequence_id=seq, source_id=sid, removed_ids=[], delay_ids=[], delay_values=[], note={})
        if kind == 'subset':
            if sid not in targeted:                                  # sources listed are the PRESENT ones
                e['removed_ids'] = sid_ids.tolist(); e['note']['dropped_whole_source'] = True
        elif kind != 'clean' and sid in targeted:
            rng = rng_for(protocol_id, scenario['id'], seed, seq, sid)
            lo, hi = bounds[seq]; dur = hi - lo
            if kind == 'removal':
                gone = rng.random(len(sid_ids)) < scenario['param']
                e['removed_ids'] = sid_ids[gone].tolist()
            elif kind in ('outage', 'combined'):
                r = scenario['param'] if kind == 'outage' else scenario['param']['outage']
                L = min(r * R, dur)
                a = lo if L >= dur else float(rng.uniform(lo, hi - L))
                ee = np.asarray(s.event_end)[order]
                gone = (ee >= a) & (ee < a + L)                        # half-open [a, a+L)
                e['removed_ids'] = sid_ids[gone].tolist(); e['note']['outage'] = [a, a + L]
                if kind == 'combined':
                    d = scenario['param']['delay'] * R
                    late = (~gone) & (ee >= a + L)                     # recovery delay on surviving rows
                    e['delay_ids'] = sid_ids[late].tolist(); e['delay_values'] = [d] * int(late.sum())
            elif kind == 'delay':
                d = scenario['param'] * R
                e['delay_ids'] = sid_ids.tolist(); e['delay_values'] = [d] * len(sid_ids)
            elif kind == 'out_of_order':
                dmax = scenario['param'] * R
                e['delay_ids'] = sid_ids.tolist(); e['delay_values'] = rng.uniform(0, dmax, len(sid_ids)).tolist()
            else:
                raise ValueError(f'unknown scenario kind {kind}')
        entries.append(e)
    sched = dict(schema_version=1, scenario=scenario, seed=seed, protocol_id=protocol_id,
                 numpy_version=np.__version__, input_hash=_observation_digest(sources), entries=entries)
    sched['schedule_hash'] = canonical_hash(dict(entries=entries, scenario=scenario, seed=seed))
    sched['stats'] = schedule_stats(sources, apply_schedule(sources, sched), sched)
    return sched

def _subset_rows(s, keep):
    n = copy.copy(s)
    for f in FIELDS: setattr(n, f, np.asarray(getattr(s, f))[keep])
    return n

def apply_schedule(sources, schedule):
    '''Deterministic. Delays change available_at ONLY; event times and labels never change.'''
    out = dict(sources)
    for e in schedule['entries']:
        key = (e['sequence_id'], e['source_id']); s = sources[key]; ids = np.asarray(s.observation_ids)
        keep = ~np.isin(ids, np.asarray(e['removed_ids'], dtype=ids.dtype)) if e['removed_ids'] else np.ones(len(ids), bool)
        n = _subset_rows(s, keep)
        if e['delay_ids']:
            add = dict(zip(e['delay_ids'], e['delay_values']))
            n.available_at = np.asarray(n.available_at, np.float64) + np.array([add.get(int(i), 0.0) for i in n.observation_ids])
        out[key] = n
    return out

def schedule_stats(before, after, schedule):
    tot = sum(len(s.observation_ids) for s in before.values())
    kept = sum(len(s.observation_ids) for s in after.values())
    inv = 0
    for s in after.values():                                   # arrival inversions vs event order
        if len(s.observation_ids) > 1:
            o = np.lexsort((s.observation_ids, s.event_start, s.event_end))
            inv += int((np.diff(np.asarray(s.available_at)[o]) < 0).sum())
    kind = schedule['scenario']['kind']
    return dict(rows_before=tot, rows_after=kept, rows_removed=tot - kept, arrival_inversions=inv,
                exercises_reordering=(inv > 0) if kind == 'out_of_order' else None,
                closure_availability_test=(schedule['scenario'].get('profile') == 'segment'
                                           and kind in ('delay', 'out_of_order', 'combined')))

def affected_fraction(decisions, clean, transformed, select_ids):
    '''Report observed exposure: share of decisions whose selected IDs changed. select_ids is Member 4's selector.'''
    changed = 0
    for d in decisions:
        for key in clean:
            if key[0] != d['sequence_id']: continue
            if list(select_ids(clean[key], d)) != list(select_ids(transformed[key], d)):
                changed += 1; break
    return changed / max(len(decisions), 1)

def save_schedule(path, schedule):
    from pathlib import Path
    Path(path).write_text(json.dumps(to_jsonable(schedule), sort_keys=True, allow_nan=False))

def load_schedule(path):
    from pathlib import Path
    return json.loads(Path(path).read_text())
