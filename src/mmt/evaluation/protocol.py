'''E01 - protocol validator: catch leakage and identity problems BEFORE any model runs.'''
import pandas as pd
from .io_utils import canonical_hash

SPLITS = ('train', 'validation', 'calibration', 'test')

class ProtocolError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__('; '.join(self.errors))

def attach_groups(registry, membership):
    '''Registry rows inherit group_id and split from their parent sequence.'''
    m = membership[['sequence_id', 'group_id', 'split']]
    reg = registry.drop(columns=[c for c in ('split', 'group_id') if c in registry])
    out = reg.merge(m, on='sequence_id', how='left', validate='many_to_one')
    return out

def validate_protocol(membership, registry, class_map, require_train_classes=True):
    '''membership: sequence_id, group_id, split.  registry: decision_id, sequence_id, target_class,
    score_eligible, exclusion_reason.  class_map: {raw_id: index}.  Returns a report dict or raises
    ProtocolError listing EVERY problem found.'''
    errs = []
    need_m = {'sequence_id', 'group_id', 'split'}
    need_r = {'decision_id', 'sequence_id', 'target_class', 'score_eligible'}
    if not need_m <= set(membership.columns):
        raise ProtocolError([f'membership needs columns {sorted(need_m)}'])
    if not need_r <= set(registry.columns):
        raise ProtocolError([f'registry needs columns {sorted(need_r)}'])

    idx = sorted(class_map.values())
    C = len(idx)
    if idx != list(range(C)):
        errs.append('class map indices must be exactly 0..C-1')
    if len(set(class_map)) != len(class_map):
        errs.append('class map has duplicate raw ids')

    if membership.sequence_id.duplicated().any():
        errs.append('duplicate sequence ids in membership')
    bad = set(membership.split) - set(SPLITS)
    if bad:
        errs.append(f'unknown split values {sorted(bad)}')
    for s in SPLITS:
        if (membership.split == s).sum() == 0:
            errs.append(f'split {s!r} is empty')

    per_group = membership.groupby('group_id').split.nunique()
    shared = sorted(per_group[per_group > 1].index.astype(str))
    if shared:
        errs.append(f'groups present in more than one split: {shared}')

    if registry.decision_id.duplicated().any():
        errs.append('duplicate decision ids in registry')
    unknown = set(registry.sequence_id) - set(membership.sequence_id)
    if unknown:
        errs.append(f'decisions refer to unknown sequences: {sorted(unknown)[:5]}')
    no_dec = set(membership.sequence_id) - set(registry.sequence_id)
    if no_dec:
        errs.append(f'sequences with no decisions: {sorted(no_dec)[:5]}')

    elig = registry[registry.score_eligible.astype(bool)]
    if elig.target_class.isna().any():
        errs.append('eligible decisions without a target class')
    out_of_range = elig.target_class.dropna()
    if ((out_of_range < 0) | (out_of_range >= C)).any():
        errs.append('target_class outside the class map')
    inelig_with_target = registry[~registry.score_eligible.astype(bool)].target_class.notna().sum()
    # not an error: ineligible rows may keep a label for audit, but they are never scored.

    # membership is the single source of truth for split/group: ignore any copies already on the registry
    joined = registry.drop(columns=[c for c in ('split', 'group_id') if c in registry]).merge(
        membership[['sequence_id', 'group_id', 'split']], on='sequence_id', how='left')
    support = {}
    for s in SPLITS:
        e = joined[(joined.split == s) & joined.score_eligible.astype(bool)]
        support[s] = {int(c): int((e.target_class == c).sum()) for c in range(C)}
    if require_train_classes:
        missing = [c for c in range(C) if support['train'][c] == 0]
        if missing:
            errs.append(f'training split has no eligible decision for classes {missing}')
    if errs:
        raise ProtocolError(errs)

    counts = {s: dict(sequences=int((membership.split == s).sum()),
                      groups=int(membership[membership.split == s].group_id.nunique()),
                      decisions=int((joined.split == s).sum()),
                      eligible=int(((joined.split == s) & joined.score_eligible.astype(bool)).sum()))
              for s in SPLITS}
    excl = (registry[~registry.score_eligible.astype(bool)].exclusion_reason.fillna('unknown')
            .value_counts().to_dict() if 'exclusion_reason' in registry else {})
    return dict(n_classes=C, counts=counts, class_support=support, exclusions=excl,
                membership_hash=canonical_hash(membership.sort_values('sequence_id').to_dict('list')),
                class_map_hash=canonical_hash({str(k): v for k, v in class_map.items()}),
                ineligible_with_label=int(inelig_with_target))
