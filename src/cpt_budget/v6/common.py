"""Truth-free frozen member inputs; evaluation truth is attached only on export."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from cpt_budget.config import PROJECT_ROOT, write_json
from cpt_budget.diagnostics.common import STATE_KEYS

RUN = PROJECT_ROOT / 'runs/v6-combinations-20261014'
REFERENCE = PROJECT_ROOT / 'runs/v5-exploration-20261014/comparison'
BEST = 'residual_ft_crps_guarded_a050'
KEYS = [*STATE_KEYS, 'location_id', 'depth_index']
GEOMETRY = ['x_m', 'y_m', 'z_center_m', 'geometry_scale_m', 'profile_id', 'near_group_id']

@lru_cache(maxsize=1)
def _frozen():
    return pd.read_parquet(REFERENCE / 'predictions_oof.parquet')

def load_frozen():
    """Return sorted cell keys, allowed geometry and 32 prediction columns only."""
    raw = _frozen()
    wide = raw.pivot(index=KEYS, columns='model_id', values='pred_log_qc').sort_index()
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy()).all():
        raise ValueError('Frozen member matrix must be complete')
    geometry = raw.loc[raw.model_id.eq(BEST)].set_index(KEYS)[GEOMETRY]
    return wide.join(geometry, validate='one_to_one').reset_index().rename_axis(columns=None)

def emit_candidate(model_id, values, *, wide=None, output_root=None,
                   probability=None, metadata=None):
    """Export full aligned evaluation frame, stripping all member distributions.

    values and optional native event probability must align exactly with wide.
    output_root is the family directory; writes output_root/model_id/predictions.parquet.
    Truth is never returned to the candidate generator.
    """
    wide = load_frozen() if wide is None else wide
    value = np.asarray(values, dtype=float).reshape(-1)
    if len(value) != len(wide) or not np.isfinite(value).all():
        raise ValueError('Candidate must supply one finite value per frozen cell')
    raw = _frozen()
    template = raw.loc[raw.model_id.eq(BEST)].copy()
    keep = KEYS + GEOMETRY + ['observed_set_id', 'investigation_group', 'policy_id',
                             'role', 'split_role', 'truth_log_qc', 'truth_qc_mpa']
    frame = wide[KEYS].merge(template[keep], on=KEYS, validate='one_to_one', how='left')
    if len(frame) != 8640 or frame.truth_log_qc.isna().any() or frame.duplicated(KEYS).any():
        raise ValueError('Only the full frozen matrix may be exported')
    frame['pred_log_qc'] = value
    frame['pred_qc_mpa'] = np.exp(value)
    if not np.isfinite(frame.pred_qc_mpa).all() or (frame.pred_qc_mpa <= 0).any():
        raise ValueError('Invalid physical prediction')
    if probability is not None:
        p = np.asarray(probability, dtype=float).reshape(-1)
        if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid native event probabilities')
        frame['p_low_1'] = p
    frame['model_id'] = model_id
    frame['run_id'], frame['study_version'] = 'v6-combinations-20261014', '0.6-combinations'
    target = Path(output_root or RUN / 'candidates') / model_id
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / 'predictions.parquet', index=False)
    write_json(target / 'protocol.json', {'model_id': model_id, 'scope': 'exploratory_validation_only',
               'states': 60, 'rows': len(frame), 'native_event_probability': probability is not None,
               'source': str(REFERENCE), **(metadata or {})})
    return target / 'predictions.parquet'
