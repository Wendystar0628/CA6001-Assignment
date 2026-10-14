"""Third exploratory round: frozen event decoder on three new legal bases.

These bases were nominated after inspecting rounds one and two. This is a
continued exploratory search, not independent confirmation or preregistration.
The loader deliberately reads only keys and point predictions from each cache;
old query truth, risk scores, native probabilities and intervals never enter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpt_budget.v6.common import KEYS, RUN, emit_candidate, load_frozen
from cpt_budget.v6.indicator import IndicatorLayerPredictor, SETTINGS


BASES = {
    'residual_stack': ('adaptive', 'support_residual_stack_structural'),
    'stack': ('adaptive', 'support_stack_structural'),
    'fpca50': ('functional', 'v6_fpca_residual_w050'),
}
MODES = ('potts', 'cardinality')
MODELS = {f'v6_coupled_{short}_{mode}': (short, mode)
          for short in BASES for mode in MODES}


def load_base_points(path: Path, wide: pd.DataFrame) -> np.ndarray:
    """Column-projected cache read and exact cell alignment, without labels."""
    base = pd.read_parquet(path, columns=[*KEYS, 'pred_log_qc'])
    if base.duplicated(KEYS).any() or wide.duplicated(KEYS).any():
        raise ValueError('Base and target keys must be unique')
    if len(base) != len(wide):
        raise ValueError('Base must cover the complete target matrix')
    aligned = wide[KEYS].merge(base, on=KEYS, how='left', sort=False, validate='one_to_one')
    values = aligned.pred_log_qc.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Base keys differ from the target matrix or predictions are nonfinite')
    return values


def main():
    from cpt_budget.config import write_json
    from cpt_budget.diagnostics.common import STATE_KEYS, iter_validation_states

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'coupled_events')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'round': 3, 'models': MODELS, 'bases': BASES,
        'scope': 'exploratory_validation_only',
        'selection_provenance': 'Three bases nominated after rounds one and two were evaluated; this is not an independent confirmation.',
        'event_settings': SETTINGS,
        'additional_hyperparameter_search': 'none; only replace the amplitude base in two existing fixed decoders',
        'input_columns': [*KEYS, 'pred_log_qc'],
        'native_probability': 'recompute exact marginals under the Gibbs model using each new amplitude base',
        'point_thickness': 'computed only from actual decoded point qc < 1 MPa; not expected thickness',
    }
    write_json(args.output / 'frozen_protocol.json', protocol)
    wide = load_frozen()
    bases = {short: load_base_points(RUN / family / model / 'predictions.parquet', wide)
             for short, (family, model) in BASES.items()}
    values = {name: np.full(len(wide), np.nan) for name in MODELS}
    probabilities = {name: np.full(len(wide), np.nan) for name in MODELS}
    diagnostics = []
    for state in iter_validation_states():
        mask = np.ones(len(wide), dtype=bool)
        for key in STATE_KEYS:
            mask &= wide[key].eq(state.metadata[key]).to_numpy()
        positions = np.flatnonzero(mask)
        query = wide.loc[mask, ['site_id', 'location_id', 'depth_index', 'x_m', 'y_m', 'z_center_m']].reset_index(drop=True)
        for name, (short, mode) in MODELS.items():
            model = IndicatorLayerPredictor(mode=mode, model_id=name)
            model.condition(state.observed)
            bundle = model.predict(query, base_values=bases[short][positions])
            values[name][positions] = bundle.frame.pred_log_qc
            probabilities[name][positions] = bundle.frame.p_low_1
            for location, frame in bundle.frame.groupby('location_id'):
                diagnostics.append({**state.metadata, 'model_id': name, 'location_id': location,
                    'base_model': BASES[short][1], 'tau': model.tau, 'cardinality': model.cardinality,
                    'target_count': float(frame.indicator_target_count.iloc[0]),
                    'decoded_count': int((frame.pred_log_qc < 0).sum()),
                    'expected_count': float(frame.p_low_1.sum()),
                    'changed_cells': int(frame.indicator_changed_cells.iloc[0])})
        print(f'coupled event decoding completed: {state.key}', flush=True)
    for name, (short, mode) in MODELS.items():
        emit_candidate(name, values[name], wide=wide, probability=probabilities[name],
            output_root=args.output, metadata={**protocol, 'base_model': BASES[short][1], 'event_mode': mode})
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame['run_id'], diagnostic_frame['study_version'] = 'v6-combinations-20261014', '0.6-combinations'
    diagnostic_frame.to_csv(args.output / 'decoding_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
