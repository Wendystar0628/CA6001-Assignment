"""Run the pre-registered adaptations on the same sixty validation states."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd
from cpt_budget.config import write_json
from cpt_budget.types import PredictionBundle
from .common import (OUTPUT_ROOT, diagnostic_config, iter_validation_states,
                     evaluated_frame, metrics_for, run_predictor)


def residual(config):
    from .residual_adapter import BaselineResidualTabPFNPredictor, REGISTERED_ALPHAS
    root = OUTPUT_ROOT / 'residual_adapter/validation'
    names = {a: f'tabpfn_residual_a{int(a * 100):03d}' for a in REGISTERED_ALPHAS}
    all_parts = {a: [] for a in REGISTERED_ALPHAS}
    model = BaselineResidualTabPFNPredictor(config)
    for state in iter_validation_states(config):
        paths = {a: root / name / 'states' / f'{state.key}.parquet' for a, name in names.items()}
        if all(path.exists() for path in paths.values()):
            pieces = {a: pd.read_parquet(p) for a, p in paths.items()}
        else:
            started = time.perf_counter()
            model.condition(state.observed.copy())
            variants = model.predict_variants(state.query.copy(), alphas=REGISTERED_ALPHAS)
            pieces = {}
            for alpha, bundle in variants.items():
                bundle.model_id = names[alpha]
                frame = evaluated_frame(bundle, state)
                frame['shared_prediction_seconds'] = time.perf_counter() - started
                paths[alpha].parent.mkdir(parents=True, exist_ok=True)
                temporary = paths[alpha].with_suffix('.tmp.parquet')
                frame.to_parquet(temporary, index=False)
                temporary.replace(paths[alpha])
                pieces[alpha] = frame
            print(f'residual variants: {state.key}', flush=True)
        for alpha, frame in pieces.items():
            all_parts[alpha].append(frame)
    for alpha, pieces in all_parts.items():
        target = root / names[alpha]
        pd.concat(pieces, ignore_index=True).to_parquet(target / 'predictions.parquet', index=False)
        pd.DataFrame([metrics_for(p, config) for p in pieces]).to_csv(target / 'metrics.csv', index=False)
        write_json(target / 'model_definition.json', {**model.metadata_, 'alpha': alpha,
            'scope': 'exploratory_validation_only', 'paired_states': 60,
            'selection': 'All alpha candidates retained; no evaluation label enters predictor.'})


def context(config, mode):
    from .context_adapter import ContextAdapterPredictor, SourceContextBank
    def factory(fid):
        bank = (SourceContextBank.load(OUTPUT_ROOT / 'context_adaptation' / f'fold_{fid}')
                if mode == 'source_bank' else None)
        return ContextAdapterPredictor(config, source_bank=bank, mode=mode)
    return run_predictor(factory, OUTPUT_ROOT / 'context_adaptation' / mode / 'validation', config=config)


def point_guarded(config):
    """Reconstruct registered point-only variants from cached native medians."""
    from .residual_adapter import guarded_residual_frame
    native_root = OUTPUT_ROOT / 'residual_adapter/validation/tabpfn_residual_a100/states'
    for alpha in (.25, .5):
        name = f'tabpfn_guarded_a{int(alpha * 100):03d}'
        target = OUTPUT_ROOT / 'point_guarded' / name
        (target / 'states').mkdir(parents=True, exist_ok=True)
        frames, metrics = [], []
        for state in iter_validation_states(config):
            native = pd.read_parquet(native_root / f'{state.key}.parquet')
            frame = guarded_residual_frame(native, native.baseline_log_qc.to_numpy(),
                                           native.residual_pred_log_qc.to_numpy(), alpha)
            final = evaluated_frame(PredictionBundle(frame, name), state)
            final.to_parquet(target / 'states' / f'{state.key}.parquet', index=False)
            frames.append(final)
            metrics.append(metrics_for(final, config))
        pd.concat(frames, ignore_index=True).to_parquet(target / 'predictions.parquet', index=False)
        pd.DataFrame(metrics).to_csv(target / 'metrics.csv', index=False)
        write_json(target / 'model_definition.json', {'alpha': alpha,
            'scope': 'exploratory_addition_after_unguarded_residual_results',
            'point_rule': 'Preserve the IDW side of log(1 MPa); only point outputs.',
            'query_truth_used_for_prediction': False,
            'low_value_identification_improvement_claim': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=['residual', 'source_bank', 'honest', 'point_guarded'])
    args = parser.parse_args()
    config = diagnostic_config()
    if args.candidate == 'residual':
        residual(config)
    elif args.candidate == 'point_guarded':
        point_guarded(config)
    else:
        context(config, args.candidate)


if __name__ == '__main__':
    main()
