"""Retained inference component, consolidated from cpt_budget.diagnostics.analyze (v0.8-engineering)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cpt_budget.experiment import BUDGETS
METRICS = ['log_mae', 'h_point_mae_m', 'recall', 'false_positive_rate', 'risk_recall', 'risk_fpr', 'risk_brier', 'coverage_90', 'width_90_log']

def curve_metrics(metrics):
    rows = []
    for key, group in metrics.groupby(['fold_id', 'site_id', 'seed', 'model_id'], sort=True):
        group = group.sort_values('budget_n')
        if group.budget_n.tolist() != list(BUDGETS):
            raise ValueError(f'Missing or duplicated budgets in {key}')
        row = dict(zip(['fold_id', 'site_id', 'seed', 'model_id'], key))
        x = group.budget_n.to_numpy(float)
        for name in METRICS:
            y = pd.to_numeric(group[name], errors='coerce').to_numpy(float)
            row[name] = float(np.trapezoid(y, x) / (x[-1] - x[0])) if np.isfinite(y).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
