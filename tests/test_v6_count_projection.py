from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from cpt_budget.v6.count_projection import load_frozen_counts, project_count, project_frame
from cpt_budget.v6.thickness_head import minimally_relabel


def test_ascending_projection_has_minimum_l1_change_under_fixed_flip_rule():
    base = np.array([-.8, -.2, .1, .5, -.01])
    for k in range(len(base)+1):
        predicted = project_count(base, k)
        alternatives = []
        for selected in combinations(range(len(base)), k):
            desired = np.zeros(len(base), bool)
            desired[list(selected)] = True
            alternatives.append(np.abs(minimally_relabel(base, desired)-base).sum())
        assert (predicted < 0).sum() == k
        assert np.abs(predicted-base).sum() == pytest.approx(min(alternatives))


def test_count_loader_reads_only_prediction_allowlist_and_checks_formula(tmp_path):
    row = dict(fold_id=0, site_id="s", seed=17, budget_n=5, location_id="l",
        half_count_decision=4, predicted_continuous_count=3., residual_point_count=5,
        truth_qc_mpa="This must not be parsed or used")
    path = tmp_path / "counts.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    result = load_frozen_counts(path)
    assert "truth_qc_mpa" not in result
    row["half_count_decision"] = 3
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="formula"):
        load_frozen_counts(path)


def test_projection_keeps_same_count_and_row_shuffle_alignment():
    wide = pd.DataFrame([dict(fold_id=0, site_id="s", seed=17, budget_n=5,
        location_id="l", depth_index=d, truth_log_qc=1000.) for d in range(24)])
    base = np.linspace(-1., 1., 24)
    counts = pd.DataFrame([dict(fold_id=0, site_id="s", seed=17, budget_n=5,
        location_id="l", half_count_decision=5)])
    original, _ = project_frame(wide, base, counts)
    changed = wide.sample(frac=1, random_state=2)
    shuffled, diagnostic = project_frame(changed, base[changed.index], counts)
    np.testing.assert_allclose(shuffled, original[changed.index])
    assert diagnostic.actual_point_count.iloc[0] == 5
