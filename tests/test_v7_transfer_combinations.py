import numpy as np
import pandas as pd
import pytest

from cpt_budget.v7.transfer_combinations import frozen_profile_counts
from cpt_budget.v6.count_projection import project_frame


def cells():
    return pd.DataFrame([dict(fold_id=0, site_id="s", seed=17, budget_n=5,
        location_id=f"l{i}", depth_index=d) for i in range(3) for d in range(24)])


def test_fixed_count_projection_preserves_each_profile_and_ignores_labels():
    wide = cells()
    source = np.r_[np.ones(24), -np.ones(24), np.r_[-np.ones(7), np.ones(17)]]
    counts = frozen_profile_counts(wide, source)
    np.testing.assert_array_equal(counts.half_count_decision, [0, 24, 7])
    base = np.tile(np.linspace(-2., 2., 24), 3)
    first, diagnostic = project_frame(wide, base, counts)
    adversarial = wide.assign(truth_log_qc=np.arange(len(wide))*1e8, risk_probability=1.)
    second, _ = project_frame(adversarial, base, frozen_profile_counts(adversarial, source))
    np.testing.assert_array_equal(first, second)
    assert diagnostic.frozen_count.eq(diagnostic.actual_point_count).all()
    unchanged = (first < 0.) == (base < 0.)
    np.testing.assert_array_equal(first[unchanged], base[unchanged])
    assert np.all(abs(first[~unchanged]) == .05)


def test_count_source_rejects_missing_depths_and_nonfinite_predictions():
    wide = cells()
    with pytest.raises(ValueError, match="complete"):
        frozen_profile_counts(wide.iloc[1:], np.ones(len(wide)-1))
    values = np.ones(len(wide))
    values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        frozen_profile_counts(wide, values)
