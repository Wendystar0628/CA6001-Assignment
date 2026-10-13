import pandas as pd
import pytest

from cpt_budget.diagnostics.training_chain import select_source_episode


def test_probe_selection_is_source_only_and_preserves_whole_query_ids():
    records = [
        dict(fold_id=0, role=role, site_id=site, episode_id="e1",
             support_location_ids=[f"s{i}" for i in range(5)],
             query_location_ids=["q3", "q1", "q2"])
        for role, site in [("test", "aaa"), ("validation", "aab"), ("train", "source")]
    ]
    episode, queries = select_source_episode(pd.DataFrame(records))
    assert episode["site_id"] == "source"
    assert queries == ["q1", "q2"]
    with pytest.raises(ValueError, match="No eligible source"):
        select_source_episode(pd.DataFrame(records[:-1]))
