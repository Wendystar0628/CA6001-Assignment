"""Retained inference component, consolidated from cpt_budget.models.tabpfn_adapter (v0.8-engineering)."""
from __future__ import annotations
from importlib.metadata import version
from pathlib import Path
from cpt_budget.types import QUERY_COLUMNS

def _runtime():
    installed = version('tabpfn')
    if installed != '9.2.0':
        raise RuntimeError(f'This adapter requires tabpfn==9.2.0; found {installed}')
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return (torch, TabPFNRegressor, ModelVersion)

def estimator_options(config=None):
    torch, _, _ = _runtime()
    config = config or {}
    settings = config.get('tabpfn', {})
    if str(settings.get('package_version', '9.2.0')) != '9.2.0' or settings.get('model_generation', 'v2') != 'v2':
        raise ValueError('The registered adapter fixes package 9.2.0 and v2 weights')
    if settings.get('target_context_only', True) is not True:
        raise ValueError('Historical context banks are outside the registered experiment')
    if settings.get('n_estimators', 1) != 1:
        raise ValueError('The registered experiment uses exactly one estimator')
    return dict(n_estimators=1, device=settings.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'), random_state=int(config.get('split', {}).get('seed', 20261013)), n_preprocessing_jobs=1, show_progress_bar=False, categorical_features_indices=[], inference_config={'MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE': 1000000})

def make_estimator(config=None, artifact=None, *, training=False):
    _, cls, model_version = _runtime()
    options = estimator_options(config)
    path = artifact or (config or {}).get('tabpfn', {}).get('base_checkpoint_path')
    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = Path((config or {}).get('_root', '.')) / path
        if artifact is not None and (not path.is_file()):
            raise FileNotFoundError(path)
        options['model_path'] = path
    return cls.create_default_for_version(model_version.V2, fit_mode='batched' if training else 'fit_preprocessors', **options)

def geometry_only(frame):
    """Avoid propagating measurements through query or prediction frames."""
    return frame[[c for c in QUERY_COLUMNS if c in frame]].copy()
