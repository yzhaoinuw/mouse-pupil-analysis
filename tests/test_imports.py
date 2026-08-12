# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:13:07 2026

@author: yzhao
"""

import sys
from inspect import signature

import pytest


def _forget_legacy_package():
    """Drop cached legacy modules so a fresh import re-runs the deprecation warning.

    A module-level ``warnings.warn`` fires once per process, so without this the
    assertion below would silently depend on no earlier test having imported the
    shim first.
    """
    for name in [
        name
        for name in sys.modules
        if name == "pupil_tracking" or name.startswith("pupil_tracking.")
    ]:
        del sys.modules[name]


def test_import_package():
    import mouse_pupil_analysis  # noqa: F401


def test_prediction_functions_are_available_from_their_own_module():
    from mouse_pupil_analysis.pupil_predictions import (
        DEFAULT_CHECKPOINT,
        generate_pupil_predictions,
    )

    assert DEFAULT_CHECKPOINT.is_file()
    assert generate_pupil_predictions.__module__ == "mouse_pupil_analysis.pupil_predictions"
    parameters = signature(generate_pupil_predictions).parameters
    assert "calculate_velocity" not in parameters
    assert "acquisition_fps" not in parameters


def test_legacy_package_and_module_paths_forward_to_the_renamed_package():
    import mouse_pupil_analysis
    from mouse_pupil_analysis.pupil_predictions import generate_pupil_mask_prediction

    _forget_legacy_package()

    with pytest.warns(DeprecationWarning, match="import mouse_pupil_analysis"):
        import pupil_tracking

    from pupil_tracking.run_pupil_analysis import (
        generate_pupil_mask_prediction as legacy_generate_pupil_mask_prediction,
    )

    assert pupil_tracking.analyze_video is mouse_pupil_analysis.analyze_video
    assert legacy_generate_pupil_mask_prediction is generate_pupil_mask_prediction


def test_legacy_module_paths_forward_names_the_renamed_module_omits_from_dunder_all():
    # ``run_pupil_analysis`` declares ``__all__``, so a star-import wrapper would
    # drop these re-exports even though the legacy module used to expose them.
    from mouse_pupil_analysis import run_pupil_analysis as renamed
    from pupil_tracking import run_pupil_analysis as legacy

    for name in ("AnalysisConfig", "configure_cli_logging", "run_analysis"):
        assert name not in renamed.__all__, f"{name} is no longer the case this guards"
        assert getattr(legacy, name) is getattr(renamed, name)
