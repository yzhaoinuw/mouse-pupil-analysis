# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:13:07 2026

@author: yzhao
"""

from inspect import signature


def test_import_package():
    import pupil_tracking  # noqa: F401


def test_prediction_functions_are_available_from_their_own_module():
    from pupil_tracking.pupil_predictions import (
        DEFAULT_CHECKPOINT,
        generate_pupil_predictions,
    )

    assert DEFAULT_CHECKPOINT.is_file()
    assert generate_pupil_predictions.__module__ == "pupil_tracking.pupil_predictions"
    parameters = signature(generate_pupil_predictions).parameters
    assert "calculate_velocity" not in parameters
    assert "acquisition_fps" not in parameters
