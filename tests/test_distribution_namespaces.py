"""Regression tests for wheel and source-distribution namespace checks."""

from scripts.verify_distribution_namespaces import forbidden_namespace_members


def test_forbidden_namespace_is_detected_in_wheel_layout():
    members = [
        "mouse_pupil_analysis/__init__.py",
        "pupil_tracking/__init__.py",
    ]

    assert forbidden_namespace_members(members) == ["pupil_tracking/__init__.py"]


def test_forbidden_namespace_is_detected_below_sdist_root():
    members = [
        "mouse_pupil_analysis-0.2.0/mouse_pupil_analysis/__init__.py",
        "mouse_pupil_analysis-0.2.0/pupil_tracking/__init__.py",
    ]

    assert forbidden_namespace_members(members) == [
        "mouse_pupil_analysis-0.2.0/pupil_tracking/__init__.py"
    ]


def test_project_name_substrings_do_not_trigger_false_positives():
    members = [
        "mouse_pupil_analysis/__init__.py",
        "mouse_pupil_analysis-0.2.0/mouse_pupil_analysis/checkpoints/model.pth",
        "mouse_pupil_analysis-0.2.0.dist-info/METADATA",
    ]

    assert forbidden_namespace_members(members) == []
