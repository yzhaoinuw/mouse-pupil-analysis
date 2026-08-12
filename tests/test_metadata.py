"""Guard the metadata that a citation depends on.

A DOI is only useful if the version recorded in the package, the build metadata,
and the citation file all agree. These checks are cheap and catch drift before a
release rather than after one is archived.
"""

import re
from pathlib import Path

import pytest

import mouse_pupil_analysis

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 predates tomllib.
    tomllib = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
CITATION = PROJECT_ROOT / "CITATION.cff"

pytestmark = pytest.mark.skipif(
    not PYPROJECT.is_file() or tomllib is None,
    reason="Metadata checks need a source checkout and tomllib (Python 3.11+).",
)


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def test_installed_version_matches_pyproject():
    assert mouse_pupil_analysis.__version__ == _project()["version"]


def test_citation_version_matches_pyproject():
    citation = CITATION.read_text(encoding="utf-8")
    match = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', citation, re.MULTILINE)

    assert match is not None, "CITATION.cff has no version field."
    assert match.group(1).strip() == _project()["version"]


def test_distribution_name_is_the_published_one():
    # The import name and the distribution name intentionally differ; __init__
    # looks the version up by distribution name, so a rename must update both.
    assert _project()["name"] == "mouse-pupil-analysis"


def test_packaging_does_not_claim_the_unrelated_pupil_tracking_namespace():
    # The unrelated PyPI distribution ``pupil-tracking`` installs its own
    # ``pupil_tracking/__init__.py``. If this project also shipped that path, the
    # two distributions would own one import namespace: installing either would
    # overwrite the other's file, and uninstalling either would delete it while
    # pip still reported the survivor as installed.
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    assert not any(pattern.startswith("pupil_tracking") for pattern in include)
    assert not (PROJECT_ROOT / "pupil_tracking").exists()


def test_console_scripts_target_the_renamed_package():
    # The commands themselves are part of the public interface and must not
    # change, but they have to resolve inside the renamed package.
    scripts = _project()["scripts"]

    assert set(scripts) == {"extract-frames", "run-pupil-analysis"}
    for command, target in scripts.items():
        assert target.startswith("mouse_pupil_analysis."), command


def test_public_api_matches_the_lazy_export_map():
    expected = set(mouse_pupil_analysis._EXPORTS) | {"__version__"}

    assert set(mouse_pupil_analysis.__all__) == expected
    for name in mouse_pupil_analysis.__all__:
        assert getattr(mouse_pupil_analysis, name) is not None


def test_changelog_records_the_current_version():
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{_project()['version']}]" in changelog
