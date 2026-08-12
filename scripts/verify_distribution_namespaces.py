"""Reject built distributions that claim forbidden top-level package names."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

FORBIDDEN_PACKAGE_NAMES = frozenset({"pupil_tracking"})


def forbidden_namespace_members(member_names: Iterable[str]) -> list[str]:
    """Return archive members containing a forbidden package path component.

    Source distributions prefix every member with a project-version directory,
    while wheels do not. Inspecting path components works for both layouts.
    """
    return [
        name
        for name in member_names
        if FORBIDDEN_PACKAGE_NAMES.intersection(PurePosixPath(name).parts)
    ]


def distribution_members(path: Path) -> list[str]:
    """Read member names from a wheel or gzipped source distribution."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported distribution artifact: {path}")


def verify_distribution(path: Path) -> None:
    """Raise when a distribution contains a forbidden package namespace."""
    leaked = forbidden_namespace_members(distribution_members(path))
    if leaked:
        raise RuntimeError(f"{path.name} ships a forbidden namespace: {leaked}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify built artifacts do not claim unrelated package namespaces."
    )
    parser.add_argument("distribution_dir", nargs="?", default="dist", type=Path)
    args = parser.parse_args()

    distributions = sorted(path for path in args.distribution_dir.iterdir() if path.is_file())
    if not distributions:
        raise SystemExit(f"No distributions found in {args.distribution_dir}.")

    for distribution in distributions:
        try:
            verify_distribution(distribution)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{distribution.name}: no forbidden package namespace")


if __name__ == "__main__":
    main()
