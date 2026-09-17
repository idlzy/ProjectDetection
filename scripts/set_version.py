#!/usr/bin/env python3
"""Synchronize the project version without creating a Git tag."""

from __future__ import annotations

import argparse
import os
import re
import stat
import tempfile
from pathlib import Path


SEMVER_PATTERN = re.compile(
    r"^(?:v)?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
VERSION_TARGETS = (
    (
        Path("pyproject.toml"),
        re.compile(r'(?m)^(version\s*=\s*")[^"]+("\s*)$'),
    ),
    (
        Path("src/project_detection/__init__.py"),
        re.compile(r'(?m)^(__version__\s*=\s*")[^"]+("\s*)$'),
    ),
)


def normalize_version(raw_version: str) -> str:
    match = SEMVER_PATTERN.fullmatch(raw_version.strip())
    if match is None:
        raise ValueError(
            "version must use MAJOR.MINOR.PATCH format, for example 0.1.1 "
            "or v0.1.1"
        )
    return ".".join(match.groups())


def prepare_updates(project_root: Path, version: str):
    updates = []
    current_versions = []
    for relative_path, pattern in VERSION_TARGETS:
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError("required version file does not exist: %s" % path)
        original = path.read_text(encoding="utf-8")
        matches = list(pattern.finditer(original))
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one version declaration in %s, found %d"
                % (path, len(matches))
            )
        declaration = matches[0]
        current_versions.append(
            (relative_path, original[declaration.start(1) + len(declaration.group(1)):declaration.start(2)])
        )
        updated = pattern.sub(
            lambda match: "%s%s%s" % (match.group(1), version, match.group(2)),
            original,
            count=1,
        )
        updates.append((path, original, updated))
    return updates, current_versions


def atomic_write(path: Path, content: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary_path), mode)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def apply_updates(updates) -> None:
    written = []
    try:
        for path, original, updated in updates:
            if updated == original:
                continue
            atomic_write(path, updated)
            written.append((path, original))
    except Exception:
        for path, original in reversed(written):
            atomic_write(path, original)
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize pyproject.toml and project_detection.__version__. "
            "This command does not create a Git tag."
        )
    )
    parser.add_argument("version", help="release version, such as 0.1.1 or v0.1.1")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the planned update without writing files",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        version = normalize_version(args.version)
        project_root = args.root.resolve()
        updates, current_versions = prepare_updates(project_root, version)
        if not args.dry_run:
            apply_updates(updates)
            verified_updates, verified_versions = prepare_updates(project_root, version)
            del verified_updates
            if any(current != version for _, current in verified_versions):
                raise RuntimeError("version verification failed after writing files")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print("Error: %s" % error)
        return 2

    action = "Would set" if args.dry_run else "Set"
    previous = ", ".join(
        "%s=%s" % (path, current) for path, current in current_versions
    )
    print("%s project version to %s" % (action, version))
    print("Previous versions: %s" % previous)
    print("Updated files:")
    for relative_path, _ in VERSION_TARGETS:
        print("  %s" % relative_path)
    print("Git tag was not created. Suggested tag: v%s" % version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
