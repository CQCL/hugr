#!/usr/bin/env python

import json
import subprocess
import sys
from pathlib import Path


def get_changed_files() -> list[Path]:
    """Get list of changed extension files in the PR"""
    # Use git to get the list of files changed compared to main
    cmd = [
        "git",
        "diff",
        "--name-only",
        "origin/main",
        "--",
        "specification/std_extensions/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    changed_files = [Path(f) for f in result.stdout.splitlines() if f.endswith(".json")]
    return changed_files


def check_version_changes(changed_files: list[Path]):
    """Check if versions have been updated in changed files"""
    errors = []

    for file_path in changed_files:
        # Skip files that don't exist anymore (deleted files)
        if not file_path.exists():
            continue

        # Get the version in the current branch
        with file_path.open("r") as f:
            current = json.load(f)
            current_version = current.get("version")

        # Get the version in the main branch
        try:
            cmd = ["git", "show", f"origin/main:{file_path}"]
            result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603

            if result.returncode == 0:
                # File exists in main
                main_content = json.loads(result.stdout)
                main_version = main_content.get("version")

                if current_version == main_version:
                    errors.append(
                        f"Error: {file_path} was modified but version was not updated"
                    )

            else:
                # New file - no version check needed
                pass

        except json.JSONDecodeError:
            # File is new or not valid JSON in main
            pass

    return errors


def main() -> int:
    changed_files = get_changed_files()
    if not changed_files:
        print("No extension files changed.")
        return 0

    print(f"Changed extension files: {', '.join(map(str,changed_files))}")

    errors = check_version_changes(changed_files)
    if errors:
        for error in errors:
            print(error)
        return 1

    print("All changed extension files have updated versions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
