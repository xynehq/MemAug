"""
Build Training Templates for All Commits.
Recursively processes all commits in the dataset directory.
"""

import json
import argparse
from pathlib import Path
from typing import List
from mem_aug.utils.build_training_template import build_template


def find_commit_directories(dataset_dir: Path) -> List[Path]:
    """
    Recursively find all commit directories in the dataset.
    A commit directory is identified by having all required files.
    """
    required_files = ["task.json", "reasoning.json", "fim_dataset.json", "diff_ast.jsonl"]
    commit_dirs = []

    # Walk through all directories
    for path in dataset_dir.rglob("*"):
        if path.is_dir():
            # Check if this directory has all required files
            has_all_files = all((path / f).exists() for f in required_files)
            if has_all_files:
                commit_dirs.append(path)

    return sorted(commit_dirs)


def process_commit(commit_dir: Path, force: bool = False) -> bool:
    """
    Process a single commit directory.
    Returns True if successful, False otherwise.
    """
    output_path = commit_dir / "template.txt"

    # Skip if template already exists and not forcing
    if output_path.exists() and not force:
        print(f"  ⏭️  Skipping {commit_dir.name} (template exists)")
        return True

    try:
        print(f"  🔨 Processing {commit_dir.name}...")
        template = build_template(commit_dir)

        # Write template
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)

        print(f"  ✅ Generated template ({len(template):,} chars)")
        return True

    except Exception as e:
        print(f"  ❌ Error processing {commit_dir.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build training templates for all commits in the dataset"
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path("data/ast_dataset"),
        help="Path to dataset directory (default: data/ast_dataset)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing template files"
    )
    parser.add_argument(
        "-r", "--repo",
        type=str,
        help="Process only specific repository (e.g., 'ripgrep')"
    )
    parser.add_argument(
        "-c", "--commit",
        type=str,
        help="Process only specific commit (e.g., 'commit_7')"
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        return 1

    print(f"🔍 Searching for commit directories in: {dataset_dir}")

    # Find all commit directories
    commit_dirs = find_commit_directories(dataset_dir)

    if not commit_dirs:
        print("❌ No commit directories found!")
        return 1

    # Filter by repository if specified
    if args.repo:
        commit_dirs = [d for d in commit_dirs if args.repo in str(d)]
        print(f"📌 Filtering by repository: {args.repo}")

    # Filter by commit if specified
    if args.commit:
        commit_dirs = [d for d in commit_dirs if args.commit in d.name]
        print(f"📌 Filtering by commit: {args.commit}")

    print(f"📊 Found {len(commit_dirs)} commit(s) to process\n")

    # Group commits by repository for better organization
    repos = {}
    for commit_dir in commit_dirs:
        # Extract repo name (parent of commit directory)
        repo_name = commit_dir.parent.name
        if repo_name not in repos:
            repos[repo_name] = []
        repos[repo_name].append(commit_dir)

    # Process each repository
    total_processed = 0
    total_success = 0
    total_failed = 0
    total_skipped = 0

    for repo_name in sorted(repos.keys()):
        repo_commits = repos[repo_name]
        print(f"\n📦 Repository: {repo_name} ({len(repo_commits)} commits)")
        print("─" * 60)

        for commit_dir in repo_commits:
            total_processed += 1

            # Check if already exists
            if (commit_dir / "template.txt").exists() and not args.force:
                total_skipped += 1
                print(f"  ⏭️  {commit_dir.name} (already exists)")
                continue

            success = process_commit(commit_dir, args.force)
            if success:
                total_success += 1
            else:
                total_failed += 1

    # Print summary
    print("\n" + "=" * 60)
    print("📈 Summary:")
    print(f"   Total commits: {total_processed}")
    print(f"   ✅ Generated: {total_success}")
    print(f"   ⏭️  Skipped: {total_skipped}")
    print(f"   ❌ Failed: {total_failed}")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())
