"""
Utility script to clean up generated files from the dataset.
Useful for re-running generation scripts that now skip existing files.
"""
import os
import sys
import re
from typing import List, Set


def find_files_to_clean(base_dir: str, target_files: Set[str]) -> List[str]:
    """
    Find all files matching the target filenames in the dataset.

    Args:
        base_dir: Base directory containing AST dataset
        target_files: Set of filenames to search for (e.g., {'task.json', 'reasoning.json'})

    Returns:
        List of file paths to remove
    """
    files_to_remove = []

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return files_to_remove

    # Walk through all repositories and commits
    for repo_name in os.listdir(base_dir):
        repo_path = os.path.join(base_dir, repo_name)
        if not os.path.isdir(repo_path):
            continue

        # Find all commit directories
        commit_pattern = re.compile(r'^commit_(\d+)$')
        for d in os.listdir(repo_path):
            if os.path.isdir(os.path.join(repo_path, d)) and commit_pattern.match(d):
                commit_dir = os.path.join(repo_path, d)

                # Check for target files
                for filename in target_files:
                    file_path = os.path.join(commit_dir, filename)
                    if os.path.exists(file_path):
                        files_to_remove.append(file_path)

    return files_to_remove


def remove_files(file_paths: List[str], dry_run: bool = False) -> int:
    """
    Remove the specified files.

    Args:
        file_paths: List of file paths to remove
        dry_run: If True, only show what would be removed without actually removing

    Returns:
        Number of files removed
    """
    if not file_paths:
        return 0

    removed_count = 0

    for file_path in file_paths:
        try:
            if dry_run:
                print(f"  [DRY RUN] Would remove: {file_path}")
            else:
                os.remove(file_path)
                print(f"  ✓ Removed: {file_path}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {file_path}: {e}")

    return removed_count


def cleanup_dataset(
    base_dir: str = 'data/ast_dataset',
    remove_tasks: bool = False,
    remove_reasoning: bool = False,
    remove_fim: bool = False,
    dry_run: bool = False,
    repos: List[str] = None
):
    """
    Clean up generated files from the dataset.

    Args:
        base_dir: Base directory containing AST dataset
        remove_tasks: Remove task.json files
        remove_reasoning: Remove reasoning.json files
        remove_fim: Remove fim_dataset.json files
        dry_run: Show what would be removed without actually removing
        repos: List of specific repositories to clean (None = all repos)
    """
    print(f"\n{'='*60}")
    print(f"Dataset Cleanup Utility")
    print(f"{'='*60}")

    if not any([remove_tasks, remove_reasoning, remove_fim]):
        print("\n⚠️  No cleanup targets specified!")
        print("Use --tasks, --reasoning, or --fim flags")
        return

    # Determine which files to remove
    target_files = set()
    if remove_tasks:
        target_files.add('task.json')
    if remove_reasoning:
        target_files.add('reasoning.json')
    if remove_fim:
        target_files.add('fim_dataset.json')

    print(f"\nTarget files: {', '.join(sorted(target_files))}")
    print(f"Base directory: {base_dir}")
    if dry_run:
        print(f"Mode: DRY RUN (no files will be deleted)")
    else:
        print(f"Mode: LIVE (files will be deleted)")

    # If specific repos provided, only clean those
    if repos:
        print(f"Repositories: {', '.join(repos)}")
        total_removed = 0
        for repo in repos:
            repo_path = os.path.join(base_dir, repo)
            if not os.path.isdir(repo_path):
                print(f"\n⚠️  Repository '{repo}' not found, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"Cleaning repository: {repo}")
            print(f"{'='*60}")

            files_to_remove = find_files_to_clean(repo_path, target_files)

            if files_to_remove:
                print(f"\nFound {len(files_to_remove)} file(s) to remove:\n")
                removed = remove_files(files_to_remove, dry_run)
                total_removed += removed
                print(f"\n✓ Processed {removed} file(s) in {repo}")
            else:
                print(f"\nNo files to remove in {repo}")
    else:
        print(f"Repositories: ALL")

        print(f"\nScanning for files...")
        files_to_remove = find_files_to_clean(base_dir, target_files)

        if files_to_remove:
            print(f"\nFound {len(files_to_remove)} file(s) to remove:\n")
            total_removed = remove_files(files_to_remove, dry_run)
        else:
            print(f"\nNo files found matching criteria")
            total_removed = 0

    print(f"\n{'='*60}")
    if dry_run:
        print(f"DRY RUN: Would remove {total_removed} file(s)")
    else:
        print(f"Cleanup complete: Removed {total_removed} file(s)")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for dataset cleanup."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean up generated files from AST dataset',
        epilog='Examples:\n'
               '  cleanup-dataset --tasks --dry-run\n'
               '  cleanup-dataset --reasoning --fim repo1 repo2\n'
               '  cleanup-dataset --tasks --reasoning\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--tasks',
        action='store_true',
        help='Remove task.json files'
    )

    parser.add_argument(
        '--reasoning',
        action='store_true',
        help='Remove reasoning.json files'
    )

    parser.add_argument(
        '--fim',
        action='store_true',
        help='Remove fim_dataset.json files'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='data/ast_dataset',
        help='Base directory containing AST dataset (default: data/ast_dataset)'
    )

    parser.add_argument(
        'repos',
        nargs='*',
        help='Specific repositories to clean (default: all repos)'
    )

    args = parser.parse_args()

    # If no targets specified, show help
    if not any([args.tasks, args.reasoning, args.fim]):
        parser.print_help()
        sys.exit(1)

    cleanup_dataset(
        base_dir=args.base_dir,
        remove_tasks=args.tasks,
        remove_reasoning=args.reasoning,
        remove_fim=args.fim,
        dry_run=args.dry_run,
        repos=args.repos if args.repos else None
    )


if __name__ == "__main__":
    cli()
