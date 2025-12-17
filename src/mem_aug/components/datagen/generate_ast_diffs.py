"""
Generates AST diffs between consecutive commits for all repositories.
This is Step 2 of the pipeline after generate_commit_ast.py.
"""

import os
import json
import re
from typing import List
from mem_aug.utils.build_commit_diff import main as build_diff


def find_repo_dirs(base_dir: str) -> List[str]:
    """Finds repository directories within the base commit data directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


def find_commit_dirs(repo_dir: str) -> List[str]:
    """Finds and sorts commit directories within a repository directory."""
    commit_pattern = re.compile(r"^commit_(\d+)$")
    commit_dirs = []
    for d in os.listdir(repo_dir):
        if os.path.isdir(os.path.join(repo_dir, d)) and commit_pattern.match(d):
            commit_dirs.append(d)

    # Sort based on the numeric part of the directory name
    commit_dirs.sort(key=lambda x: int(commit_pattern.match(x).group(1)))
    return commit_dirs


def process_repository_diffs(repo_name: str, base_dir: str = "data/ast_dataset"):
    """
    Processes a single repository to generate AST diffs between consecutive commits.

    Args:
        repo_name: Name of the repository to process
        base_dir: Base directory containing AST dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"{'='*60}")

    repo_path = os.path.join(base_dir, repo_name)
    commit_dirs = find_commit_dirs(repo_path)

    if len(commit_dirs) < 2:
        print(
            f"Skipping {repo_name}: Not enough commits to compare (found {len(commit_dirs)})."
        )
        return

    diffs_generated = 0
    diffs_skipped = 0

    # Compare consecutive commits
    for i in range(len(commit_dirs) - 1):
        commit1_name = commit_dirs[i]
        commit2_name = commit_dirs[i + 1]

        commit1_dir = os.path.join(repo_path, commit1_name)
        commit2_dir = os.path.join(repo_path, commit2_name)

        file1 = os.path.join(commit1_dir, "ast.jsonl")
        file2 = os.path.join(commit2_dir, "ast.jsonl")
        output_file = os.path.join(commit2_dir, "diff_ast.jsonl")
        commit_data_file = os.path.join(commit2_dir, "commit_data.json")

        # Skip if output file already exists
        if os.path.exists(output_file):
            print(
                f"  ⊘ Skipping {commit1_name} → {commit2_name}: diff_ast.jsonl already exists"
            )
            diffs_skipped += 1
            continue

        if not os.path.exists(file1):
            print(
                f"  ✗ Skipping {commit1_name} → {commit2_name}: AST file not found for {commit1_name}"
            )
            diffs_skipped += 1
            continue
        if not os.path.exists(file2):
            print(
                f"  ✗ Skipping {commit1_name} → {commit2_name}: AST file not found for {commit2_name}"
            )
            diffs_skipped += 1
            continue

        # Check if commit data exists and contains Rust changes
        use_commit_data = False
        if os.path.exists(commit_data_file):
            try:
                with open(commit_data_file, "r") as f:
                    commit_data = json.load(f)
                    # Check if the diff contains changes to .rs files
                    if ".rs" in commit_data.get("diff", ""):
                        use_commit_data = True
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                pass  # Silently skip if can't read commit data

        try:
            # Call build_commit_diff directly
            build_diff(
                file1,
                file2,
                output_file,
                commit_data_file=commit_data_file if use_commit_data else None,
            )
            print(f"  ✓ Generated diff: {commit1_name} → {commit2_name}")
            diffs_generated += 1
        except Exception as e:
            print(f"  ✗ Error generating diff {commit1_name} → {commit2_name}: {e}")
            diffs_skipped += 1

    print(
        f"\nCompleted {repo_name}: {diffs_generated} diffs generated, {diffs_skipped} skipped"
    )


def run_diff_pipeline(repos: List[str] = None, base_dir: str = "data/ast_dataset"):
    """
    Runs the diff generation pipeline for specified repositories or all repositories.

    Args:
        repos: List of repository names to process. If None, processes all repos.
        base_dir: Base directory containing AST dataset
    """
    if repos is None:
        repo_names = find_repo_dirs(base_dir)
        if not repo_names:
            print(f"No repository data found in {base_dir}.")
            print("Please run 'generate-ast' first.")
            return
    else:
        repo_names = repos

    print(f"\n{'='*60}")
    print(f"Starting AST diff generation for {len(repo_names)} repositories")
    print(f"{'='*60}")

    for repo_name in repo_names:
        process_repository_diffs(repo_name, base_dir)

    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for AST diff generation."""
    import sys

    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        run_diff_pipeline(repos)
    else:
        # Process all repos
        run_diff_pipeline()


if __name__ == "__main__":
    cli()
