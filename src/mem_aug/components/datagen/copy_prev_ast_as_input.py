"""
Copies the AST from the previous commit to the current commit as input_ast.jsonl.
This makes the training data clearer by explicitly showing:
- input_ast.jsonl: The code state BEFORE the commit (from previous commit)
- ast.jsonl: The code state AFTER the commit (current commit)
- diff_ast.jsonl: The differences between input and output

This helps prevent data leakage by clearly separating input from output.
"""
import os
import shutil
import re
from typing import List


def find_repo_dirs(base_dir: str) -> List[str]:
    """Finds repository directories within the base dataset directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


def find_commit_dirs(repo_dir: str) -> List[str]:
    """Finds and sorts commit directories within a repository directory."""
    commit_pattern = re.compile(r'^commit_(\d+)$')
    commit_dirs = []
    for d in os.listdir(repo_dir):
        if os.path.isdir(os.path.join(repo_dir, d)) and commit_pattern.match(d):
            commit_dirs.append(d)

    # Sort based on the numeric part of the directory name
    commit_dirs.sort(key=lambda x: int(commit_pattern.match(x).group(1)))
    return commit_dirs


def copy_prev_ast_to_current(repo_name: str, base_dir: str = 'data/ast_dataset'):
    """
    For each commit (starting from commit_2), copy the previous commit's ast.jsonl
    as input_ast.jsonl in the current commit directory.

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
        print(f"Skipping {repo_name}: Not enough commits (found {len(commit_dirs)}).")
        return

    copied_count = 0
    skipped_count = 0

    # Start from commit_2 (index 1) since commit_1 has no previous commit
    for i in range(1, len(commit_dirs)):
        prev_commit_name = commit_dirs[i - 1]
        curr_commit_name = commit_dirs[i]

        prev_commit_dir = os.path.join(repo_path, prev_commit_name)
        curr_commit_dir = os.path.join(repo_path, curr_commit_name)

        prev_ast_file = os.path.join(prev_commit_dir, 'ast.jsonl')
        curr_input_ast_file = os.path.join(curr_commit_dir, 'input_ast.jsonl')

        # Check if previous commit has ast.jsonl
        if not os.path.exists(prev_ast_file):
            print(f"  ✗ Skipping {curr_commit_name}: No ast.jsonl in {prev_commit_name}")
            skipped_count += 1
            continue

        # Copy prev_ast_file to curr_input_ast_file
        try:
            shutil.copy2(prev_ast_file, curr_input_ast_file)
            print(f"  ✓ Copied {prev_commit_name}/ast.jsonl → {curr_commit_name}/input_ast.jsonl")
            copied_count += 1
        except Exception as e:
            print(f"  ✗ Error copying for {curr_commit_name}: {e}")
            skipped_count += 1

    print(f"\nCompleted {repo_name}: {copied_count} files copied, {skipped_count} skipped")


def run_copy_pipeline(repos: List[str] = None, base_dir: str = 'data/ast_dataset'):
    """
    Runs the copy pipeline for specified repositories or all repositories.

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
    print(f"Copying previous AST as input for {len(repo_names)} repositories")
    print(f"{'='*60}")

    for repo_name in repo_names:
        copy_prev_ast_to_current(repo_name, base_dir)

    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for copying previous AST as input."""
    import sys

    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        run_copy_pipeline(repos)
    else:
        # Process all repos
        run_copy_pipeline()


if __name__ == "__main__":
    cli()
