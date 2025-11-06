"""
Generates unified diff strings for all repositories and commits.
Processes diff_ast.jsonl files and saves unified diff strings to unified_diffs.jsonl.
"""
import os
import json
import re
from typing import List, Dict, Any
from mem_aug.utils.generate_unified_diffs import process_diff_ast_file


def find_repo_dirs(base_dir: str) -> List[str]:
    """Finds repository directories within the base commit data directory."""
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


def process_commit_unified_diffs(
    commit_dir: str,
    commit_name: str,
    simple: bool = True,
    separate_imports: bool = True
) -> Dict[str, Any]:
    """
    Process a single commit directory to generate unified diff strings.

    Args:
        commit_dir: Path to commit directory
        commit_name: Name of the commit (e.g., 'commit_3')
        simple: If True, generate simple diffs without file headers
        separate_imports: If True, generate separate diff strings for imports and code

    Returns:
        Dictionary with statistics about the processing
    """
    diff_ast_file = os.path.join(commit_dir, 'diff_ast.jsonl')
    output_file = os.path.join(commit_dir, 'unified_diffs.jsonl')

    # Skip if diff_ast.jsonl doesn't exist
    if not os.path.exists(diff_ast_file):
        return {
            'status': 'skipped',
            'reason': 'diff_ast.jsonl not found',
            'diffs_generated': 0
        }

    # Skip if output file already exists
    if os.path.exists(output_file):
        return {
            'status': 'skipped',
            'reason': 'unified_diffs.jsonl already exists',
            'diffs_generated': 0
        }

    try:
        # Process the diff_ast.jsonl file
        results = process_diff_ast_file(diff_ast_file, simple=simple, separate_imports=separate_imports)

        # Save results to unified_diffs.jsonl
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

        return {
            'status': 'success',
            'diffs_generated': len(results),
            'output_file': output_file
        }

    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
            'diffs_generated': 0
        }


def process_repository_unified_diffs(
    repo_name: str,
    base_dir: str = 'data/ast_dataset',
    simple: bool = True,
    separate_imports: bool = True
):
    """
    Processes a single repository to generate unified diff strings for all commits.

    Args:
        repo_name: Name of the repository to process
        base_dir: Base directory containing AST dataset
        simple: If True, generate simple diffs without file headers
        separate_imports: If True, generate separate diff strings for imports and code
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"{'='*60}")

    repo_path = os.path.join(base_dir, repo_name)
    commit_dirs = find_commit_dirs(repo_path)

    if not commit_dirs:
        print(f"Skipping {repo_name}: No commits found.")
        return

    diffs_generated = 0
    commits_processed = 0
    commits_skipped = 0
    errors = 0

    # Process each commit directory
    for commit_name in commit_dirs:
        commit_dir = os.path.join(repo_path, commit_name)

        result = process_commit_unified_diffs(
            commit_dir,
            commit_name,
            simple=simple,
            separate_imports=separate_imports
        )

        if result['status'] == 'success':
            print(f"  ✓ Generated unified diffs: {commit_name} ({result['diffs_generated']} entries)")
            diffs_generated += result['diffs_generated']
            commits_processed += 1
        elif result['status'] == 'skipped':
            print(f"  ⊘ Skipping {commit_name}: {result['reason']}")
            commits_skipped += 1
        elif result['status'] == 'error':
            print(f"  ✗ Error processing {commit_name}: {result['reason']}")
            errors += 1

    print(f"\nCompleted {repo_name}:")
    print(f"  - {commits_processed} commits processed")
    print(f"  - {diffs_generated} unified diffs generated")
    print(f"  - {commits_skipped} commits skipped")
    if errors > 0:
        print(f"  - {errors} errors")


def run_unified_diff_pipeline(
    repos: List[str] = None,
    base_dir: str = 'data/ast_dataset',
    simple: bool = True,
    separate_imports: bool = True
):
    """
    Runs the unified diff generation pipeline for specified repositories or all repositories.

    Args:
        repos: List of repository names to process. If None, processes all repos.
        base_dir: Base directory containing AST dataset
        simple: If True, generate simple diffs without file headers
        separate_imports: If True, generate separate diff strings for imports and code
    """
    if repos is None:
        repo_names = find_repo_dirs(base_dir)
        if not repo_names:
            print(f"No repository data found in {base_dir}.")
            print("Please run 'generate-ast' and 'generate-ast-diffs' first.")
            return
    else:
        repo_names = repos

    print(f"\n{'='*60}")
    print(f"Starting unified diff generation for {len(repo_names)} repositories")
    print(f"Format: {'Simple' if simple else 'Full'}")
    print(f"Separate imports: {'Yes' if separate_imports else 'No'}")
    print(f"{'='*60}")

    for repo_name in repo_names:
        process_repository_unified_diffs(
            repo_name,
            base_dir,
            simple=simple,
            separate_imports=separate_imports
        )

    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for unified diff generation."""
    import sys

    # Parse arguments
    repos = []
    simple = True
    separate_imports = True

    for arg in sys.argv[1:]:
        if arg == '--full':
            simple = False
        elif arg == '--no-separate-imports':
            separate_imports = False
        elif arg.startswith('--'):
            print(f"Unknown option: {arg}")
            print("\nUsage: python generate_unified_diff_strings.py [repo1 repo2 ...] [--full] [--no-separate-imports]")
            print("\nOptions:")
            print("  --full                  Generate full diffs with file headers (default: simple)")
            print("  --no-separate-imports   Don't generate separate diff strings for imports (default: separate)")
            print("\nExamples:")
            print("  python generate_unified_diff_strings.py                       # Process all repos, simple format, separate imports")
            print("  python generate_unified_diff_strings.py hyperswitch           # Process specific repo")
            print("  python generate_unified_diff_strings.py --full                # Process all repos, full format")
            print("  python generate_unified_diff_strings.py --no-separate-imports # Process all repos, combined diffs")
            return
        else:
            repos.append(arg)

    # Run the pipeline
    if repos:
        run_unified_diff_pipeline(repos, simple=simple, separate_imports=separate_imports)
    else:
        run_unified_diff_pipeline(simple=simple, separate_imports=separate_imports)


if __name__ == "__main__":
    cli()
