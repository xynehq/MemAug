"""
Generate Retrieval Labels for All Commits.
Recursively processes all commits in the dataset directory and generates
retrieval labels based on AST graph hops for training retrieval loss.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from mem_aug.utils.extract_ast_hops import get_retrieval_labels


def find_commit_directories(dataset_dir: Path) -> List[Path]:
    """
    Recursively find all commit directories in the dataset.
    A commit directory is identified by having ast.jsonl file.
    """
    commit_dirs = []

    # Walk through all directories
    for path in dataset_dir.rglob("*"):
        if path.is_dir():
            # Check if this directory has ast.jsonl
            if (path / "ast.jsonl").exists():
                commit_dirs.append(path)

    return sorted(commit_dirs)


def process_commit(
    commit_dir: Path,
    positive_hops: List[int] = [0, 1],
    negative_sample_size: int = 5,
    force: bool = False
) -> bool:
    """
    Process a single commit directory and generate retrieval labels.
    Only processes functions that appear in diff_ast.jsonl (changed functions).
    Uses full ast.jsonl to build the dependency graph.
    Returns True if successful, False otherwise.
    """
    ast_file = commit_dir / "ast.jsonl"
    diff_ast_file = commit_dir / "diff_ast.jsonl"
    output_path = commit_dir / "retrieval_labels.json"

    # Skip if labels already exist and not forcing
    if output_path.exists() and not force:
        print(f"  ⏭️  Skipping {commit_dir.name} (labels exist)")
        return True

    # Skip if no diff_ast.jsonl (no changes in this commit)
    if not diff_ast_file.exists():
        print(f"  ⏭️  Skipping {commit_dir.name} (no diff_ast.jsonl)")
        return True

    try:
        print(f"  🔨 Processing {commit_dir.name}...")

        # Load full AST nodes (for building dependency graph)
        ast_nodes = []
        with open(ast_file, 'r') as f:
            for line in f:
                ast_nodes.append(json.loads(line))

        # Load diff AST to get changed function IDs
        diff_nodes = []
        with open(diff_ast_file, 'r') as f:
            for line in f:
                diff_nodes.append(json.loads(line))

        # Get function IDs from diff_ast (only changed functions)
        function_ids = [
            node['id'] for node in diff_nodes
            if node['kind'] == 'function_item'
        ]

        if not function_ids:
            print(f"  ⚠️  No changed functions found in {commit_dir.name}")
            return True

        # Generate labels for each changed function
        results = {
            'commit': commit_dir.name,
            'commit_path': str(commit_dir),
            'total_functions_in_ast': len([n for n in ast_nodes if n['kind'] == 'function_item']),
            'changed_functions': len(function_ids),
            'positive_hops': positive_hops,
            'negative_sample_size': negative_sample_size,
            'function_labels': {}
        }

        success_count = 0
        for func_id in function_ids:
            try:
                labels = get_retrieval_labels(
                    func_id,
                    ast_nodes,
                    positive_hops=positive_hops,
                    negative_sample_size=negative_sample_size
                )
                results['function_labels'][func_id] = labels
                success_count += 1
            except Exception as e:
                print(f"    ⚠️  Error processing {func_id}: {e}")
                continue

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✅ Generated labels for {success_count}/{len(function_ids)} changed functions")
        return True

    except Exception as e:
        print(f"  ❌ Error processing {commit_dir.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrieval labels for all commits in the dataset"
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
        help="Overwrite existing label files"
    )
    parser.add_argument(
        "-r", "--repo",
        type=str,
        help="Process only specific repository (e.g., 'bat')"
    )
    parser.add_argument(
        "-c", "--commit",
        type=str,
        help="Process only specific commit (e.g., 'commit_10')"
    )
    parser.add_argument(
        "--positive-hops",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Hop levels to use as positive samples (default: 0 1)"
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=5,
        help="Number of negative samples per function (default: 5)"
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        return 1

    print(f"🔍 Searching for commit directories in: {dataset_dir}")
    print(f"📊 Configuration:")
    print(f"   Positive hops: {args.positive_hops}")
    print(f"   Negative samples: {args.negative_samples}")

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
            if (commit_dir / "retrieval_labels.json").exists() and not args.force:
                total_skipped += 1
                print(f"  ⏭️  {commit_dir.name} (already exists)")
                continue

            success = process_commit(
                commit_dir,
                positive_hops=args.positive_hops,
                negative_sample_size=args.negative_samples,
                force=args.force
            )
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


def cli():
    """CLI entry point for console script."""
    exit(main())


if __name__ == "__main__":
    cli()
