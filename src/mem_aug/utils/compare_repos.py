"""
Compare two repository locations and extract function-level diffs.
This tool allows direct comparison without going through the commit pipeline.
"""
import os
import sys
import tempfile
from mem_aug.utils.extract_rust_ast import main as extract_ast
from mem_aug.utils.build_commit_diff import main as build_diff


def compare_repos(repo1_path: str, repo2_path: str, output_file: str = "diff_output.jsonl"):
    """
    Compare two repository locations and generate function-level diffs.

    Args:
        repo1_path: Path to the first repository (before state)
        repo2_path: Path to the second repository (after state)
        output_file: Path to output JSONL file containing diffs
    """
    # Validate paths
    if not os.path.isdir(repo1_path):
        print(f"Error: Repository path does not exist: {repo1_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(repo2_path):
        print(f"Error: Repository path does not exist: {repo2_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Comparing repositories:")
    print(f"  Before: {repo1_path}")
    print(f"  After:  {repo2_path}")
    print(f"{'='*60}\n")

    # Create temporary directory for AST files
    with tempfile.TemporaryDirectory() as temp_dir:
        ast1_file = os.path.join(temp_dir, "repo1_ast.jsonl")
        ast2_file = os.path.join(temp_dir, "repo2_ast.jsonl")

        # Extract AST from both repositories
        print(f"[1/3] Extracting AST from {repo1_path}...")
        extract_ast(repo1_path, ast1_file)

        print(f"\n[2/3] Extracting AST from {repo2_path}...")
        extract_ast(repo2_path, ast2_file)

        # Compare ASTs and generate diff
        print(f"\n[3/3] Generating diffs...")
        build_diff(ast1_file, ast2_file, output_file)

    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for comparing repositories."""
    if len(sys.argv) < 3:
        print("Usage: compare-repos <repo1_path> <repo2_path> [output_file]", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  compare-repos /path/to/repo1 /path/to/repo2 diff.jsonl", file=sys.stderr)
        print("  compare-repos ./before ./after", file=sys.stderr)
        sys.exit(1)

    repo1_path = sys.argv[1]
    repo2_path = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "diff_output.jsonl"

    compare_repos(repo1_path, repo2_path, output_file)


if __name__ == "__main__":
    cli()
