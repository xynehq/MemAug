"""
Builds FIM (Fill-in-the-Middle) dataset from AST diffs.
This is Step 3 of the pipeline after generate_ast_diffs.py.
"""
import os
import json
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

MARGIN = 3  # Number of context lines around the change


def extract_imports_from_code(code: str) -> str:
    """Extract import statements from code."""
    lines = code.split('\n')
    imports = []

    for line in lines:
        line = line.strip()
        if (line.startswith('import ') or
            line.startswith('from ') or
            line.startswith('use ') or
            line.startswith('#include')):
            imports.append(line)

    return '\n'.join(imports) if imports else ""


def find_exact_changes_in_span(before_span: str, after_span: str) -> Tuple[int, int, int, int]:
    """
    Find the exact line indices within the spans where changes occur.
    Returns (before_change_start, before_change_end, after_change_start, after_change_end)
    """
    before_lines = before_span.split('\n')
    after_lines = after_span.split('\n')

    matcher = SequenceMatcher(None, before_lines, after_lines, autojunk=False)

    # Find the first and last changed line indices
    before_changed = []
    after_changed = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            before_changed.extend(range(i1, i2))
            after_changed.extend(range(j1, j2))

    if not before_changed and not after_changed:
        # No changes found (shouldn't happen, but handle gracefully)
        return 0, len(before_lines), 0, len(after_lines)

    # Get the range of changed lines
    before_start = min(before_changed) if before_changed else 0
    before_end = max(before_changed) + 1 if before_changed else len(before_lines)
    after_start = min(after_changed) if after_changed else 0
    after_end = max(after_changed) + 1 if after_changed else len(after_lines)

    return before_start, before_end, after_start, after_end


def create_fim_sample(diff_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a FIM format sample from diff data with context lines around the exact change."""
    file_path = diff_data.get('file', '')
    status = diff_data.get('status', 'modified')

    # Map status to operation type
    status_to_op = {
        'modified': 'UPDATE',
        'added': 'ADD',
        'removed': 'DELETE'
    }
    operation_type = status_to_op.get(status, 'UPDATE')

    before_code = diff_data.get('before_code', '')
    after_code = diff_data.get('after_code', '')
    diff_span = diff_data.get('diff_span', {})

    # Handle ADDED functions - entire function is new
    if status == 'added':
        if not after_code:
            return None

        imports = extract_imports_from_code(after_code)
        return {
            "file_name": file_path,
            "imports": imports,
            "operation_type": operation_type,
            "fim_prefix": "",
            "fim_suffix": "",
            "fim_middle": after_code,
            "metadata": {
                "commit_hash": diff_data.get('commit_metadata', {}).get('commit_hash', ''),
                "author": diff_data.get('commit_metadata', {}).get('author_name', ''),
                "commit_message": diff_data.get('commit_metadata', {}).get('commit_message', ''),
                "function_id": diff_data.get('id', ''),
                "kind": diff_data.get('kind', ''),
                "diff_context": {
                    "lines_before_change": 0,
                    "lines_after_change": 0,
                    "exact_change_start": 0,
                    "exact_change_end": len(after_code.split('\n')),
                    "context_window": "entire function (new)"
                }
            }
        }

    # Handle DELETED functions - entire function is removed
    if status == 'removed':
        if not before_code:
            return None

        imports = extract_imports_from_code(before_code)
        return {
            "file_name": file_path,
            "imports": imports,
            "operation_type": operation_type,
            "fim_prefix": "",
            "fim_suffix": "",
            "fim_middle": "",  # Empty because function was deleted
            "metadata": {
                "commit_hash": diff_data.get('commit_metadata', {}).get('commit_hash', ''),
                "author": diff_data.get('commit_metadata', {}).get('author_name', ''),
                "commit_message": diff_data.get('commit_metadata', {}).get('commit_message', ''),
                "function_id": diff_data.get('id', ''),
                "kind": diff_data.get('kind', ''),
                "diff_context": {
                    "lines_before_change": 0,
                    "lines_after_change": 0,
                    "exact_change_start": 0,
                    "exact_change_end": len(before_code.split('\n')),
                    "context_window": "entire function (deleted)",
                    "deleted_code": before_code
                }
            }
        }

    # Handle MODIFIED functions - only changed lines
    if not diff_span or 'before' not in diff_span or 'after' not in diff_span:
        return None

    before_span = diff_span['before']
    after_span = diff_span['after']

    # If both before and after are empty (imports-only change), skip
    if not before_span and not after_span:
        return None

    # Split spans into lines
    before_span_lines = before_span.split('\n')
    after_span_lines = after_span.split('\n')

    # Find exactly which lines within the span changed
    before_change_start, before_change_end, after_change_start, after_change_end = \
        find_exact_changes_in_span(before_span, after_span)

    # Extract unchanged prefix lines from the span
    span_prefix_lines = before_span_lines[:before_change_start]

    # Extract the actual changed lines from after_span
    changed_lines = after_span_lines[after_change_start:after_change_end]

    # Extract unchanged suffix lines from the span
    span_suffix_lines = after_span_lines[after_change_end:]

    # Find where the before_span appears in the full function code
    before_lines = before_code.split('\n')
    span_start_idx = -1

    for i in range(len(before_lines)):
        if i + len(before_span_lines) <= len(before_lines):
            section = '\n'.join(before_lines[i:i+len(before_span_lines)])
            if section == before_span:
                span_start_idx = i
                break

    if span_start_idx == -1:
        return None

    # Calculate positions in the full function
    actual_change_start = span_start_idx + before_change_start
    actual_change_end = span_start_idx + before_change_end

    # Add additional context from the full function
    context_start = max(0, span_start_idx - MARGIN)
    context_end = min(len(before_lines), span_start_idx + len(before_span_lines) + MARGIN)

    # Build complete prefix: context before span + unchanged prefix from span
    prefix_context_lines = before_lines[context_start:span_start_idx]
    fim_prefix_lines = prefix_context_lines + span_prefix_lines

    # Build complete suffix: unchanged suffix from span + context after span
    span_end_idx = span_start_idx + len(before_span_lines)
    suffix_context_lines = before_lines[span_end_idx:context_end]
    fim_suffix_lines = span_suffix_lines + suffix_context_lines

    fim_prefix = '\n'.join(fim_prefix_lines)
    fim_suffix = '\n'.join(fim_suffix_lines)
    fim_middle = '\n'.join(changed_lines)

    # Extract imports from the full file
    imports = extract_imports_from_code(after_code)

    return {
        "file_name": file_path,
        "imports": imports,
        "operation_type": operation_type,
        "fim_prefix": fim_prefix,
        "fim_suffix": fim_suffix,
        "fim_middle": fim_middle,
        "metadata": {
            "commit_hash": diff_data.get('commit_metadata', {}).get('commit_hash', ''),
            "author": diff_data.get('commit_metadata', {}).get('author_name', ''),
            "commit_message": diff_data.get('commit_metadata', {}).get('commit_message', ''),
            "function_id": diff_data.get('id', ''),
            "kind": diff_data.get('kind', ''),
            "diff_context": {
                "lines_before_change": len(fim_prefix_lines),
                "lines_after_change": len(fim_suffix_lines),
                "exact_change_start": actual_change_start,
                "exact_change_end": actual_change_end,
                "context_window": f"[{context_start}:{context_end}] of function"
            }
        }
    }


def create_function_series(changes: List[Dict[str, Any]], function_id: str) -> List[Dict[str, Any]]:
    """Create a series of FIM samples for multiple changes in the same function with proper chaining."""
    fim_samples = []

    # Use the original order from the input file for sequential changes
    sorted_changes = changes

    # Process each change
    for i, change in enumerate(sorted_changes):
        fim_sample = create_fim_sample(change)
        if fim_sample:
            # Add series metadata
            fim_sample["metadata"]["series_info"] = {
                "step": i + 1,
                "total_steps": len(sorted_changes),
                "function_id": function_id,
                "change_sequence": [c.get('id', '') for c in sorted_changes],
                "is_series": True
            }
            fim_samples.append(fim_sample)

    return fim_samples


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


def process_commit_directory(commit_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Process a single commit directory to extract FIM samples."""
    diff_file = os.path.join(commit_dir, 'diff_ast.jsonl')

    # Skip if diff_ast.jsonl doesn't exist
    if not os.path.exists(diff_file):
        return {}

    # Group diff data by function
    function_changes = {}

    # Process diff data and group by function
    with open(diff_file, 'r') as f:
        for line in f:
            if line.strip():
                diff_data = json.loads(line)
                function_id = diff_data.get('id', '')

                if function_id not in function_changes:
                    function_changes[function_id] = []
                function_changes[function_id].append(diff_data)

    # Create a dictionary structure with function_id as key and array of changes as value
    fim_functions = {}

    # Process each function's changes
    for function_id, changes in function_changes.items():
        if len(changes) == 1:
            # Single change in function - create one FIM sample
            fim_sample = create_fim_sample(changes[0])
            if fim_sample:
                fim_functions[function_id] = [fim_sample]
        else:
            # Multiple changes in same function - create series of samples with proper chaining
            series_samples = create_function_series(changes, function_id)
            if series_samples:
                fim_functions[function_id] = series_samples

    return fim_functions


def process_repository_fim(repo_name: str, base_dir: str = 'data/ast_dataset'):
    """
    Processes a single repository to generate FIM dataset from AST diffs.

    Args:
        repo_name: Name of the repository to process
        base_dir: Base directory containing AST dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"{'='*60}")

    repo_path = os.path.join(base_dir, repo_name)
    commit_dirs = find_commit_dirs(repo_path)

    if not commit_dirs:
        print(f"Skipping {repo_name}: No commits found.")
        return

    samples_generated = 0
    commits_processed = 0

    # Process each commit directory
    for commit_name in commit_dirs:
        commit_dir = os.path.join(repo_path, commit_name)
        diff_file = os.path.join(commit_dir, 'diff_ast.jsonl')

        # Skip if diff_ast.jsonl doesn't exist
        if not os.path.exists(diff_file):
            continue

        # Process the commit
        fim_functions = process_commit_directory(commit_dir)

        if fim_functions:
            # Calculate total samples for this commit
            commit_samples = sum(len(samples) for samples in fim_functions.values())

            # Save dataset in the same directory as diff_ast.jsonl
            dataset = {
                "repository": repo_name,
                "commit": commit_name,
                "fim_functions": fim_functions,
                "function_count": len(fim_functions),
                "sample_count": commit_samples
            }

            output_file = os.path.join(commit_dir, 'fim_dataset.json')
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)

            print(f"  ✓ Generated FIM dataset: {commit_name} ({commit_samples} samples in {len(fim_functions)} functions)")
            samples_generated += commit_samples
            commits_processed += 1

    print(f"\nCompleted {repo_name}: {samples_generated} samples generated across {commits_processed} commits")


def run_fim_pipeline(repos: List[str] = None, base_dir: str = 'data/ast_dataset'):
    """
    Runs the FIM dataset generation pipeline for specified repositories or all repositories.

    Args:
        repos: List of repository names to process. If None, processes all repos.
        base_dir: Base directory containing AST dataset
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
    print(f"Starting FIM dataset generation for {len(repo_names)} repositories")
    print(f"{'='*60}")

    for repo_name in repo_names:
        process_repository_fim(repo_name, base_dir)

    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for FIM dataset generation."""
    import sys

    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        run_fim_pipeline(repos)
    else:
        # Process all repos
        run_fim_pipeline()


if __name__ == "__main__":
    cli()
