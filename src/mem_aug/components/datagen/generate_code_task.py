"""
Generate FIM (Fill-in-the-Middle), Code Completion, and Full Function Prediction tasks
from AST data for LLM fine-tuning.

This script processes ast.jsonl files and creates diverse instruction-based tasks
suitable for training code understanding and generation models.
"""

import os
import json
import random
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_lines(code: str) -> int:
    """Count the number of lines in a code snippet."""
    if not code:
        return 0
    return len(code.strip().split('\n'))


def load_ast_entries(ast_file: str, min_lines: int = 5, max_lines: int = 30) -> List[Dict[str, Any]]:
    """
    Load AST entries from ast.jsonl file and filter by code size.
    
    Args:
        ast_file: Path to ast.jsonl file
        min_lines: Minimum number of lines for code to be included
        max_lines: Maximum number of lines for code to be included
        
    Returns:
        List of AST entries with min_lines <= code <= max_lines
    """
    entries = []
    
    try:
        with open(ast_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    code = entry.get('code', '')
                    
                    # Filter by code size
                    line_count = count_lines(code)
                    if code and min_lines <= line_count <= max_lines:
                        entries.append(entry)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    
        logger.info(f"Loaded {len(entries)} AST entries ({min_lines}-{max_lines} lines)")
        return entries
        
    except FileNotFoundError:
        logger.error(f"AST file not found: {ast_file}")
        return []


def find_mask_span(code: str) -> Optional[Tuple[str, str, str]]:
    """
    Find a suitable span to mask in the code for FIM task.
    Returns (prefix, masked_span, suffix) or None if no suitable span found.
    
    Masking strategies:
    1. Function body (between { and })
    2. Statement block
    3. Expression
    """
    lines = code.split('\n')
    
    # Strategy 1: Mask function body
    # Find opening brace and matching closing brace
    brace_depth = 0
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        for char in line:
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    end_idx = i
                    break
        if end_idx != -1:
            break
    
    # If we found a valid function body to mask
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx + 1:
        prefix = '\n'.join(lines[:start_idx + 1])  # Include opening brace
        masked = '\n'.join(lines[start_idx + 1:end_idx])  # Body content
        suffix = '\n'.join(lines[end_idx:])  # Include closing brace
        
        if masked.strip():  # Only return if masked content is non-empty
            return (prefix, masked, suffix)
    
    # Strategy 2: Mask middle portion (simple fallback)
    if len(lines) >= 5:
        mid_start = len(lines) // 3
        mid_end = 2 * len(lines) // 3
        
        prefix = '\n'.join(lines[:mid_start])
        masked = '\n'.join(lines[mid_start:mid_end])
        suffix = '\n'.join(lines[mid_end:])
        
        if masked.strip():
            return (prefix, masked, suffix)
    
    return None


def generate_fim_task(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate a Fill-in-the-Middle task from an AST entry.
    
    Returns:
        Task dict with task_type, ast_id, prefix, middle (to predict), suffix, imports
    """
    code = entry.get('code', '')
    ast_id = entry.get('id', '')
    
    if not code or not ast_id:
        return None
    
    mask_result = find_mask_span(code)
    if not mask_result:
        return None
    
    prefix, middle, suffix = mask_result
    
    # Create FIM task with explicit prefix, middle, suffix
    task = {
        "task_type": "fim",
        "ast_id": ast_id,
        "file": entry.get('file', ''),
        "kind": entry.get('kind', ''),
        "imports": entry.get('imports', []),
        "prefix": prefix,
        "middle": middle,  # This is what needs to be predicted
        "suffix": suffix
    }
    
    return task


def generate_code_completion_task(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate a Code Completion task from an AST entry.
    
    Returns:
        Task dict with task_type, ast_id, input (partial code), output (code to generate), imports
    """
    code = entry.get('code', '')
    ast_id = entry.get('id', '')
    
    if not code or not ast_id:
        return None
    
    lines = code.split('\n')
    
    # Use first 40-60% as context, rest as completion target
    split_point = random.randint(int(len(lines) * 0.4), int(len(lines) * 0.6))
    
    if split_point < 1 or split_point >= len(lines) - 1:
        return None
    
    partial_code = '\n'.join(lines[:split_point])
    code_to_generate = '\n'.join(lines[split_point:])
    
    if not partial_code.strip() or not code_to_generate.strip():
        return None
    
    task = {
        "task_type": "code_completion",
        "ast_id": ast_id,
        "file": entry.get('file', ''),
        "kind": entry.get('kind', ''),
        "imports": entry.get('imports', []),
        "input": partial_code,  # Part of code given as input
        "output": code_to_generate  # Other part to generate
    }
    
    return task


def generate_full_function_task(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate a Full Function Prediction task from an AST entry.
    Predicts the whole function given metadata as input.
    
    Returns:
        Task dict with task_type, ast_id, input (metadata prompt), output (complete function), imports
    """
    code = entry.get('code', '')
    ast_id = entry.get('id', '')
    kind = entry.get('kind', '')
    file_path = entry.get('file', '')
    
    if not code or not ast_id:
        return None
    
    # Create a descriptive prompt based on AST metadata (without imports in text)
    prompt_parts = []
    
    # Add file context
    if file_path:
        prompt_parts.append(f"File: {file_path}")
    
    # Add AST ID as identifier
    prompt_parts.append(f"AST ID: {ast_id}")
    
    # Add kind information
    if kind:
        prompt_parts.append(f"Type: {kind}")
    
    # Add attributes if available
    attributes = entry.get('attributes', [])
    if attributes:
        prompt_parts.append(f"Attributes: {', '.join(attributes)}")
    
    prompt_parts.append("\nGenerate the complete implementation:")
    
    task = {
        "task_type": "full_function_prediction",
        "ast_id": ast_id,
        "file": file_path,
        "kind": kind,
        "imports": entry.get('imports', []),  # Imports as separate field
        "input": '\n'.join(prompt_parts),  # Metadata as input
        "output": code  # Complete function to predict
    }
    
    return task


def generate_tasks_from_commit(
    commit_dir: str,
    max_tasks_per_commit: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate diverse tasks from a single commit directory.
    Task counts are based on the number of diffs in the commit:
    - FIM: 1.5x diff_count (min 2)
    - Code Completion: 1.0x diff_count (min 1)
    - Full Function: 0.3x diff_count (min 1)
    
    Args:
        commit_dir: Path to commit directory containing ast.jsonl and diff_ast.jsonl
        max_tasks_per_commit: Maximum number of tasks to generate per commit
        
    Returns:
        List of generated tasks
    """
    ast_file = os.path.join(commit_dir, 'ast.jsonl')
    diff_ast_file = os.path.join(commit_dir, 'diff_ast.jsonl')
    
    if not os.path.exists(ast_file):
        logger.warning(f"AST file not found: {ast_file}")
        return []
    
    # Count diffs to determine task distribution
    diff_count = 0
    if os.path.exists(diff_ast_file):
        with open(diff_ast_file, 'r') as f:
            diff_count = sum(1 for _ in f)
    
    if diff_count == 0:
        logger.warning(f"No diffs found in {diff_ast_file}, using default counts")
        diff_count = 10  # Default fallback
    
    # Load AST entries
    entries = load_ast_entries(ast_file, min_lines=5, max_lines=30)
    
    if not entries:
        logger.warning(f"No suitable AST entries found in {ast_file}")
        return []
    
    # Shuffle entries for randomness
    random.shuffle(entries)
    
    # Calculate target counts based on diff_count
    # Apply min/max constraints
    max_fim = min(len(entries) // 3, max_tasks_per_commit // 2)
    max_completion = min(len(entries) // 3, max_tasks_per_commit // 3)
    max_function = min(len(entries) // 3, max_tasks_per_commit // 5)
    
    target_counts = {
        "fim": min(max(round(diff_count * 1.5), 2), max_fim),
        "code_completion": min(max(round(diff_count * 1.0), 1), max_completion),
        "full_function_prediction": min(max(round(diff_count * 0.3), 1), max_function)
    }
    
    logger.info(f"Diff count: {diff_count}, Target counts: FIM={target_counts['fim']}, Completion={target_counts['code_completion']}, Function={target_counts['full_function_prediction']}")
    
    tasks = []
    entry_idx = 0
    
    # Generate FIM tasks
    fim_count = 0
    while fim_count < target_counts["fim"] and entry_idx < len(entries):
        task = generate_fim_task(entries[entry_idx])
        if task:
            tasks.append(task)
            fim_count += 1
        entry_idx += 1
    
    # Generate Code Completion tasks
    completion_count = 0
    while completion_count < target_counts["code_completion"] and entry_idx < len(entries):
        task = generate_code_completion_task(entries[entry_idx])
        if task:
            tasks.append(task)
            completion_count += 1
        entry_idx += 1
    
    # Generate Full Function Prediction tasks
    function_count = 0
    while function_count < target_counts["full_function_prediction"] and entry_idx < len(entries):
        task = generate_full_function_task(entries[entry_idx])
        if task:
            tasks.append(task)
            function_count += 1
        entry_idx += 1
    
    logger.info(f"Generated {len(tasks)} tasks: {fim_count} FIM, {completion_count} completion, {function_count} full function")
    
    return tasks


def process_repository(
    repo_name: str,
    base_dir: str = "data/ast_dataset",
    max_tasks_per_commit: int = 100
) -> bool:
    """
    Process all commits in a repository to generate FIM/Completion tasks.
    Tasks are stored in each commit directory alongside the AST data.
    
    Args:
        repo_name: Name of the repository
        base_dir: Base directory containing AST dataset
        max_tasks_per_commit: Maximum tasks per commit
        
    Returns:
        True if successful, False otherwise
    """
    repo_path = os.path.join(base_dir, repo_name)
    
    if not os.path.isdir(repo_path):
        logger.error(f"Repository directory not found: {repo_path}")
        return False
    
    # Find commit directories
    commit_pattern = re.compile(r'^commit_(\d+)$')
    commit_dirs = []
    
    for d in os.listdir(repo_path):
        full_path = os.path.join(repo_path, d)
        if os.path.isdir(full_path) and commit_pattern.match(d):
            commit_dirs.append((d, full_path))
    
    # Sort by commit number
    commit_dirs.sort(key=lambda x: int(commit_pattern.match(x[0]).group(1)))
    
    if not commit_dirs:
        logger.warning(f"No commit directories found in {repo_path}")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing repository: {repo_name}")
    logger.info(f"Found {len(commit_dirs)} commits")
    logger.info(f"{'='*60}\n")
    
    total_tasks = 0
    successful_commits = 0
    
    for commit_name, commit_path in commit_dirs:
        logger.info(f"Processing {commit_name}...")
        
        tasks = generate_tasks_from_commit(
            commit_path,
            max_tasks_per_commit=max_tasks_per_commit
        )
        
        if tasks:
            # Save tasks in the commit directory itself
            commit_output_file = os.path.join(commit_path, "fim_tasks.jsonl")
            with open(commit_output_file, 'w') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')
            
            total_tasks += len(tasks)
            successful_commits += 1
            logger.info(f"  ✓ Saved {len(tasks)} tasks to {commit_output_file}")
    
    if successful_commits > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Repository {repo_name} complete!")
        logger.info(f"  Commits processed: {successful_commits}/{len(commit_dirs)}")
        logger.info(f"  Total tasks generated: {total_tasks}")
        logger.info(f"{'='*60}\n")
        return True
    
    return False


def main(repos: List[str] = None, base_dir: str = "data/ast_dataset"):
    """
    Main entry point for generating FIM/Completion tasks.
    
    Args:
        repos: List of repository names to process (None for all)
        base_dir: Base directory containing AST dataset
    """
    if repos is None:
        # Find all repository directories
        if not os.path.isdir(base_dir):
            logger.error(f"Base directory not found: {base_dir}")
            return
        
        repos = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    if not repos:
        logger.warning("No repositories found to process")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting FIM/Completion task generation")
    logger.info(f"Repositories to process: {len(repos)}")
    logger.info(f"{'='*60}\n")
    
    successful = 0
    for repo in repos:
        if process_repository(repo, base_dir):
            successful += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed: {successful}/{len(repos)} repositories processed")
    logger.info(f"{'='*60}\n")


def cli():
    """Command-line interface for FIM/Completion task generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate FIM, Code Completion, and Full Function Prediction tasks from AST data"
    )
    parser.add_argument(
        "repos",
        nargs="*",
        help="Repository names to process (default: all repositories in base_dir)"
    )
    parser.add_argument(
        "--base-dir",
        default="data/ast_dataset",
        help="Base directory containing AST dataset (default: data/ast_dataset)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum tasks per commit (default: 100)"
    )
    
    args = parser.parse_args()
    
    repos = args.repos if args.repos else None
    main(repos, args.base_dir)


if __name__ == "__main__":
    cli()
