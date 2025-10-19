"""
Build Training Template for Qwen 2.5 Coder.
Generates FIM training templates from commit data.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path: Path) -> Dict[str, Any]:
    """Load JSONL file and create a dictionary keyed by function_id."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data[entry['id']] = entry
    return data


def escape_text(text: str) -> str:
    """Escape text for template output."""
    # No escaping needed for the template, just return as-is
    return text


def build_system_prompt() -> str:
    """Build the system prompt."""
    return """You are an expert Rust developer. Your task is to complete code changes based on provided context and reasoning. You will be given:
1. A list of tasks to accomplish for this commit
2. For each function modification, you'll see the AST identifier, before-code context, reasoning, and Fill-in-Middle samples

Complete each FIM by generating the middle portion that correctly implements the described changes."""


def build_task_list(tasks: List[str]) -> str:
    """Build the task list section."""
    task_lines = []
    for i, task in enumerate(tasks, 1):
        task_lines.append(f"{i}. {task}")
    return "\n".join(task_lines)


def build_update_block(
    function_id: str,
    reasoning_text: str,
    before_code: str,
    fim_samples: List[Dict[str, Any]]
) -> str:
    """Build an UPDATE operation block."""
    block = f"<update>\n"
    block += f"<ast_id>{function_id}</ast_id>\n"
    block += f"<context>\n{escape_text(before_code)}\n</context>\n"
    block += f"<reasoning>\n{escape_text(reasoning_text)}\n</reasoning>\n"

    # Add FIM samples (separated by <sep> if multiple)
    for i, fim in enumerate(fim_samples):
        if i > 0:
            block += "<sep>\n"

        prefix = escape_text(fim['fim_prefix'])+"\n"
        suffix = escape_text(fim['fim_suffix'])+"\n"
        middle = escape_text(fim['fim_middle'])+"\n"

        block += f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}\n"

    block += "</update>\n"
    return block


def build_add_block(
    function_id: str,
    reasoning_text: str,
    fim_samples: List[Dict[str, Any]]
) -> str:
    """Build an ADD operation block."""
    block = f"<add>\n"
    block += f"<ast_id>{function_id}</ast_id>\n"
    block += f"<context>\nN/A (new code)\n</context>\n"
    block += f"<reasoning>\n{escape_text(reasoning_text)}\n</reasoning>\n"

    # Add FIM samples (separated by <sep> if multiple)
    for i, fim in enumerate(fim_samples):
        if i > 0:
            block += "<sep>\n"

        prefix = escape_text(fim['fim_prefix'])
        suffix = escape_text(fim['fim_suffix'])
        middle = escape_text(fim['fim_middle'])

        block += f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}\n"

    block += "</add>\n"
    return block


def build_delete_block(
    function_id: str,
    reasoning_text: str,
    before_code: str,
    fim_samples: List[Dict[str, Any]]
) -> str:
    """Build a DELETE operation block."""
    block = f"<delete>\n"
    block += f"<ast_id>{function_id}</ast_id>\n"
    block += f"<context>\n{escape_text(before_code)}\n</context>\n"
    block += f"<reasoning>\n{escape_text(reasoning_text)}\n</reasoning>\n"

    # Add FIM samples (separated by <sep> if multiple)
    for i, fim in enumerate(fim_samples):
        if i > 0:
            block += "<sep>\n"

        prefix = escape_text(fim['fim_prefix'])
        suffix = escape_text(fim['fim_suffix'])
        middle = escape_text(fim['fim_middle'])

        block += f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}\n"

    block += "</delete>\n"
    return block


def build_template(commit_dir: Path) -> str:
    """Build the complete training template for a commit."""

    # Load all necessary files
    task_data = load_json(commit_dir / "task.json")
    reasoning_data = load_json(commit_dir / "reasoning.json")
    fim_data = load_json(commit_dir / "fim_dataset.json")
    diff_ast_data = load_jsonl(commit_dir / "diff_ast.jsonl")

    # Extract data
    tasks = task_data['tasks']
    reasoning_map = {r['function_id']: r for r in reasoning_data['function_reasoning']}
    fim_functions = fim_data['fim_functions']

    # Start building template
    template = ""

    # System message
    template += "<|im_start|>system\n"
    template += build_system_prompt()
    template += "<|im_end|>\n"

    # User message with task list
    template += "<|im_start|>user\n"
    template += "**Commit Tasks:**\n"
    template += build_task_list(tasks)
    template += "<|im_end|>\n"

    # Assistant message with operation blocks
    template += "<|im_start|>assistant\n"

    # Process each function that has FIM samples
    for function_id, fim_samples in fim_functions.items():
        if not fim_samples:
            continue

        # Get operation type from first FIM sample
        operation_type = fim_samples[0]['operation_type']

        # Get reasoning if available
        reasoning_entry = reasoning_map.get(function_id, {})
        reasoning_text = reasoning_entry.get('reasoning', 'No reasoning provided.')

        # Get before_code from diff_ast if available
        diff_entry = diff_ast_data.get(function_id, {})
        before_code = diff_entry.get('before_code', 'N/A')

        # Build appropriate block based on operation type
        if operation_type == 'UPDATE':
            template += "\n"
            template += build_update_block(function_id, reasoning_text, before_code, fim_samples)
        elif operation_type == 'ADD':
            template += "\n"
            template += build_add_block(function_id, reasoning_text, fim_samples)
        elif operation_type == 'DELETE':
            template += "\n"
            template += build_delete_block(function_id, reasoning_text, before_code, fim_samples)

    template += "<|im_end|>\n"

    return template


def main():
    parser = argparse.ArgumentParser(
        description="Build training template for Qwen 2.5 Coder from commit data"
    )
    parser.add_argument(
        "commit_dir",
        type=Path,
        help="Path to commit directory (e.g., data/ast_dataset/ripgrep/commit_7)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: <commit_dir>/template.txt)"
    )

    args = parser.parse_args()

    commit_dir = args.commit_dir
    if not commit_dir.exists():
        print(f"Error: Commit directory not found: {commit_dir}")
        return 1

    # Check required files exist
    required_files = ["task.json", "reasoning.json", "fim_dataset.json", "diff_ast.jsonl"]
    for file in required_files:
        if not (commit_dir / file).exists():
            print(f"Error: Required file not found: {commit_dir / file}")
            return 1

    # Build template
    print(f"Building template for: {commit_dir}")
    template = build_template(commit_dir)

    # Determine output path
    output_path = args.output if args.output else commit_dir / "template.txt"

    # Write template
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"Template written to: {output_path}")
    print(f"Template size: {len(template)} characters")

    return 0


if __name__ == "__main__":
    exit(main())
