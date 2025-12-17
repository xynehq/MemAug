"""
Builds a training dataset from enhanced_tasks.json and fim_tasks.jsonl.
Generates dataset.jsonl with different formats: task, fim, code_completion, and full_completion.
"""

import os
import json
import re
from typing import List, Dict, Any


# System messages for different task types
SYSTEM_MESSAGES = {
    "task": """You are a Hyperswitch Rust Repository Master - an elite Rust developer with deep expertise in the Hyperswitch payment orchestration platform. You possess comprehensive knowledge of:

- **Architecture**: Payment routing, connector integrations, merchant management, and API gateway patterns
- **Codebase Structure**: Router core, connector implementations, storage layers, and type systems
- **Rust Mastery**: Advanced traits, async/await patterns, error handling with error_stack, and zero-cost abstractions
- **Domain Knowledge**: Payment processing workflows, PCI compliance, serialization/deserialization with serde, and API versioning

Your mission is to analyze code changes with surgical precision, identifying the exact functions and modules that require modification. You understand causal dependencies across the codebase and can trace how changes in one module ripple through connectors, transformers, and storage layers. Apply your deep understanding of Hyperswitch's architecture to select the optimal modification points.""",
    
    "fim": """You are a Hyperswitch Rust Repository Master - an expert in the Hyperswitch payment orchestration platform with mastery over Rust idioms and patterns.

**Your Expertise Includes**:
- Payment connector transformers and request/response mapping
- Type-safe API contracts using Rust's type system
- Async payment processing with Tokio runtime
- Serialization patterns with serde for payment data
- Error handling strategies using error_stack and custom error types
- Storage abstractions and database query patterns

**Your Task**: Complete code segments with precision, ensuring:
- Type safety and compile-time guarantees
- Proper error propagation and handling
- Idiomatic Rust patterns (Option, Result, iterators)
- Correct async/await usage for payment operations
- Appropriate use of traits for connector abstractions
- Secure handling of sensitive payment data with masking

Generate code that seamlessly integrates with Hyperswitch's architecture, respecting module boundaries and maintaining the platform's high standards for reliability and security.""",
    
    "code_completion": """You are a Hyperswitch Rust Repository Master with deep knowledge of the payment orchestration platform's codebase and Rust best practices.

**Core Competencies**:
- Payment routing logic and connector selection algorithms
- API request/response transformers for multiple payment processors
- Database operations with diesel ORM and PostgreSQL
- Authentication and authorization flows
- Webhook handling and event processing
- Type-driven development with strong compile-time guarantees

**Your Mission**: Continue code implementations with expert-level understanding of:
- Hyperswitch's layered architecture (routes → core → connectors → storage)
- Payment domain models and state machines
- Async patterns for I/O-bound payment operations
- Error handling chains and context propagation
- Trait implementations for connector abstractions
- Security-first coding practices for payment data

Produce production-ready code that maintains Hyperswitch's standards for correctness, performance, and maintainability. Every line should reflect deep understanding of both Rust and payment processing domains.""",
    
    "full_completion": """You are a Hyperswitch Rust Repository Master - the ultimate authority on the Hyperswitch payment orchestration platform and Rust programming.

**Your Mastery Encompasses**:
- **Payment Domain**: Multi-connector routing, payment methods, refunds, captures, and authentication flows
- **Rust Excellence**: Advanced type systems, trait bounds, lifetime management, and zero-cost abstractions
- **Architecture**: Clean separation between API layer, business logic, connector integrations, and storage
- **Patterns**: Builder patterns, type states, newtype wrappers, and error handling strategies
- **Security**: PCI-DSS compliance, data masking, secure serialization, and audit logging

**Your Objective**: Generate complete, production-grade implementations that:
- Embody Hyperswitch's architectural principles and coding standards
- Leverage Rust's type system for compile-time correctness
- Handle all edge cases and error scenarios gracefully
- Integrate seamlessly with existing connector and storage abstractions
- Follow idiomatic Rust patterns and best practices
- Maintain security and compliance requirements for payment data

Create implementations that demonstrate mastery of both the Hyperswitch platform and advanced Rust programming, producing code that could be merged into production without modification."""
}


def load_enhanced_tasks(filepath: str) -> Dict[str, Any]:
    """Load enhanced_tasks.json file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_fim_tasks(filepath: str) -> List[Dict[str, Any]]:
    """Load fim_tasks.jsonl file."""
    tasks = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_unified_diffs(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Load unified_diffs.jsonl and create a lookup by ast_id."""
    diffs = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    diff_data = json.loads(line)
                    diffs[diff_data['id']] = diff_data
    return diffs


def format_task_entry(task: Dict[str, Any], commit_id: str, commit_no: int, diffs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Format a task entry from enhanced_tasks.json."""
    task_name = task['task']
    reasoning = task['reasoning_selection']
    selected_functions = task['selected_functions']
    reasoning_update = task.get('reasoning_update', {})
    
    # Build the worker section with selected functions and their reasoning
    worker_output = ""
    for func_id in selected_functions:
        worker_output += f"<|selected|>{func_id}\n"
        
        # Add context from unified_diffs if available
        if func_id in diffs:
            diff_data = diffs[func_id]
            diff_str = diff_data.get('diff_string', '')
            if diff_str:
                worker_output += f"<|context|>{diff_str}<|context_end|>\n"
        
        # Add reasoning for this function
        if func_id in reasoning_update:
            worker_output += f"<|reasoning|>plan\n{reasoning_update[func_id]}<|reasoning_end|>\n"
        
        # Add action/edit placeholder
        if func_id in diffs:
            diff_data = diffs[func_id]
            diff_str = diff_data.get('diff_string', '')
            if diff_str:
                worker_output += f"<|action|>edit\n{diff_str}\n"
        
        worker_output += "<|selection_end|>\n"
    
    # Format the complete conversation
    instruction = f"""<|im_start|>system
{SYSTEM_MESSAGES['task']}<|im_end|>"""
    
    input_text = f"""<|im_start|>user
{task_name}<|im_end|>"""
    
    output = f"""<|im_start|>selector
<|reasoning|>analysis
To fix failing CI builds related to conditional compilation, {reasoning}<|reasoning_end|>
<|select|>{'<|select|>'.join(selected_functions)}<|im_end|>
<|im_start|>worker
{worker_output}<|im_end|>"""
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "commit_id": commit_id,
        "commit_no": commit_no,
        "task_type": "task"
    }


def format_fim_entry(fim_task: Dict[str, Any], commit_id: str, commit_no: int) -> Dict[str, Any]:
    """Format a FIM task entry."""
    ast_id = fim_task['ast_id']
    prefix = fim_task['prefix']
    middle = fim_task['middle']
    suffix = fim_task['suffix']
    imports = fim_task.get('imports', [])
    
    imports_str = '\n'.join(imports) if imports else ''
    
    instruction = f"""<|im_start|>system
{SYSTEM_MESSAGES['fim']}<|im_end|>"""
    
    input_text = f"""<|im_start|>user
Task: Fill in the middle part of the code
Instruction: Predict the middle part of the code snippet given the prefix and suffix. Include any required imports in the output.
AST ID: {ast_id}
<|fim_prefix|>
{prefix}
<|fim_suffix|>
{suffix}
<|fim_middle|>"""
    
    output = f"""{middle}
<|imports|>
{imports_str}
<|im_end|>"""
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "commit_id": commit_id,
        "commit_no": commit_no,
        "task_type": "fim"
    }


def format_code_completion_entry(task: Dict[str, Any], commit_id: str, commit_no: int) -> Dict[str, Any]:
    """Format a code completion task entry."""
    ast_id = task['ast_id']
    input_code = task['input']
    output_code = task['output']
    imports = task.get('imports', [])
    
    imports_str = '\n'.join(imports) if imports else ''
    
    instruction = f"""<|im_start|>system
{SYSTEM_MESSAGES['code_completion']}<|im_end|>"""
    
    input_text = f"""<|im_start|>user
Task: Complete the code
Instruction: Predict the code for the given ast_id
AST ID: {ast_id}
<|im_end|>
<|im_start|>worker
<|code|>
{input_code}
Output:"""
    
    output = f"""{output_code}
<|imports|>
{imports_str}
<|im_end|>"""
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "commit_id": commit_id,
        "commit_no": commit_no,
        "task_type": "code_completion"
    }


def format_full_completion_entry(task: Dict[str, Any], commit_id: str, commit_no: int) -> Dict[str, Any]:
    """Format a full function prediction task entry."""
    ast_id = task['ast_id']
    input_prompt = task['input']
    full_code = task['output']
    imports = task.get('imports', [])
    
    imports_str = '\n'.join(imports) if imports else ''
    
    instruction = f"""<|im_start|>system
{SYSTEM_MESSAGES['full_completion']}<|im_end|>"""
    
    input_text = f"""<|im_start|>user
Task: Generate complete implementation
Instruction: Predict the whole code for the given
AST ID: {ast_id}
<|im_end|>
<|im_start|>worker
Output:"""
    
    output = f"""<|imports|>
{imports_str}
<|code|>
{full_code}
<|im_end|>"""
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "commit_id": commit_id,
        "commit_no": commit_no,
        "task_type": "full_completion"
    }


def process_commit_directory(commit_dir: str) -> List[Dict[str, Any]]:
    """Process a single commit directory and generate dataset entries."""
    dataset_entries = []
    
    # Extract commit number from directory name (e.g., commit_3 -> 3)
    commit_dir_name = os.path.basename(commit_dir)
    commit_pattern = re.compile(r"^commit_(\d+)$")
    match = commit_pattern.match(commit_dir_name)
    commit_no = int(match.group(1)) if match else 0
    
    # Load commit data to get commit_id
    commit_data_path = os.path.join(commit_dir, 'commit_data.json')
    if not os.path.exists(commit_data_path):
        print(f"Warning: commit_data.json not found in {commit_dir}")
        return dataset_entries
    
    with open(commit_data_path, 'r') as f:
        commit_data = json.load(f)
        commit_id = commit_data.get('commit_hash', 'unknown')
    
    # Load unified diffs
    unified_diffs_path = os.path.join(commit_dir, 'unified_diffs.jsonl')
    diffs = load_unified_diffs(unified_diffs_path)
    
    # Process enhanced_tasks.json
    enhanced_tasks_path = os.path.join(commit_dir, 'enhanced_tasks.json')
    if os.path.exists(enhanced_tasks_path):
        enhanced_data = load_enhanced_tasks(enhanced_tasks_path)
        for task in enhanced_data.get('tasks', []):
            entry = format_task_entry(task, commit_id, commit_no, diffs)
            dataset_entries.append(entry)
    
    # Process fim_tasks.jsonl
    fim_tasks_path = os.path.join(commit_dir, 'fim_tasks.jsonl')
    if os.path.exists(fim_tasks_path):
        fim_tasks = load_fim_tasks(fim_tasks_path)
        for fim_task in fim_tasks:
            task_type = fim_task.get('task_type', 'fim')
            
            if task_type == 'fim':
                entry = format_fim_entry(fim_task, commit_id, commit_no)
                dataset_entries.append(entry)
            elif task_type == 'code_completion':
                entry = format_code_completion_entry(fim_task, commit_id, commit_no)
                dataset_entries.append(entry)
            elif task_type == 'full_function_prediction':
                entry = format_full_completion_entry(fim_task, commit_id, commit_no)
                dataset_entries.append(entry)
    
    return dataset_entries


def find_commit_dirs(repo_dir: str) -> List[str]:
    """Find and sort commit directories within a repository directory."""
    commit_pattern = re.compile(r"^commit_(\d+)$")
    commit_dirs = []
    for d in os.listdir(repo_dir):
        full_path = os.path.join(repo_dir, d)
        if os.path.isdir(full_path) and commit_pattern.match(d):
            commit_dirs.append(full_path)
    
    # Sort based on the numeric part of the directory name
    commit_dirs.sort(key=lambda x: int(commit_pattern.search(os.path.basename(x)).group(1)))
    return commit_dirs


def find_repo_dirs(base_dir: str) -> List[str]:
    """Find repository directories within the base directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]


def build_dataset(base_dir: str = "data/ast_dataset", output_file: str = None):
    """
    Build the complete training dataset from all repositories and commits.
    
    Args:
        base_dir: Base directory containing repository data
        output_file: Output file path. If None, uses base_dir/dataset.jsonl
    """
    if output_file is None:
        output_file = os.path.join(base_dir, "dataset.jsonl")
    
    all_entries = []
    
    # Find all repository directories
    repo_dirs = find_repo_dirs(base_dir)
    
    print(f"Found {len(repo_dirs)} repositories")
    
    for repo_dir in repo_dirs:
        repo_name = os.path.basename(repo_dir)
        print(f"\nProcessing repository: {repo_name}")
        
        # Find all commit directories in this repo
        commit_dirs = find_commit_dirs(repo_dir)
        print(f"  Found {len(commit_dirs)} commits")
        
        for commit_dir in commit_dirs:
            commit_name = os.path.basename(commit_dir)
            print(f"    Processing {commit_name}...")
            
            entries = process_commit_directory(commit_dir)
            all_entries.extend(entries)
            print(f"      Generated {len(entries)} dataset entries")
    
    # Write all entries to output file
    print(f"\nWriting {len(all_entries)} entries to {output_file}")
    with open(output_file, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nDataset generation complete!")
    print(f"Total entries: {len(all_entries)}")
    
    # Print statistics
    task_types = {}
    for entry in all_entries:
        task_type = entry.get('task_type', 'unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1
    
    print("\nDataset statistics:")
    for task_type, count in sorted(task_types.items()):
        print(f"  {task_type}: {count}")


def cli():
    """Command-line interface for dataset generation."""
    import sys
    
    base_dir = "data/ast_dataset"
    output_file = None
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    build_dataset(base_dir, output_file)


if __name__ == "__main__":
    cli()
