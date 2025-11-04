"""
Builds Edit Sequence dataset from AST diffs and reasoning.
This transforms FIM-style data into Edit Sequence format with multi-span edits.
"""
import os
import json
import re
import random
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher


def extract_edit_spans(before_code: str, after_code: str) -> List[Dict[str, str]]:
    """
    Extract multi-span edits line by line using difflib.
    Returns a list of {span_before, span_after} dictionaries.
    """
    if not before_code and after_code:
        # ADD operation
        return [{"span_before": "", "span_after": after_code}]
    if before_code and not after_code:
        # DELETE operation
        return [{"span_before": before_code, "span_after": ""}]
    if not before_code and not after_code:
        return []

    before_lines = before_code.splitlines(keepends=True)
    after_lines = after_code.splitlines(keepends=True)
    
    sm = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    edit_spans = []
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        span_before = ''.join(before_lines[i1:i2])
        span_after = ''.join(after_lines[j1:j2])
        edit_spans.append({
            "span_before": span_before,
            "span_after": span_after
        })
    
    return edit_spans


def create_edit_sequence_entry(
    ast_id: str,
    diff_data: Dict[str, Any],
    reasoning: Optional[str] = None,
    retrieval_data: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Create an Edit Sequence entry from diff data.
    
    Args:
        ast_id: The AST identifier (function_id)
        diff_data: The diff data from diff_ast.jsonl
        reasoning: Optional reasoning text for this function
        retrieval_data: Optional dict containing 'hops' and 'retrieved_labels' from retrieval_labels.json
    
    Returns:
        Edit sequence entry or None if invalid
    """
    before_code = diff_data.get('before_code', '')
    after_code = diff_data.get('after_code', '')
    status = diff_data.get('status', 'modified')
    
    # For added functions, before_code is None/empty
    if status == 'added':
        before_code = ''
    
    # For deleted functions, after_code is None/empty
    if status == 'removed':
        after_code = ''
    
    # Extract edit spans
    edit_sequence = extract_edit_spans(before_code, after_code)
    
    if not edit_sequence:
        return None
    
    # Get metadata
    commit_metadata = diff_data.get('commit_metadata', {})
    
    # Extract hops and retrieved_labels from retrieval_data
    hops = {}
    retrieved_labels = []
    if retrieval_data:
        hops = retrieval_data.get('hops', {})
        retrieved_labels = retrieval_data.get('retrieved_labels', [])
    
    entry = {
        "ast_id": ast_id,
        "original_function": before_code if before_code else "N/A (new code)",
        "reasoning": reasoning if reasoning else "No reasoning provided.",
        "edit_sequence": edit_sequence,
        "hops": hops,
        "retrieved_labels": retrieved_labels,
        "metadata": {
            "commit_hash": commit_metadata.get('commit_hash', ''),
            "author": commit_metadata.get('author_name', ''),
            "commit_message": commit_metadata.get('commit_message', ''),
            "timestamp": commit_metadata.get('commit_date', ''),
            "repository": diff_data.get('file', '').split('/')[0] if '/' in diff_data.get('file', '') else '',
            "file_path": diff_data.get('file', ''),
            "operation_type": status.upper(),
            "kind": diff_data.get('kind', '')
        }
    }
    
    return entry


def load_reasoning_data(commit_dir: str) -> Dict[str, str]:
    """
    Load reasoning data from reasoning.json.
    Returns a dict mapping function_id to reasoning text.
    """
    reasoning_file = os.path.join(commit_dir, 'reasoning.json')
    reasoning_map = {}
    
    if os.path.exists(reasoning_file):
        with open(reasoning_file, 'r') as f:
            data = json.load(f)
            for entry in data.get('function_reasoning', []):
                function_id = entry.get('function_id', '')
                reasoning = entry.get('reasoning', '')
                if function_id and reasoning:
                    reasoning_map[function_id] = reasoning
    
    return reasoning_map


def load_retrieval_labels(commit_dir: str) -> Dict[str, Dict]:
    """
    Load retrieval labels data from retrieval_labels.json.
    Returns a dict mapping function_id to hops data and retrieved_labels.
    """
    retrieval_file = os.path.join(commit_dir, 'retrieval_labels.json')
    retrieval_map = {}
    
    if os.path.exists(retrieval_file):
        with open(retrieval_file, 'r') as f:
            data = json.load(f)
            function_labels = data.get('function_labels', {})
            for function_id, labels in function_labels.items():
                hops = labels.get('hops', {})
                if hops:
                    # Generate retrieved_labels: max 20 entries
                    # Priority: hop0 + hop1, then random from hop2
                    retrieved_labels = []
                    
                    # Add all from hop0
                    hop0 = hops.get('0', [])
                    retrieved_labels.extend(hop0)
                    
                    # Add all from hop1
                    hop1 = hops.get('1', [])
                    retrieved_labels.extend(hop1)
                    
                    # If we have less than 20, fill from hop2 randomly
                    if len(retrieved_labels) < 20:
                        hop2 = hops.get('2', [])
                        remaining_slots = 20 - len(retrieved_labels)
                        
                        if hop2:
                            # Randomly sample from hop2
                            sample_size = min(remaining_slots, len(hop2))
                            hop2_sample = random.sample(hop2, sample_size)
                            retrieved_labels.extend(hop2_sample)
                    
                    # Limit to max 20 entries
                    retrieved_labels = retrieved_labels[:20]
                    
                    retrieval_map[function_id] = {
                        'hops': hops,
                        'retrieved_labels': retrieved_labels
                    }
    
    return retrieval_map


def process_commit_directory(commit_dir: str, repo_name: str) -> List[Dict[str, Any]]:
    """
    Process a single commit directory to extract Edit Sequence entries.
    
    Args:
        commit_dir: Path to commit directory
        repo_name: Name of the repository
    
    Returns:
        List of edit sequence entries
    """
    diff_file = os.path.join(commit_dir, 'diff_ast.jsonl')
    
    # Skip if diff_ast.jsonl doesn't exist
    if not os.path.exists(diff_file):
        return []
    
    # Load reasoning data
    reasoning_map = load_reasoning_data(commit_dir)
    
    # Load retrieval labels (hops data and retrieved_labels)
    retrieval_map = load_retrieval_labels(commit_dir)
    
    edit_sequences = []
    
    # Process each diff entry
    with open(diff_file, 'r') as f:
        for line in f:
            if line.strip():
                diff_data = json.loads(line)
                ast_id = diff_data.get('id', '')
                
                if not ast_id:
                    continue
                
                # Get reasoning for this function
                reasoning = reasoning_map.get(ast_id)
                
                # Get retrieval data (hops and retrieved_labels) for this function
                retrieval_data = retrieval_map.get(ast_id)
                
                # Create edit sequence entry
                entry = create_edit_sequence_entry(ast_id, diff_data, reasoning, retrieval_data)
                
                if entry:
                    # Update repository name in metadata
                    entry['metadata']['repository'] = repo_name
                    edit_sequences.append(entry)
    
    return edit_sequences


def generate_edit_sequence_template(
    edit_sequences: List[Dict[str, Any]],
    tasks: List[str]
) -> str:
    """
    Generate a training template in the Edit Sequence format.
    
    Args:
        edit_sequences: List of edit sequence entries
        tasks: List of commit tasks
    
    Returns:
        Formatted template string
    """
    template_parts = []
    
    # System prompt
    template_parts.append("<|im_start|>system")
    template_parts.append("You are an expert Rust developer. Your task is to complete code changes based on provided context and reasoning. You will be given:")
    template_parts.append("")
    template_parts.append("1. The AST identifier for the function to update")
    template_parts.append("2. The full original function code")
    template_parts.append("3. Reasoning describing what the update must accomplish")
    template_parts.append("4. A list of edits (edit_sequence) specifying which spans to replace")
    template_parts.append("")
    template_parts.append("For each edit, generate the updated code correctly implementing the described changes.")
    template_parts.append("<|im_end|>")
    
    # User prompt with tasks
    template_parts.append("<|im_start|>user")
    template_parts.append("**Commit Tasks:**")
    for i, task in enumerate(tasks, 1):
        template_parts.append(f"{i}. {task}")
    template_parts.append("<|im_end|>")
    
    # Assistant response with edit sequences
    template_parts.append("<|im_start|>assistant")
    template_parts.append("")
    
    for entry in edit_sequences:
        template_parts.append("<update>")
        template_parts.append(f"<ast_id>{entry['ast_id']}</ast_id>")
        template_parts.append("<original_function>")
        template_parts.append(entry['original_function'])
        template_parts.append("</original_function>")
        template_parts.append("<reasoning>")
        template_parts.append(entry['reasoning'])
        template_parts.append("</reasoning>")
        template_parts.append("<edit_sequence>")
        template_parts.append(json.dumps(entry['edit_sequence'], indent=2))
        template_parts.append("</edit_sequence>")
        template_parts.append("</update>")
        template_parts.append("")
    
    template_parts.append("<|im_end|>")
    
    return '\n'.join(template_parts)


def process_repository_edit_sequence(repo_name: str, base_dir: str = 'data/ast_dataset'):
    """
    Process a single repository to generate Edit Sequence dataset.
    
    Args:
        repo_name: Name of the repository to process
        base_dir: Base directory containing AST dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"{'='*60}")
    
    repo_path = os.path.join(base_dir, repo_name)
    
    # Find commit directories
    commit_pattern = re.compile(r'^commit_(\d+)$')
    commit_dirs = []
    
    for d in os.listdir(repo_path):
        if os.path.isdir(os.path.join(repo_path, d)) and commit_pattern.match(d):
            commit_dirs.append(d)
    
    # Sort by commit number
    commit_dirs.sort(key=lambda x: int(commit_pattern.match(x).group(1)))
    
    if not commit_dirs:
        print(f"Skipping {repo_name}: No commits found.")
        return
    
    entries_generated = 0
    commits_processed = 0
    total_edit_spans = 0
    
    # Process each commit directory
    for commit_name in commit_dirs:
        commit_dir = os.path.join(repo_path, commit_name)
        
        # Process the commit
        edit_sequences = process_commit_directory(commit_dir, repo_name)
        
        if edit_sequences:
            # Load tasks for template generation
            task_file = os.path.join(commit_dir, 'task.json')
            tasks = []
            
            if os.path.exists(task_file):
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    tasks = task_data.get('tasks', [])
            
            # Calculate total number of edits across all entries
            total_edits = sum(len(entry['edit_sequence']) for entry in edit_sequences)
            
            # Save edit sequence dataset
            dataset = {
                "repository": repo_name,
                "commit": commit_name,
                "edit_sequences": edit_sequences,
                "entry_count": len(edit_sequences),
                "total_edit_spans": total_edits
            }
            
            output_file = os.path.join(commit_dir, 'edit_sequence_dataset.json')
            with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            # Generate and save template
            template = generate_edit_sequence_template(edit_sequences, tasks)
            template_file = os.path.join(commit_dir, 'edit_sequence_template.txt')
            with open(template_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(template)
            
            # print(f"  ✓ Generated Edit Sequence dataset: {commit_name} ({len(edit_sequences)} entries, {total_edits} edit spans)")
            entries_generated += len(edit_sequences)
            total_edit_spans += total_edits
            commits_processed += 1
    
    print(f"\nCompleted {repo_name}: {entries_generated} entries, {total_edit_spans} edit spans across {commits_processed} commits")


def run_edit_sequence_pipeline(repos: List[str] = None, base_dir: str = 'data/ast_dataset'):
    """
    Run the Edit Sequence dataset generation pipeline.
    
    Args:
        repos: List of repository names to process. If None, processes all repos.
        base_dir: Base directory containing AST dataset
    """
    if repos is None:
        # Find all repository directories
        if not os.path.isdir(base_dir):
            print(f"Error: Base directory '{base_dir}' not found.")
            return
        
        repo_names = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))]
        
        if not repo_names:
            print(f"No repository data found in {base_dir}.")
            print("Please run 'generate-ast' and 'generate-ast-diffs' first.")
            return
    else:
        repo_names = repos
    
    print(f"\n{'='*60}")
    print(f"Starting Edit Sequence dataset generation for {len(repo_names)} repositories")
    print(f"{'='*60}")
    
    for repo_name in repo_names:
        process_repository_edit_sequence(repo_name, base_dir)
    
    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for Edit Sequence dataset generation."""
    import sys
    
    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        run_edit_sequence_pipeline(repos)
    else:
        # Process all repos
        run_edit_sequence_pipeline()


if __name__ == "__main__":
    cli()
