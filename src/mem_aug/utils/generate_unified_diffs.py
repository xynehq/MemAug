"""
Generate unified diff strings from diff_ast.jsonl files.
Produces git-style diff output for each code change.
"""
import json
import difflib
from typing import List, Dict, Any


def generate_unified_diff(before_code: str, after_code: str, file_path: str = "file") -> str:
    """
    Generate a unified diff string from before and after code.
    
    Args:
        before_code: The original code
        after_code: The modified code
        file_path: The file path for the diff header
    
    Returns:
        Unified diff string in git diff format
    """
    # Handle None values
    if before_code is None:
        before_code = ""
    if after_code is None:
        after_code = ""
    
    # Split into lines, keeping line endings
    before_lines = before_code.splitlines(keepends=True)
    after_lines = after_code.splitlines(keepends=True)
    
    # Generate unified diff
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=''
    )
    
    # Join the diff lines
    diff_text = '\n'.join(diff)
    
    return diff_text


def generate_simple_unified_diff(before_code: str, after_code: str) -> str:
    """
    Generate a simple unified diff string without file headers.
    Shows only the @@ header and the actual changes.
    
    Args:
        before_code: The original code
        after_code: The modified code
    
    Returns:
        Simple unified diff string
    """
    # Handle None values
    if before_code is None:
        before_code = ""
    if after_code is None:
        after_code = ""
    
    # Split into lines
    before_lines = before_code.splitlines(keepends=False)
    after_lines = after_code.splitlines(keepends=False)
    
    # Generate unified diff
    diff = list(difflib.unified_diff(
        before_lines,
        after_lines,
        lineterm='',
        n=0  # No context lines
    ))
    
    # Remove the file header lines (first two lines: --- and +++)
    if len(diff) >= 2:
        diff = diff[2:]
    
    # Join the diff lines
    diff_text = '\n'.join(diff)
    
    return diff_text


def process_diff_ast_file(jsonl_file: str, simple: bool = False, separate_imports: bool = True) -> List[Dict[str, Any]]:
    """
    Process a diff_ast.jsonl file and generate unified diffs for each entry.
    
    Args:
        jsonl_file: Path to the diff_ast.jsonl file
        simple: If True, generate simple diffs without file headers
        separate_imports: If True, generate separate diff strings for imports and code
    
    Returns:
        List of dictionaries containing id, file, diff_string (for code), and imports_diff_string
    """
    results = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                ast_id = data.get('id', '')
                file_path = data.get('file', 'unknown')
                before_code = data.get('before_code', '')
                after_code = data.get('after_code', '')
                status = data.get('status', 'modified')
                
                # Get import changes
                before_imports = data.get('before_imports', [])
                after_imports = data.get('after_imports', [])
                imports_changed = data.get('imports_changed', False)
                
                # Generate code diff
                if simple:
                    code_diff_string = generate_simple_unified_diff(before_code, after_code)
                else:
                    code_diff_string = generate_unified_diff(before_code, after_code, file_path)
                
                # Generate imports diff separately if imports changed
                imports_diff_string = ""
                if separate_imports and imports_changed and (before_imports or after_imports):
                    before_imports_str = '\n'.join(before_imports)
                    after_imports_str = '\n'.join(after_imports)
                    
                    if simple:
                        imports_diff_string = generate_simple_unified_diff(before_imports_str, after_imports_str)
                    else:
                        imports_diff_string = generate_unified_diff(before_imports_str, after_imports_str, file_path)
                
                results.append({
                    'id': ast_id,
                    'file': file_path,
                    'status': status,
                    'imports_changed': imports_changed,
                    'code_changed': data.get('code_changed', False),
                    'diff_string': code_diff_string,
                    'imports_diff_string': imports_diff_string
                })
    
    return results


def print_diffs(results: List[Dict[str, Any]], show_headers: bool = True):
    """
    Print the unified diffs in a readable format.
    
    Args:
        results: List of diff results
        show_headers: Whether to show headers for each diff
    """
    for i, result in enumerate(results):
        if show_headers:
            print(f"\n{'='*80}")
            print(f"ID: {result['id']}")
            print(f"File: {result['file']}")
            print(f"Status: {result['status']}")
            
            # Show what changed
            changes = []
            if result.get('code_changed'):
                changes.append('code')
            if result.get('imports_changed'):
                changes.append('imports')
            if changes:
                print(f"Changed: {', '.join(changes)}")
            
            print(f"{'='*80}")
        
        # Print imports diff if present
        if result.get('imports_diff_string'):
            print("\n[IMPORTS DIFF]")
            print(result['imports_diff_string'])
        
        # Print code diff
        if result['diff_string']:
            if result.get('imports_diff_string'):
                print("\n[CODE DIFF]")
            print(result['diff_string'])
        else:
            if not result.get('imports_diff_string'):
                print("(No changes)")
        
        if i < len(results) - 1:
            print()


def main():
    """Main function for CLI usage."""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python generate_unified_diffs.py <path_to_diff_ast.jsonl> [--simple] [--no-separate-imports]")
        print("\nOptions:")
        print("  --simple                Generate simple diffs without file headers")
        print("  --no-separate-imports   Don't generate separate diff strings for imports")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    simple = '--simple' in sys.argv
    separate_imports = '--no-separate-imports' not in sys.argv
    
    if not os.path.exists(jsonl_file):
        print(f"Error: File '{jsonl_file}' not found.")
        sys.exit(1)
    
    # Process the file
    results = process_diff_ast_file(jsonl_file, simple=simple, separate_imports=separate_imports)
    
    # Print the results
    print_diffs(results, show_headers=True)
    
    print(f"\n{'='*80}")
    print(f"Total diffs generated: {len(results)}")
    if separate_imports:
        imports_count = sum(1 for r in results if r.get('imports_diff_string'))
        print(f"Diffs with separate imports: {imports_count}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
