"""
Verification script to validate generated edit sequences against diff_ast data.
"""
import json
import os
from typing import Dict, List, Any
from difflib import SequenceMatcher


def apply_edit_sequence(original_code: str, edit_sequence: List[Dict[str, str]]) -> str:
    """
    Apply edit sequence to original code to reconstruct the after_code.
    
    Args:
        original_code: The original function code
        edit_sequence: List of {span_before, span_after} edits
    
    Returns:
        Reconstructed code after applying all edits
    """
    result = original_code
    
    for edit in edit_sequence:
        span_before = edit['span_before']
        span_after = edit['span_after']
        
        if span_before in result:
            result = result.replace(span_before, span_after, 1)
        elif not span_before:  # Adding new code
            result += span_after
        else:
            print(f"WARNING: Could not find span_before in code:")
            print(f"  Looking for: {span_before[:100]}...")
            print(f"  In code: {result[:100]}...")
            return None
    
    return result


def verify_commit(commit_dir: str) -> Dict[str, Any]:
    """
    Verify edit sequences for a single commit.
    
    Returns:
        Dictionary with verification results
    """
    diff_file = os.path.join(commit_dir, 'diff_ast.jsonl')
    edit_seq_file = os.path.join(commit_dir, 'edit_sequence_dataset.json')
    
    if not os.path.exists(diff_file) or not os.path.exists(edit_seq_file):
        return {"status": "skip", "reason": "missing files"}
    
    # Load diff data
    diff_data = {}
    with open(diff_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                diff_data[entry['id']] = entry
    
    # Load edit sequence data
    with open(edit_seq_file, 'r') as f:
        edit_seq_data = json.load(f)
    
    results = {
        "total": len(edit_seq_data['edit_sequences']),
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    # Verify each edit sequence
    for entry in edit_seq_data['edit_sequences']:
        ast_id = entry['ast_id']
        
        if ast_id not in diff_data:
            results['failed'] += 1
            results['failures'].append({
                "ast_id": ast_id,
                "reason": "AST ID not found in diff_ast.jsonl"
            })
            continue
        
        diff_entry = diff_data[ast_id]
        original_code = entry['original_function']
        edit_sequence = entry['edit_sequence']
        
        # For added functions, original should be empty or "N/A (new code)"
        if diff_entry['status'] == 'added':
            if original_code == "N/A (new code)" or original_code == "":
                # Check if edit sequence produces the after_code
                if len(edit_sequence) == 1 and edit_sequence[0]['span_before'] == "":
                    if edit_sequence[0]['span_after'] == diff_entry['after_code']:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['failures'].append({
                            "ast_id": ast_id,
                            "reason": "Added function: span_after doesn't match after_code",
                            "expected": diff_entry['after_code'][:200],
                            "got": edit_sequence[0]['span_after'][:200]
                        })
                else:
                    results['failed'] += 1
                    results['failures'].append({
                        "ast_id": ast_id,
                        "reason": f"Added function should have 1 edit with empty span_before, got {len(edit_sequence)} edits"
                    })
            else:
                results['failed'] += 1
                results['failures'].append({
                    "ast_id": ast_id,
                    "reason": f"Added function should have empty original_function, got: {original_code[:100]}"
                })
            continue
        
        # For modified/deleted functions
        expected_before = diff_entry['before_code']
        expected_after = diff_entry['after_code']
        
        # Verify original_function matches before_code
        if original_code != expected_before:
            results['failed'] += 1
            results['failures'].append({
                "ast_id": ast_id,
                "reason": "original_function doesn't match before_code",
                "expected_len": len(expected_before),
                "got_len": len(original_code)
            })
            continue
        
        # Apply edit sequence and verify it produces after_code
        reconstructed = apply_edit_sequence(original_code, edit_sequence)
        
        if reconstructed is None:
            results['failed'] += 1
            results['failures'].append({
                "ast_id": ast_id,
                "reason": "Failed to apply edit sequence (span_before not found)"
            })
            continue
        
        if reconstructed == expected_after:
            results['passed'] += 1
        else:
            # Check similarity
            matcher = SequenceMatcher(None, reconstructed, expected_after)
            similarity = matcher.ratio()
            
            results['failed'] += 1
            results['failures'].append({
                "ast_id": ast_id,
                "reason": f"Reconstructed code doesn't match after_code (similarity: {similarity:.2%})",
                "expected_len": len(expected_after),
                "got_len": len(reconstructed),
                "expected_preview": expected_after[:200],
                "got_preview": reconstructed[:200]
            })
    
    return results


def main():
    """Verify edit sequences for commit_2."""
    commit_dir = "data/ast_dataset/ripgrep/commit_2"
    
    print("="*60)
    print("Verifying Edit Sequences for commit_2")
    print("="*60)
    
    results = verify_commit(commit_dir)
    
    if results.get('status') == 'skip':
        print(f"Skipped: {results['reason']}")
        return
    
    print(f"\nTotal entries: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    if results['failures']:
        print(f"\n{'='*60}")
        print("FAILURES:")
        print("="*60)
        for i, failure in enumerate(results['failures'], 1):
            print(f"\n{i}. AST ID: {failure['ast_id']}")
            print(f"   Reason: {failure['reason']}")
            if 'expected' in failure:
                print(f"   Expected: {failure['expected']}")
                print(f"   Got: {failure['got']}")
            if 'expected_preview' in failure:
                print(f"   Expected preview: {failure['expected_preview']}")
                print(f"   Got preview: {failure['got_preview']}")
    else:
        print("\n✓ All edit sequences are valid!")


if __name__ == "__main__":
    main()
