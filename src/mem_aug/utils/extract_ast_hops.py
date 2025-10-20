"""
Extract AST graph hops for retrieval label generation.

This module analyzes function dependencies to create labels for retrieval loss:
- Level 0: Current function (self-reference)
- Level 1: Direct dependencies (imports, calling functions)
- Level 2: Second-degree dependencies (functions called by level 1)
"""

import json
import re
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict


def extract_function_calls(code: str) -> Set[str]:
    """
    Extract function calls from code.
    
    Args:
        code: Source code string
    
    Returns:
        Set of function names called in the code
    """
    # Pattern for function calls: function_name(
    # Handles: func(), obj.method(), module::function()
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:::[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\('
    
    matches = re.findall(pattern, code)
    
    # Filter out common keywords that aren't function calls
    keywords = {
        'if', 'while', 'for', 'match', 'loop', 'return',
        'let', 'mut', 'const', 'static', 'fn', 'struct',
        'enum', 'impl', 'trait', 'type', 'use', 'mod'
    }
    
    function_calls = set()
    for match in matches:
        # Get the last part after :: if present
        func_name = match.split('::')[-1]
        if func_name not in keywords:
            function_calls.add(match)
    
    return function_calls


def parse_imports(imports: List[str]) -> Set[str]:
    """
    Parse import statements to extract imported function/module names.
    
    Args:
        imports: List of import statements
    
    Returns:
        Set of imported names
    """
    imported_names = set()
    
    for import_stmt in imports:
        # Handle: use std::io::{self, Read, Write};
        # Handle: use module::function;
        # Handle: use crate::module::*;
        
        # Remove 'use ' prefix
        if import_stmt.startswith('use '):
            import_stmt = import_stmt[4:].strip()
        
        # Remove trailing semicolon
        import_stmt = import_stmt.rstrip(';').strip()
        
        # Handle braced imports: {A, B, C}
        if '{' in import_stmt:
            # Extract the part before {
            prefix = import_stmt.split('{')[0].strip()
            # Extract items in braces
            braced = import_stmt.split('{')[1].split('}')[0]
            items = [item.strip() for item in braced.split(',')]
            
            for item in items:
                if item and item != 'self':
                    # Add both full path and just the name
                    if prefix:
                        imported_names.add(f"{prefix.rstrip(':')}{item}")
                    imported_names.add(item)
        else:
            # Simple import: use module::function;
            parts = import_stmt.split('::')
            # Add the last part (the actual imported name)
            if parts:
                imported_names.add(parts[-1])
                # Also add the full path
                imported_names.add(import_stmt)
    
    return imported_names


def get_function_dependencies(
    function_id: str,
    ast_nodes: List[Dict],
    max_hops: int = 2
) -> Dict[int, Set[str]]:
    """
    Get function dependencies up to max_hops levels.
    
    Args:
        function_id: ID of the function to analyze
        ast_nodes: List of all AST nodes from ast.jsonl
        max_hops: Maximum number of hops (default: 2)
    
    Returns:
        Dictionary mapping hop level to set of function IDs:
        {
            0: {current_function_id},
            1: {direct_dependencies},
            2: {second_degree_dependencies}
        }
    """
    # Build lookup dictionaries
    id_to_node = {node['id']: node for node in ast_nodes}
    name_to_ids = defaultdict(list)
    
    # Map function names to their IDs
    for node in ast_nodes:
        if node['kind'] == 'function_item':
            # Extract function name from ID (last part after ::)
            func_name = node['id'].split('::')[-1]
            name_to_ids[func_name].append(node['id'])
    
    # Initialize result
    hops = {0: {function_id}}
    
    if function_id not in id_to_node:
        return hops
    
    current_node = id_to_node[function_id]
    
    # Level 1: Direct dependencies
    level_1 = set()
    
    # 1a. Imported functions
    if 'imports' in current_node:
        imported_names = parse_imports(current_node['imports'])
        for name in imported_names:
            # Try to find matching function IDs
            simple_name = name.split('::')[-1]
            if simple_name in name_to_ids:
                level_1.update(name_to_ids[simple_name])
    
    # 1b. Called functions
    if 'code' in current_node:
        called_functions = extract_function_calls(current_node['code'])
        for call in called_functions:
            # Try to match with function IDs
            simple_name = call.split('::')[-1]
            if simple_name in name_to_ids:
                level_1.update(name_to_ids[simple_name])
    
    # Remove self-reference
    level_1.discard(function_id)
    hops[1] = level_1
    
    # Level 2: Second-degree dependencies (if max_hops >= 2)
    if max_hops >= 2:
        level_2 = set()
        
        for dep_id in level_1:
            if dep_id in id_to_node:
                dep_node = id_to_node[dep_id]
                
                # Get functions called by this dependency
                if 'code' in dep_node:
                    called = extract_function_calls(dep_node['code'])
                    for call in called:
                        simple_name = call.split('::')[-1]
                        if simple_name in name_to_ids:
                            level_2.update(name_to_ids[simple_name])
        
        # Remove functions already in level 0 and 1
        level_2 -= hops[0]
        level_2 -= hops[1]
        hops[2] = level_2
    
    return hops


def get_retrieval_labels(
    function_id: str,
    ast_nodes: List[Dict],
    positive_hops: List[int] = [0, 1],
    negative_sample_size: int = 5
) -> Dict[str, List[str]]:
    """
    Generate retrieval labels for a function.
    
    Args:
        function_id: ID of the function
        ast_nodes: List of all AST nodes
        positive_hops: Which hop levels to use as positive samples (default: [0, 1])
        negative_sample_size: Number of negative samples to generate
    
    Returns:
        Dictionary with:
        {
            'positive': [list of function IDs that should be retrieved],
            'negative': [list of function IDs that should NOT be retrieved],
            'hops': {0: set(), 1: set(), 2: set()}
        }
    """
    # Get dependency hops
    hops = get_function_dependencies(function_id, ast_nodes, max_hops=2)
    
    # Positive samples: functions from specified hop levels
    positive = set()
    for hop_level in positive_hops:
        if hop_level in hops:
            positive.update(hops[hop_level])
    
    # Negative samples: random functions not in any hop level
    all_function_ids = {
        node['id'] for node in ast_nodes 
        if node['kind'] == 'function_item'
    }
    
    # Functions in any hop level
    in_hops = set()
    for hop_set in hops.values():
        in_hops.update(hop_set)
    
    # Candidates for negative samples
    negative_candidates = all_function_ids - in_hops
    
    # Sample negatives
    import random
    negative = list(negative_candidates)
    random.shuffle(negative)
    negative = negative[:negative_sample_size]
    
    return {
        'positive': list(positive),
        'negative': negative,
        'hops': {k: list(v) for k, v in hops.items()}
    }


def analyze_single_function(
    function_id: str,
    ast_file: Path,
    output_file: Optional[Path] = None
) -> Dict:
    """
    Analyze a single function and generate retrieval labels.
    
    Args:
        function_id: ID of the function to analyze
        ast_file: Path to ast.jsonl file
        output_file: Optional path to save results
    
    Returns:
        Dictionary with analysis results
    """
    # Load AST nodes
    ast_nodes = []
    with open(ast_file, 'r') as f:
        for line in f:
            ast_nodes.append(json.loads(line))
    
    # Get retrieval labels
    labels = get_retrieval_labels(function_id, ast_nodes)
    
    # Add metadata
    result = {
        'function_id': function_id,
        'ast_file': str(ast_file),
        'labels': labels,
        'statistics': {
            'total_functions': len([n for n in ast_nodes if n['kind'] == 'function_item']),
            'hop_0_count': len(labels['hops'].get(0, [])),
            'hop_1_count': len(labels['hops'].get(1, [])),
            'hop_2_count': len(labels['hops'].get(2, [])),
            'positive_count': len(labels['positive']),
            'negative_count': len(labels['negative'])
        }
    }
    
    # Save if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved analysis to {output_file}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extract_ast_hops.py <function_id> <ast_file> [output_file]")
        print("\nExample:")
        print("  python extract_ast_hops.py 'src/main.rs::function::main' data/ast_dataset/bat/commit_10/ast.jsonl")
        sys.exit(1)
    
    function_id = sys.argv[1]
    ast_file = Path(sys.argv[2])
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = analyze_single_function(function_id, ast_file, output_file)
    
    print(f"\n=== Analysis for {function_id} ===")
    print(f"Total functions in AST: {result['statistics']['total_functions']}")
    print(f"\nHop levels:")
    print(f"  Level 0 (self): {result['statistics']['hop_0_count']} functions")
    print(f"  Level 1 (direct): {result['statistics']['hop_1_count']} functions")
    print(f"  Level 2 (indirect): {result['statistics']['hop_2_count']} functions")
    print(f"\nRetrieval labels:")
    print(f"  Positive samples: {result['statistics']['positive_count']}")
    print(f"  Negative samples: {result['statistics']['negative_count']}")
    
    if result['labels']['hops'].get(1):
        print(f"\nLevel 1 dependencies:")
        for dep in result['labels']['hops'][1][:5]:
            print(f"  - {dep}")
        if len(result['labels']['hops'][1]) > 5:
            print(f"  ... and {len(result['labels']['hops'][1]) - 5} more")
