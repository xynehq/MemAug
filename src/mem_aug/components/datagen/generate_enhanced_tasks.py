"""
Enhanced task generation system that processes batches of AST elements and generates 
structured tasks with affected elements, reasoning for selection and updates.
"""
import os
import json
import asyncio
import re
import yaml
from typing import List, Dict, Any
from mem_aug.utils.repo_manager import get_project_root

# Try to import OpenAI, but make it optional
try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai and python-dotenv not installed. Install with: pip install openai python-dotenv")

# Global semaphore will be initialized based on config
LLM_SEMAPHORE = None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config" / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('enhanced_tasks', {})
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {
            'max_ast_elements_per_batch': 5,
            'max_related_group_size': 4,
            'enable_operation_type_reasoning': True,
            'max_concurrent_llm_calls': 5,
            'max_concurrent_commits': 2
        }


def initialize_semaphore(config: Dict[str, Any] = None) -> asyncio.Semaphore:
    """Initialize the global semaphore based on configuration."""
    global LLM_SEMAPHORE
    if config is None:
        config = load_config()
    
    max_concurrent = config.get('max_concurrent_llm_calls', 5)
    LLM_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    return LLM_SEMAPHORE

ENHANCED_SYSTEM_PROMPT = """
You are generating LLM TRAINING DATA for code modification tasks. This data will be used to train language models on how to think through and implement code changes.

IMPORTANT: This is LLM training data. You must create complete, instructional examples that teach the model:
1. How to analyze a programming task
2. How to reason about which functions need to be selected/modified
3. How to plan specific changes for each selected function
4. How to think through the implementation step-by-step

You will be given:
1. A batch of AST elements (functions, structs, modules, etc.) from a codebase that may or may not be directly related
2. Commit metadata including author, date, and commit message
3. Context about the repository and programming language

Your goal is to generate SMALL, ATOMIC, INDEPENDENT tasks that serve as training examples:
- Each task demonstrates complete reasoning from problem to solution
- Shows the thinking process for selecting which functions to modify
- Explains the specific changes needed in each selected function
- Can be implemented by an LLM in one focused session
- Are completely independent and can be built in any order

CRITICAL REQUIREMENTS FOR LLM TRAINING DATA:
- Balance task granularity: merge related changes into cohesive tasks, but keep unrelated changes separate
- Each task should represent one logical feature/fix that makes sense to implement together
- Tasks can include multiple related functions if they're part of the same logical change
- Avoid both: (1) overly granular tasks that break logical units, (2) overly complex tasks spanning unrelated features
- Each task must be implementable independently without depending on other tasks
- All reasoning must be explicit and instructional for model learning
- Use the commit message and metadata to infer the **intent** behind changes

INSTRUCTION COMPLETENESS REQUIREMENTS:
- reasoning_selection must teach the model HOW to analyze the ENTIRE CODEBASE and identify relevant functions based on architectural knowledge, NOT by referencing the commit message, task description, or provided AST elements
- The reasoning should demonstrate deep understanding of the repository structure, dependencies, and code relationships AS IF the model is starting from scratch with only the task goal
- Show the thinking process as if the model knows the whole codebase and is selecting functions based on their role in the system, WITHOUT referencing any input data
- NEVER reference "the commit message indicates" or "looking at AST elements" - instead show pure codebase analysis reasoning
- EMPHASIZE STRONG CAUSAL REASONING: Show explicit cause-and-effect chains between functions (e.g., "Function A calls Function B with parameter X, so when A changes its output format, B must be updated to handle the new format", "Function C depends on the return value from Function D, therefore modifying D's behavior requires corresponding changes in C")
- HANDLE ALL OPERATION TYPES: Include reasoning for deleted AST elements ("Function X needs to be removed because it's no longer called after refactoring Y"), new functions ("Function Z must be added to handle the new requirement introduced by changes in Function W"), and modified functions
- For DELETED functions: Explain why the function is no longer needed and which changes made it obsolete
- For NEW functions: Explain what functionality gap needs to be filled and how the new function integrates with existing code
- reasoning_update must provide enough detail for the model to learn implementation patterns
- Include WHY each change is necessary (causal reasoning)
- Include HOW the changes work together (system reasoning)
- Keep reasoning detailed but concise — focus on clarity, not verbosity
- For every function in selected_functions, provide a reasoning_update entry
- Exclude unrelated AST elements (make separate tasks if needed)
- CRITICAL: reasoning_update MUST be a dictionary with AST ID keys, NOT a string
- CRITICAL: When referencing functions in reasoning_selection, ALWAYS use the complete AST ID (e.g., "src/search.rs::Iter<'b, 's>::function::next_line_match"), not just short names like "next_line_match"

Output format (strict JSON only):
{
  "tasks": [
    {
      "task": "Cohesive task description representing one logical change...",
      "reasoning_selection": "Thinking process for selecting which functions are needed...",
      "selected_functions": ["ast_id_1", "ast_id_2"],
      "reasoning_update": {
        "ast_id_1": "What specific change is needed in this function and why (MODIFY/DELETE/ADD)",
        "ast_id_2": "What specific change is needed in this function and why (MODIFY/DELETE/ADD)"
      }
    }
  ]
}

IMPORTANT: End your response with <END> token when generation is complete.

EXAMPLE:
{
  "tasks": [
    {
      "task": "Replace regex find() with shortest_match() in search iterator for performance optimization",
      "reasoning_selection": "To implement shortest match optimization in ripgrep, I need to identify the core search functions and their dependencies. In the ripgrep architecture, src/search.rs contains the primary search implementation through the Iter struct. The function 'src/search.rs::Iter<'b, 's>::function::next_line_match' is the central regex matching function that currently uses regex.find() to locate pattern matches and return full match ranges as (start, end) tuples. For performance optimization, this function needs to switch to regex.shortest_match() which only returns the end position. CAUSAL DEPENDENCY: The function 'src/search.rs::Iter<'b, 's>::function::find_line' directly consumes the match position output from next_line_match - specifically, it receives the (start, end) tuple and uses both values to calculate line boundaries. Therefore, when next_line_match changes from returning (start, end) to returning only end position, find_line will receive malformed input and fail. This creates a cascading failure: next_line_match output format change → find_line input expectation mismatch → line boundary calculation failure. Consequently, both functions must be modified together to maintain the search pipeline's data contract.",
      "selected_functions": ["src/search.rs::Iter<'b, 's>::function::next_line_match", "src/search.rs::Iter<'b, 's>::function::find_line"],
      "reasoning_update": {
        "src/search.rs::Iter<'b, 's>::function::next_line_match": "MODIFY: Step 1: Locate all regex.find() calls in the function. Step 2: Replace each regex.find() with regex.shortest_match() which returns only the end position instead of (start, end) tuple. Step 3: Update variable declarations from 'let (s, e) = match ...' to 'let e = match ...' since shortest_match returns only end position. Step 4: Update the offset calculation to use only end position: change '(self.start + s, self.start + e)' to 'self.start + e'.",
        "src/search.rs::Iter<'b, 's>::function::find_line": "MODIFY: Modify function signature to accept single position parameter instead of range, and update internal boundary logic accordingly to maintain functionality."
      }
    }
  ]
}
<END>
"""

def load_ast_elements(edit_sequence_file: str) -> List[Dict[str, Any]]:
    """Load AST elements from edit sequence dataset file."""
    try:
        with open(edit_sequence_file, 'r') as f:
            data = json.load(f)
        
        if 'edit_sequences' in data:
            return data['edit_sequences']
        else:
            return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading AST elements from {edit_sequence_file}: {e}")
        return []


def load_commit_metadata(commit_data_file: str) -> Dict[str, Any]:
    """Load commit metadata excluding diff."""
    try:
        with open(commit_data_file, 'r') as f:
            data = json.load(f)
        
        # Remove diff to avoid including implementation details
        metadata = {
            'commit_hash': data.get('commit_hash', ''),
            'commit_name': data.get('commit_name', ''),
            'metadata': data.get('metadata', {}),
            'files_modified': [
                {
                    'file_path': f.get('file_path', ''),
                    # Don't include content, just the file path
                } 
                for f in data.get('files_modified', [])
            ]
        }
        
        # Include PR metadata if available (rich context for task generation)
        if 'pr_metadata' in data:
            metadata['pr_metadata'] = data['pr_metadata']
        if 'rich_pr_data' in data:
            metadata['rich_pr_data'] = {
                'pr_title': data['rich_pr_data'].get('pr_title', ''),
                'pr_description': data['rich_pr_data'].get('pr_description', ''),
                'pr_author': data['rich_pr_data'].get('pr_author', ''),
                'pr_labels': data['rich_pr_data'].get('pr_labels', []),
                'pr_reviewers': data['rich_pr_data'].get('pr_reviewers', []),
                'source_branch': data['rich_pr_data'].get('source_branch', ''),
                'target_branch': data['rich_pr_data'].get('target_branch', ''),
                # Don't include full reviews/comments to avoid implementation details
            }
        
        return metadata
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading commit metadata from {commit_data_file}: {e}")
        return {}


def are_elements_related(elem1: Dict, elem2: Dict) -> bool:
    """Check if two AST elements are related/coupled and should be in different batches."""
    id1 = elem1.get('ast_id', '')
    id2 = elem2.get('ast_id', '')
    
    # Same file and similar names = likely related
    if '::' in id1 and '::' in id2:
        file1 = id1.split('::')[0]
        file2 = id2.split('::')[0] 
        if file1 == file2:
            # Same file - check if they're closely related functions
            func1 = id1.split('::')[-1]
            func2 = id2.split('::')[-1]
            # If one calls the other or they have similar names, they're related
            if func1 in func2 or func2 in func1:
                return True
    
    # Check if one element references the other in retrieved_labels
    labels1 = set(elem1.get('retrieved_labels', []))
    labels2 = set(elem2.get('retrieved_labels', []))
    
    if id1 in labels2 or id2 in labels1:
        return True
    
    return False


# Removed create_unrelated_batches function - replaced with group_related_diff_sequences


def create_batch_prompt(ast_batch: List[Dict], commit_metadata: Dict) -> str:
    """Create a focused prompt for a batch of AST elements."""
    
    # Extract key information from commit metadata
    commit_msg = commit_metadata.get('metadata', {}).get('commit_message', '')
    author = commit_metadata.get('metadata', {}).get('author_name', '')
    commit_hash = commit_metadata.get('commit_hash', '')[:8]
    files_modified = [f.get('file_path', '') for f in commit_metadata.get('files_modified', [])]
    
    # PR context if available
    pr_context = ""
    if 'rich_pr_data' in commit_metadata:
        pr_data = commit_metadata['rich_pr_data']
        pr_context = f"""
PR Context:
- Title: {pr_data.get('pr_title', 'N/A')}
- Description: {pr_data.get('pr_description', 'N/A')[:300]}...
- Labels: {', '.join(pr_data.get('pr_labels', []))}
"""
    
    # Create AST elements summary
    ast_summary = []
    for i, elem in enumerate(ast_batch):
        ast_id = elem.get('ast_id', f'element_{i}')
        original_func = elem.get('original_function', '')[:150]  # Shorter preview
        edit_count = len(elem.get('edit_sequence', []))
        operation_type = elem.get('metadata', {}).get('operation_type', 'UNKNOWN')
        
        ast_summary.append(f"""
AST Element {i+1}:
- ID: {ast_id}
- Operation: {operation_type}
- Code Preview: {original_func}...
- Edit Operations: {edit_count}
""")
    
    prompt = f"""
Commit Information:
- Hash: {commit_hash}  
- Author: {author}
- Message: {commit_msg[:200]}...
- Files: {', '.join(files_modified[:3])}

{pr_context}

AST Elements Batch ({len(ast_batch)} elements):
{''.join(ast_summary)}

CRITICAL: You are creating LLM TRAINING DATA. Analyze these AST elements and create appropriate tasks.

LLM TRAINING REQUIREMENTS:
1. Generate balanced tasks - group related changes together, separate unrelated changes
2. reasoning_selection must demonstrate HOW to analyze the codebase and identify which functions need modification
3. reasoning_update must provide step-by-step implementation guidance for each selected function
4. Include detailed WHY explanations for causal reasoning training
5. Show the complete thought process from problem analysis to solution implementation
6. Make reasoning explicit and educational, not just descriptive
7. Each task should be implementable independently (no external dependencies)
8. Balance granularity: not too small (breaking logical units) and not too big (unrelated features)
9. Handle all operation types: MODIFY, DELETE, ADD

Remember: This training data will teach future LLMs HOW to think through code changes systematically. Include all the reasoning steps that a model needs to learn.
"""
    
    return prompt


async def call_llm_for_enhanced_tasks(prompt: str, system_prompt: str) -> str:
    """Call LLM API for enhanced task generation."""
    if not HAS_OPENAI:
        return json.dumps({"tasks": [{"task": "Task generation requires OpenAI library", "selected_functions": [], "reasoning_selection": "N/A", "reasoning_update": "N/A"}]})

    global LLM_SEMAPHORE
    if LLM_SEMAPHORE is None:
        initialize_semaphore()

    async with LLM_SEMAPHORE:
        try:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            )

            response = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME", "gpt-4"),  # Use GPT-4 for better reasoning
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=20000,   # Allow longer responses for detailed reasoning
                stop=["<END>"]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"  ✗ Error calling LLM API: {e}")
            return json.dumps({"tasks": []})


def validate_task_format(task: Dict) -> bool:
    """Validate that a task has the required structure."""
    required_fields = ['task', 'reasoning_selection', 'selected_functions', 'reasoning_update']
    
    for field in required_fields:
        if field not in task:
            return False
        
        if field == 'selected_functions':
            if not isinstance(task[field], list) or len(task[field]) == 0:
                return False
        elif field == 'reasoning_update':
            if not isinstance(task[field], dict):
                return False
            # Validate that reasoning_update has entries for all selected functions
            selected_functions = task.get('selected_functions', [])
            for function in selected_functions:
                if function not in task[field]:
                    return False
                if not isinstance(task[field][function], str) or not task[field][function].strip():
                    return False
        else:
            if not isinstance(task[field], str) or not task[field].strip():
                return False
    
    return True


def group_related_diff_sequences(ast_elements: List[Dict], config: Dict[str, Any] = None) -> List[List[Dict]]:
    """Group related diff sequences that should be implemented together as one logical task."""
    if not ast_elements:
        return []
    
    if config is None:
        config = load_config()
    
    max_group_size = config.get('max_related_group_size', 4)
    
    # Create groups based on related functionality
    groups = []
    used_indices = set()
    
    for i, elem in enumerate(ast_elements):
        if i in used_indices:
            continue
            
        current_group = [elem]
        used_indices.add(i)
        elem_id = elem.get('ast_id', '')
        elem_labels = set(elem.get('retrieved_labels', []))
        
        # Look for related elements
        for j, other_elem in enumerate(ast_elements[i+1:], i+1):
            if j in used_indices:
                continue
                
            other_id = other_elem.get('ast_id', '')
            other_labels = set(other_elem.get('retrieved_labels', []))
            
            # Check if elements are related (should be in same task)
            is_related = False
            
            # Same file and functionally related
            if '::' in elem_id and '::' in other_id:
                elem_file = elem_id.split('::')[0]
                other_file = other_id.split('::')[0]
                if elem_file == other_file:
                    # Check if they reference each other
                    if elem_id in other_labels or other_id in elem_labels:
                        is_related = True
                    # Check if they're part of same logical component
                    elif any(label in other_labels for label in elem_labels):
                        is_related = True
            
            if is_related and len(current_group) < max_group_size:  # Use configurable group size
                current_group.append(other_elem)
                used_indices.add(j)
        
        groups.append(current_group)
    
    return groups


def create_ast_element_batches(ast_elements: List[Dict], config: Dict[str, Any] = None) -> List[List[Dict]]:
    """Create batches of AST elements based on configuration limits."""
    if not ast_elements:
        return []
    
    if config is None:
        config = load_config()
    
    max_batch_size = config.get('max_ast_elements_per_batch', 5)
    
    # If we have fewer elements than the batch size, return as single batch
    if len(ast_elements) <= max_batch_size:
        return [ast_elements]
    
    # Split into batches
    batches = []
    for i in range(0, len(ast_elements), max_batch_size):
        batch = ast_elements[i:i + max_batch_size]
        batches.append(batch)
    
    return batches


async def process_ast_batch(ast_batch: List[Dict], commit_metadata: Dict) -> List[Dict]:
    """Process a batch of AST elements and generate tasks."""
    
    if not ast_batch:
        return []
    
    # Create prompt for the batch of AST elements
    prompt = create_batch_prompt(ast_batch, commit_metadata)
    
    # Call LLM
    response_text = await call_llm_for_enhanced_tasks(prompt, ENHANCED_SYSTEM_PROMPT)
    
    try:
        # Extract JSON from response
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_content = match.group(1)
        else:
            # Fallback: try to parse entire response as JSON
            json_content = response_text
        
        response = json.loads(json_content)
        
        if isinstance(response, dict) and "tasks" in response:
            tasks = response["tasks"]
            
            # Validate each task
            validated_tasks = []
            for task in tasks:
                if validate_task_format(task):
                    # Validate that selected functions are reasonable
                    selected = task.get('selected_functions', [])
                    if 1 <= len(selected) <= len(ast_batch):  # Should have at least 1, max all elements
                        validated_tasks.append(task)
                    else:
                        print(f"    ⚠️  Warning: Task has unreasonable function count ({len(selected)}): {task['task'][:50]}...")
                else:
                    print(f"    ⚠️  Warning: Invalid task format: {task}")
            
            return validated_tasks
        else:
            print("    ⚠️  Warning: Invalid response format, missing 'tasks' field")
            return []
            
    except json.JSONDecodeError as e:
        print(f"    ⚠️  Warning: Failed to parse enhanced tasks: {e}")
        return []


async def generate_enhanced_tasks_for_commit(commit_dir: str) -> bool:
    """Generate balanced, cohesive tasks for a single commit directory."""
    
    edit_sequence_file = os.path.join(commit_dir, 'edit_sequence_dataset.json')
    commit_data_file = os.path.join(commit_dir, 'commit_data.json')
    output_file = os.path.join(commit_dir, 'enhanced_tasks.json')
    
    # Skip if output already exists
    if os.path.exists(output_file):
        return True
    
    # Check if required input files exist
    if not os.path.exists(edit_sequence_file):
        print(f"    ⚠️  No edit sequence file found: {edit_sequence_file}")
        return False
    
    if not os.path.exists(commit_data_file):
        print(f"    ⚠️  No commit data file found: {commit_data_file}")
        return False
    
    # Load configuration
    config = load_config()
    
    # Load data
    ast_elements = load_ast_elements(edit_sequence_file)
    commit_metadata = load_commit_metadata(commit_data_file)
    
    if not ast_elements:
        print(f"    ⚠️  No AST elements found in {edit_sequence_file}")
        return False
    
    print(f"    Processing {len(ast_elements)} AST elements")
    print(f"    Batch size limit: {config.get('max_ast_elements_per_batch', 5)} elements")
    
    # Create batches based on configuration
    ast_batches = create_ast_element_batches(ast_elements, config)
    print(f"    Split into {len(ast_batches)} batches")
    
    all_tasks = []
    
    # Process each batch
    for batch_num, batch in enumerate(ast_batches, 1):
        print(f"    Processing batch {batch_num}/{len(ast_batches)} ({len(batch)} elements)")
        
        try:
            batch_tasks = await process_ast_batch(batch, commit_metadata)
            
            if batch_tasks:
                all_tasks.extend(batch_tasks)
                print(f"    ✓ Generated {len(batch_tasks)} tasks from batch {batch_num}")
            else:
                print(f"    ⚠️  No valid tasks generated from batch {batch_num}")
                
        except Exception as e:
            print(f"    ✗ Error processing batch {batch_num}: {e}")
            continue
    
    if all_tasks:
        # Create output data
        output_data = {
            'commit_hash': commit_metadata.get('commit_hash', ''),
            'commit_name': commit_metadata.get('commit_name', ''),
            'total_ast_elements': len(ast_elements),
            'total_batches': len(ast_batches),
            'tasks': all_tasks,
            'task_count': len(all_tasks),
            'generation_metadata': {
                'batching_strategy': 'configurable_batch_processing',
                'max_ast_elements_per_batch': config.get('max_ast_elements_per_batch', 5),
                'total_batches_processed': len(ast_batches),
                'ast_elements_processed': len(ast_elements),
                'batch_task_validation': True,
                'batch_sizes': [len(batch) for batch in ast_batches]
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        commit_name = commit_metadata.get('commit_name', 'unknown')
        print(f"    ✓ Generated {len(all_tasks)} balanced tasks for {commit_name}")
        return True
    else:
        print(f"    ⚠️  No valid tasks generated for commit")
        return False


async def process_repository_enhanced_tasks(repo_name: str, base_dir: str = 'data/ast_dataset'):
    """Process all commits in a repository to generate balanced tasks."""
    
    print(f"\n{'='*60}")
    print(f"Processing Enhanced Tasks for Repository: {repo_name}")
    print(f"{'='*60}")
    
    # Load configuration
    config = load_config()
    max_concurrent_commits = config.get('max_concurrent_commits', 2)
    
    # Initialize semaphore for this repository processing
    initialize_semaphore(config)
    
    repo_path = os.path.join(base_dir, repo_name)
    
    if not os.path.isdir(repo_path):
        print(f"Skipping {repo_name}: Repository directory not found")
        return
    
    # Find commit directories
    commit_pattern = re.compile(r'^commit_(\d+(?:\.\d+)?)$')  # Support commit_X.Y format
    commit_dirs = []
    
    for d in os.listdir(repo_path):
        full_path = os.path.join(repo_path, d)
        if os.path.isdir(full_path) and commit_pattern.match(d):
            commit_dirs.append(d)
    
    # Sort commits properly (handle both commit_X and commit_X.Y)
    def sort_key(commit_dir):
        match = commit_pattern.match(commit_dir)
        if match:
            commit_num = match.group(1)
            if '.' in commit_num:
                main, sub = commit_num.split('.')
                return (int(main), int(sub))
            else:
                return (int(commit_num), 0)
        return (0, 0)
    
    commit_dirs.sort(key=sort_key)
    
    if not commit_dirs:
        print(f"No commit directories found in {repo_path}")
        return
    
    print(f"Found {len(commit_dirs)} commits to process")
    print(f"Max concurrent commits: {max_concurrent_commits}")
    print(f"Max concurrent LLM calls: {config.get('max_concurrent_llm_calls', 5)}")
    
    # Create semaphore for commit processing
    commit_semaphore = asyncio.Semaphore(max_concurrent_commits)
    
    async def process_single_commit(commit_dir: str) -> bool:
        """Process a single commit with semaphore control."""
        async with commit_semaphore:
            commit_path = os.path.join(repo_path, commit_dir)
            print(f"\n  Processing {commit_dir}:")
            
            try:
                success = await generate_enhanced_tasks_for_commit(commit_path)
                if success:
                    print(f"  ✓ Completed {commit_dir}")
                return success
            except Exception as e:
                print(f"    ✗ Error processing {commit_dir}: {e}")
                return False
    
    # Process commits in parallel with concurrency control
    tasks = [process_single_commit(commit_dir) for commit_dir in commit_dirs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful commits
    successful_commits = sum(1 for result in results if result is True)
    
    print(f"\nCompleted {repo_name}: {successful_commits}/{len(commit_dirs)} commits processed successfully")


def find_repo_dirs(base_dir: str) -> List[str]:
    """Find repository directories within the base dataset directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


async def run_enhanced_task_pipeline(repos: List[str] = None, base_dir: str = 'data/ast_dataset'):
    """Run the balanced task generation pipeline."""
    
    if not HAS_OPENAI:
        print("\nError: OpenAI library not installed.")
        print("Install with: pip install openai python-dotenv")
        return
    
    if repos is None:
        repo_names = find_repo_dirs(base_dir)
        if not repo_names:
            print(f"No repository data found in {base_dir}.")
            return
    else:
        repo_names = repos
    
    print(f"\n{'='*60}")
    print(f"Starting Enhanced Task Generation for {len(repo_names)} repositories")
    print(f"{'='*60}")
    
    for repo_name in repo_names:
        await process_repository_enhanced_tasks(repo_name, base_dir)
    
    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for enhanced task generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced tasks from AST elements')
    parser.add_argument('repos', nargs='*', help='Repository names to process (default: all)')
    parser.add_argument('--base-dir', default='data/ast_dataset',
                       help='Base directory containing AST dataset (default: data/ast_dataset)')
    
    args = parser.parse_args()
    
    repos = args.repos if args.repos else None
    asyncio.run(run_enhanced_task_pipeline(repos, args.base_dir))


if __name__ == "__main__":
    cli()
