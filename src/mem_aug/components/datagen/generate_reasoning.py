"""
Generates reasoning for code changes using LLM analysis.
This step runs after generate_tasks.py and build_fim_dataset.py.
"""
import os
import json
import re
import asyncio
from typing import List, Dict, Any
import sys

try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Global semaphore to limit concurrent LLM calls
LLM_SEMAPHORE = asyncio.Semaphore(50)

ADD_REASONING_PROMPT = """
You are a developer who has been given a task. Write your step-by-step thought process as you plan to implement a new function.

Write in FIRST PERSON from the developer's perspective, showing your technical reasoning and planning.

IMPORTANT: You will be given a list of tasks from a commit. ONLY focus on the task(s) that are relevant to THIS specific function. Ignore tasks that relate to other functions or files.

Guidelines:
- Write as "I will...", "First I need to...", "This requires..."
- Show your step-by-step thinking process
- Explain technical decisions and trade-offs
- Reference specific types, patterns, APIs you'll use
- Connect the task requirements to implementation choices
- Be detailed and technical (mention data structures, control flow, edge cases)
- Scale detail based on complexity:
  - Simple functions (< 20 lines): 3-5 steps
  - Medium functions (20-50 lines): 6-10 steps
  - Complex functions (> 50 lines): 10+ steps with subsections

Example format:
"Looking at the task, I need to [task goal]. First, I'll define the function signature with [types/params]. Then I'll initialize [data structures]. For the main logic, I'll use [pattern/approach] because [reason]. I need to handle [edge case] by [solution]. Finally, I'll return [what] to satisfy [requirement]."

Output JSON only:
```json
{"reasoning": "Your step-by-step thought process here"}
```
"""

UPDATE_REASONING_PROMPT = """
You are a developer who has been given a task to modify existing code. Write your step-by-step thought process as you plan the changes.

Write in FIRST PERSON from the developer's perspective, showing your technical reasoning.

IMPORTANT: You will be given a list of tasks from a commit. ONLY focus on the task(s) that are relevant to THIS specific function change. Ignore tasks that relate to other functions or files.

Guidelines:
- Write as "I will...", "I need to change...", "Looking at the existing code..."
- Show your step-by-step thinking process
- Reference what exists and what needs to change
- Explain WHY each change is needed
- Mention specific APIs, types, patterns you'll modify/add
- Discuss trade-offs and technical decisions
- Be detailed and technical
- Scale detail based on change size:
  - Small changes (< 5 lines): 2-4 steps
  - Medium changes (5-20 lines): 5-8 steps
  - Large changes (> 20 lines): 8+ steps with detailed reasoning

Example format:
"Looking at the existing code, I see it currently [what it does]. To implement [task], I need to change [specific part] from [old approach] to [new approach] because [reason]. First, I'll modify [component] to use [new type/API]. Then I'll update [logic] to handle [new requirement]. I also need to adjust [another part] to ensure [goal]. This approach is better than [alternative] because [technical reason]."

Output JSON only:
```json
{"reasoning": "Your step-by-step thought process here"}
```
"""

DELETE_REASONING_PROMPT = """
You are a developer who has been given a task that involves removing code. Write your step-by-step thought process as you plan the deletion.

Write in FIRST PERSON from the developer's perspective, showing your technical reasoning.

IMPORTANT: You will be given a list of tasks from a commit. ONLY focus on the task(s) that are relevant to THIS specific function deletion. Ignore tasks that relate to other functions or files.

Guidelines:
- Write as "I will remove...", "This is no longer needed because...", "I can delete..."
- Show your step-by-step thinking process
- Explain WHY the code can/should be removed
- Mention what replaces it or why it's obsolete
- Reference refactoring goals or architectural changes
- Be technical about dependencies and impacts
- Scale detail based on complexity:
  - Simple functions: 2-3 steps
  - Complex functions: 4-6 steps

Example format:
"Looking at the task, I need to remove [function] because [reason]. This function was previously used for [old purpose], but now we're using [new approach] instead. I can safely delete this since [why it's safe]. This removal will simplify [what] and allow us to [benefit]."

Output JSON only:
```json
{"reasoning": "Your step-by-step thought process here"}
```
"""


async def call_llm_for_reasoning(prompt: str, context: str) -> str:
    """
    Call LLM API for reasoning generation.
    """
    if not HAS_OPENAI:
        return json.dumps({"reasoning": "Reasoning generation requires OpenAI library"})

    async with LLM_SEMAPHORE:
        try:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            )

            response = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"    ✗ Error calling LLM API: {e}")
            return json.dumps({"reasoning": ""})


def count_lines(code: str) -> int:
    """Count non-empty lines in code."""
    return len([line for line in code.split('\n') if line.strip()])


async def generate_reasoning_for_function(
    function_data: Dict[str, Any],
    tasks: List[str]
) -> Dict[str, Any]:
    """
    Generate reasoning for a single function change from diff data.
    """
    # Extract function information from diff data
    function_id = function_data.get('id', '')
    file_name = function_data.get('file', '')
    status = function_data.get('status', 'modified')
    before_code = function_data.get('before_code', '')
    after_code = function_data.get('after_code', '')
    kind = function_data.get('kind', '')
    commit_hash = function_data.get('commit_metadata', {}).get('commit_hash', '')

    # Map status to operation type
    status_to_op = {
        'added': 'ADD',
        'deleted': 'DELETE',
        'modified': 'UPDATE'
    }
    operation_type = status_to_op.get(status, 'UPDATE')

    # Build context string for LLM
    tasks_str = '\n'.join([f"{i+1}. {task}" for i, task in enumerate(tasks)])

    if operation_type == 'ADD':
        # For ADD: show task + new code
        line_count = count_lines(after_code)

        context = f"""
You are analyzing THIS specific function:
  File: {file_name}
  Function: {function_id}

Commit task list (ONLY use tasks relevant to this specific function):
{tasks_str}

New function being added ({line_count} lines):
```
{after_code}
```
"""
        prompt = ADD_REASONING_PROMPT

    elif operation_type == 'DELETE':
        # For DELETE: show task + deleted code
        line_count = count_lines(before_code)

        context = f"""
You are analyzing THIS specific function:
  File: {file_name}
  Function: {function_id}

Commit task list (ONLY use tasks relevant to this specific function deletion):
{tasks_str}

Function being removed ({line_count} lines):
```
{before_code}
```
"""
        prompt = DELETE_REASONING_PROMPT

    else:  # UPDATE
        # For UPDATE: show task + before + after
        before_lines = count_lines(before_code)
        after_lines = count_lines(after_code)

        context = f"""
You are analyzing THIS specific function:
  File: {file_name}
  Function: {function_id}

Commit task list (ONLY use tasks relevant to this specific function change):
{tasks_str}

Function modification ({before_lines} → {after_lines} lines):

BEFORE:
```
{before_code}
```

AFTER:
```
{after_code}
```
"""
        prompt = UPDATE_REASONING_PROMPT

    # Call LLM
    response_text = await call_llm_for_reasoning(prompt, context)

    try:
        # Extract JSON from code block
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            json_content = response_text

        result = json.loads(json_content)
        reasoning = result.get('reasoning', '')

        return {
            'function_id': function_id,
            'file_name': file_name,
            'operation_type': operation_type,
            'reasoning': reasoning,
            'metadata': {
                'commit_hash': commit_hash,
                'kind': kind
            }
        }
    except json.JSONDecodeError:
        print(f"    ⚠️  Warning: Failed to parse reasoning for {function_id}")
        return {
            'function_id': function_id,
            'file_name': file_name,
            'operation_type': operation_type,
            'reasoning': '',
            'metadata': {
                'commit_hash': commit_hash,
                'kind': kind
            }
        }


async def generate_reasoning_for_commit(commit_dir: str, max_retries: int = 3) -> None:
    """
    Generate reasoning for all function changes in a commit.

    Args:
        commit_dir: Path to commit directory
        max_retries: Maximum number of retry attempts on failure
    """
    diff_ast_path = os.path.join(commit_dir, 'diff_ast.jsonl')
    task_path = os.path.join(commit_dir, 'task.json')
    reasoning_file = os.path.join(commit_dir, 'reasoning.json')

    # Skip if reasoning.json already exists
    if os.path.exists(reasoning_file):
        return

    # Both files must exist
    if not os.path.exists(diff_ast_path) or not os.path.exists(task_path):
        return

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Load diff AST data
            diff_functions = []
            with open(diff_ast_path, 'r') as f:
                for line in f:
                    if line.strip():
                        diff_functions.append(json.loads(line))

            # Load tasks
            with open(task_path, 'r') as f:
                task_data = json.load(f)

            tasks = task_data.get('tasks', [])
            if not tasks:
                print(f"    ⚠️  No tasks found for commit")
                return

            if not diff_functions:
                print(f"    ⚠️  No functions found in diff_ast.jsonl")
                return

            # Generate reasoning for each function
            all_reasoning = []

            for function_data in diff_functions:
                # Only process functions with code changes
                if function_data.get('code_changed', False):
                    reasoning_data = await generate_reasoning_for_function(function_data, tasks)
                    all_reasoning.append(reasoning_data)

            if all_reasoning:
                # Extract commit info from task data
                commit_number = task_data.get('commit_number')
                commit_hash = task_data.get('commit_hash', '')

                # Fallback: if not in task data, try to extract from directory name
                if commit_number is None:
                    commit_dir_name = os.path.basename(commit_dir)
                    commit_pattern = re.compile(r'^commit_(\d+)$')
                    match = commit_pattern.match(commit_dir_name)
                    if match:
                        commit_number = int(match.group(1))

                if not commit_hash:
                    commit_data_file = os.path.join(commit_dir, 'commit_data.json')
                    if os.path.exists(commit_data_file):
                        try:
                            with open(commit_data_file, 'r') as f:
                                commit_data = json.load(f)
                                commit_hash = commit_data.get('commit_hash', '')
                        except:
                            pass

                # Extract repository name from path
                repo_name = os.path.basename(os.path.dirname(commit_dir))

                # Save reasoning
                output_data = {
                    'commit_number': commit_number,
                    'commit_hash': commit_hash,
                    'repository': repo_name,
                    'reasoning_count': len(all_reasoning),
                    'function_reasoning': all_reasoning
                }

                output_file = os.path.join(commit_dir, 'reasoning.json')
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)

                commit_hash_short = commit_hash[:8] if commit_hash else 'unknown'
                print(f"    ✓ Generated reasoning for {len(all_reasoning)} functions in commit_{commit_number} ({commit_hash_short})")
                return  # Success, exit retry loop

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"    ⚠️  Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  ✗ Error processing {commit_dir} after {max_retries} attempts: {e}")


async def process_repository_reasoning(repo_name: str, base_dir: str = 'data/ast_dataset'):
    """
    Process all commits in a repository to generate reasoning.

    Args:
        repo_name: Name of the repository to process
        base_dir: Base directory containing AST dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing repository: {repo_name}")
    print(f"{'='*60}")

    repo_path = os.path.join(base_dir, repo_name)

    if not os.path.isdir(repo_path):
        print(f"Skipping {repo_name}: Repository directory not found")
        return

    # Find all commit directories
    commit_pattern = re.compile(r'^commit_(\d+)$')
    commit_dirs = []
    for d in os.listdir(repo_path):
        if os.path.isdir(os.path.join(repo_path, d)) and commit_pattern.match(d):
            commit_dirs.append(d)

    # Sort based on the numeric part of the directory name
    commit_dirs.sort(key=lambda x: int(commit_pattern.match(x).group(1)))

    if not commit_dirs:
        print(f"Skipping {repo_name}: No commit directories found")
        return

    print(f"  Found {len(commit_dirs)} commits to process")

    reasoning_generated = 0

    # Process all commits in parallel
    commit_paths = [os.path.join(repo_path, commit_name) for commit_name in commit_dirs]
    await asyncio.gather(*[generate_reasoning_for_commit(commit_path) for commit_path in commit_paths])

    # Count how many reasoning files were created
    for commit_dir in commit_paths:
        reasoning_file = os.path.join(commit_dir, 'reasoning.json')
        if os.path.exists(reasoning_file):
            reasoning_generated += 1

    print(f"\nCompleted {repo_name}: {reasoning_generated} commits with reasoning generated")


def find_repo_dirs(base_dir: str) -> List[str]:
    """Finds repository directories within the base dataset directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


async def run_reasoning_pipeline(repos: List[str] = None, base_dir: str = 'data/ast_dataset'):
    """
    Runs the reasoning generation pipeline for specified repositories or all repositories.

    Args:
        repos: List of repository names to process. If None, processes all repos.
        base_dir: Base directory containing AST dataset
    """
    if not HAS_OPENAI:
        print("\nError: OpenAI library not installed.")
        print("Install with: uv sync --extra tasks")
        return

    if repos is None:
        repo_names = find_repo_dirs(base_dir)
        if not repo_names:
            print(f"No repository data found in {base_dir}.")
            print("Please run the full pipeline first.")
            return
    else:
        repo_names = repos

    print(f"\n{'='*60}")
    print(f"Starting reasoning generation for {len(repo_names)} repositories")
    print(f"{'='*60}")

    for repo_name in repo_names:
        await process_repository_reasoning(repo_name, base_dir)

    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for reasoning generation."""
    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        asyncio.run(run_reasoning_pipeline(repos))
    else:
        # Process all repos
        asyncio.run(run_reasoning_pipeline())


if __name__ == "__main__":
    cli()
