"""
Generates task descriptions from commit diffs using LLM.
This is an optional step that can run after generate_commit_ast.py.
"""
import os
import json
import re
import asyncio
from typing import List, Dict, Any
import sys

# Try to import OpenAI, but make it optional
try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai and python-dotenv not installed. Install with: pip install openai python-dotenv")

# Global semaphore to limit concurrent LLM calls
LLM_SEMAPHORE = asyncio.Semaphore(5)  # Allow max 5 concurrent LLM calls

SYSTEM_PROMPT = """
You are an AI system that analyzes commit diffs and generates detailed task descriptions for developers.

Your goal is to create task descriptions that are detailed enough for a developer to implement the next commit with help from the previous commit, without revealing the exact implementation details from the diff.

Guidelines:
- Output **1 to N tasks** depending on the size of the commit
- Each task should be **detailed and actionable** but not reveal exact code changes
- Include **what needs to be done** and **where it should be done**
- Mention relevant directories, file types, or system components
- Include the **purpose/goal** of the change
- Tasks must provide enough context that a developer can:
  1. Understand what feature/fix to implement
  2. Know which parts of the codebase to modify
  3. Understand the expected outcome
- Do NOT include:
  - Exact code snippets from the diff
  - Specific line numbers or exact file paths
  - Variable names or function signatures
- Use **action verbs** and **specific technical terms**
- Focus on **architectural/structural changes** and **business logic**

Output format:
```json
{"tasks": ["detailed task description 1", "detailed task description 2", "..."]}
```

Examples:
* {"tasks": ["Integrate new syntax highlighting module for Todo.txt format by adding external syntax definition repository as submodule in the assets/syntaxes directory"]}
* {"tasks": ["Implement user authentication system with login/logout functionality in the auth module", "Add session management with token-based validation in middleware layer"]}
* {"tasks": ["Refactor memory management in heap statistics collection to use direct API calls instead of mutable temporary objects for better performance"]}
"""


def chunk_diff_by_file(diff_text: str, max_chunk_size: int = 20000) -> List[str]:
    """
    Split a large diff into manageable chunks by file.
    Each chunk contains one or more file changes.
    """
    # Split diff by file headers
    diff_pattern = r'diff --git a/(.*?) b/(.*?)\n'
    matches = list(re.finditer(diff_pattern, diff_text))

    if not matches:
        return [diff_text]

    chunks = []
    for i, match in enumerate(matches):
        next_pos = match.start()
        if i + 1 < len(matches):
            next_pos = matches[i + 1].start()
        else:
            next_pos = len(diff_text)

        file_diff = diff_text[match.start():next_pos]

        # If this file diff is too large, split it further by hunks
        if len(file_diff) > max_chunk_size:
            hunk_chunks = chunk_diff_by_hunk(file_diff, max_chunk_size)
            chunks.extend(hunk_chunks)
        else:
            chunks.append(file_diff)

    return chunks


def chunk_diff_by_hunk(diff_text: str, max_chunk_size: int = 20000) -> List[str]:
    """
    Split a file diff into chunks by hunks if it's too large.
    """
    # Find hunk headers (@@ -x,y +a,b @@)
    hunk_pattern = r'@@.*?@@\n'
    matches = list(re.finditer(hunk_pattern, diff_text))

    if not matches:
        return [diff_text]

    chunks = []
    current_chunk = ""

    for i, match in enumerate(matches):
        hunk_start = match.start()

        # Find where this hunk ends (start of next hunk or end of file)
        next_hunk_start = len(diff_text)
        if i + 1 < len(matches):
            next_hunk_start = matches[i + 1].start()

        hunk_content = diff_text[hunk_start:next_hunk_start]

        # If adding this hunk would exceed chunk size, start a new chunk
        if len(current_chunk + hunk_content) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = hunk_content
        else:
            current_chunk += hunk_content

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def call_llm_for_tasks(diff_chunk: str, system_prompt: str) -> str:
    """
    Call LLM API for task generation using async OpenAI client with semaphore control.
    """
    if not HAS_OPENAI:
        return json.dumps({"tasks": ["Task generation requires OpenAI library"]})

    async with LLM_SEMAPHORE:  # Acquire semaphore before making LLM call
        try:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            )

            response = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze this commit diff and generate structured tasks:\n\n{diff_chunk}"}
                ],
                temperature=0.2
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"  ✗ Error calling LLM API: {e}")
            return json.dumps({"tasks": []})


async def process_commit_diff(diff_text: str, system_prompt: str) -> List[str]:
    """
    Process a commit diff and generate tasks, chunking if necessary.
    """
    # Check if diff is too large and needs chunking
    if len(diff_text) > 20000:
        chunks = chunk_diff_by_file(diff_text)
        print(f"    Diff chunked into {len(chunks)} parts")

        # Process chunks in parallel with semaphore controlling concurrency
        async def process_chunk(i, chunk):
            response_text = await call_llm_for_tasks(chunk, system_prompt)

            try:
                # Extract JSON from code block using regex
                json_pattern = r'```json\s*(\{.*?\})\s*```'
                match = re.search(json_pattern, response_text, re.DOTALL)
                if match:
                    json_content = match.group(1)
                else:
                    # Fallback: try to parse the entire response as JSON
                    json_content = response_text

                chunk_response = json.loads(json_content)
                if isinstance(chunk_response, dict) and "tasks" in chunk_response:
                    chunk_tasks = chunk_response["tasks"]
                    if isinstance(chunk_tasks, list):
                        return chunk_tasks
                    else:
                        return []
                else:
                    return []
            except json.JSONDecodeError:
                print(f"    ⚠️  Warning: Failed to parse tasks from chunk {i+1}")
                return []

        # Process all chunks in parallel
        chunk_tasks_lists = await asyncio.gather(*[
            process_chunk(i, chunk) for i, chunk in enumerate(chunks)
        ])

        # Flatten and deduplicate tasks
        all_tasks = []
        for chunk_tasks in chunk_tasks_lists:
            for task in chunk_tasks:
                if task not in all_tasks:
                    all_tasks.append(task)

        return all_tasks
    else:
        # Process small diff directly
        response_text = await call_llm_for_tasks(diff_text, system_prompt)
        try:
            # Extract JSON from code block using regex
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                json_content = match.group(1)
            else:
                # Fallback: try to parse the entire response as JSON
                json_content = response_text

            response = json.loads(json_content)
            if isinstance(response, dict) and "tasks" in response:
                return response["tasks"]
            else:
                print("    ⚠️  Warning: Invalid response format, missing 'tasks' field")
                return []
        except json.JSONDecodeError:
            print("    ⚠️  Warning: Failed to parse tasks from diff")
            return []


async def generate_tasks_for_commit(commit_dir: str, max_retries: int = 3) -> None:
    """
    Generate tasks for a single commit and store in the same commit directory.

    Args:
        commit_dir: Path to commit directory
        max_retries: Maximum number of retry attempts on failure
    """
    commit_data_path = os.path.join(commit_dir, 'commit_data.json')
    task_file = os.path.join(commit_dir, 'task.json')

    # Skip if task.json already exists
    if os.path.exists(task_file):
        return

    if not os.path.exists(commit_data_path):
        return

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            with open(commit_data_path, 'r') as f:
                commit_data = json.load(f)

            diff_text = commit_data.get('diff', '')
            if not diff_text:
                print(f"    ⚠️  No diff found in {commit_data_path}")
                return

            tasks = await process_commit_diff(diff_text, SYSTEM_PROMPT)

            if tasks:
                # Extract commit number from directory name (e.g., commit_10 -> 10)
                commit_dir_name = os.path.basename(commit_dir)
                commit_number = None
                commit_pattern = re.compile(r'^commit_(\d+)$')
                match = commit_pattern.match(commit_dir_name)
                if match:
                    commit_number = int(match.group(1))

                output_data = {
                    'commit_number': commit_number,
                    'commit_hash': commit_data.get('commit_hash', ''),
                    'commit_metadata': commit_data.get('metadata', {}),
                    'tasks': tasks,
                    'task_count': len(tasks)
                }

                with open(task_file, 'w') as f:
                    json.dump(output_data, f, indent=2)

                commit_hash = commit_data.get('commit_hash', '')[:8]
                print(f"    ✓ Generated {len(tasks)} tasks for commit_{commit_number} ({commit_hash})")
                return  # Success, exit retry loop
            else:
                print(f"    ⚠️  No tasks generated for commit")
                return  # No tasks is not an error, don't retry

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"    ⚠️  Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  ✗ Error processing {commit_data_path} after {max_retries} attempts: {e}")


async def process_repository_tasks(repo_name: str, base_dir: str = 'data/ast_dataset'):
    """
    Process all commits in a repository to generate tasks.

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

    tasks_generated = 0

    # Process all commits in parallel
    commit_paths = [os.path.join(repo_path, commit_name) for commit_name in commit_dirs]

    # Process commits in parallel
    await asyncio.gather(*[generate_tasks_for_commit(commit_path) for commit_path in commit_paths])

    # Count how many task files were created
    for commit_dir in commit_paths:
        task_file = os.path.join(commit_dir, 'task.json')
        if os.path.exists(task_file):
            tasks_generated += 1

    print(f"\nCompleted {repo_name}: {tasks_generated} commits with tasks generated")


def find_repo_dirs(base_dir: str) -> List[str]:
    """Finds repository directories within the base dataset directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return []
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


async def run_task_pipeline(repos: List[str] = None, base_dir: str = 'data/ast_dataset'):
    """
    Runs the task generation pipeline for specified repositories or all repositories.

    Args:
        repos: List of repository names to process. If None, processes all repos.
        base_dir: Base directory containing AST dataset
    """
    if not HAS_OPENAI:
        print("\nError: OpenAI library not installed.")
        print("Install with: pip install openai python-dotenv")
        print("\nOr add to pyproject.toml dependencies:")
        print('  "openai>=1.0.0",')
        print('  "python-dotenv>=1.0.0"')
        return

    if repos is None:
        repo_names = find_repo_dirs(base_dir)
        if not repo_names:
            print(f"No repository data found in {base_dir}.")
            print("Please run 'generate-ast' first.")
            return
    else:
        repo_names = repos

    print(f"\n{'='*60}")
    print(f"Starting task generation for {len(repo_names)} repositories")
    print(f"{'='*60}")

    for repo_name in repo_names:
        await process_repository_tasks(repo_name, base_dir)

    print(f"\n{'='*60}")
    print(f"All {len(repo_names)} repositories processed!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for task generation."""
    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        asyncio.run(run_task_pipeline(repos))
    else:
        # Process all repos
        asyncio.run(run_task_pipeline())


if __name__ == "__main__":
    cli()
