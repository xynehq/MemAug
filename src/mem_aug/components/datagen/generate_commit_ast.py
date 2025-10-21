'''
Used to generate the AST-dataset by iterating through commits of specified repositories.
'''
import subprocess
import os
import json
import shutil # For copying directories
import tempfile # For creating temporary directories
from mem_aug.utils.extract_rust_ast import main as extract_ast_main # Import the main function from the AST extraction script
from mem_aug.utils.repo_manager import load_config, list_downloaded_repos
import asyncio
import concurrent.futures

def branch_exists(repo_path, branch_name):
    result = subprocess.run(
        ['git', 'show-ref', '--verify', f'refs/heads/{branch_name}'],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def get_commit_hashes(repo_path, limit=None, order='first'):
    """
    Get commit hashes from repository, always in chronological order.

    Args:
        repo_path: Path to git repository
        limit: Maximum number of commits to return
        order: 'first' to select from oldest commits, 'last' to select from newest commits

    Returns:
        List of commit hashes in chronological order (oldest to newest)
    """
    # Determine target branch
    target_branch = None
    if branch_exists(repo_path, 'main'):
        target_branch = 'main'
    elif branch_exists(repo_path, 'master'):
        target_branch = 'master'
    else:
        # If neither main nor master exists, try to get the default branch
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            target_branch = result.stdout.strip()
        except subprocess.CalledProcessError:
            raise Exception(f"Neither 'main', 'master' nor any other branch found in {repo_path}")

    if order == 'first':
        # Get first N commits (oldest)
        # Strategy: Get ALL commits in reverse chronological order (oldest first), then take first N
        command = ['git', 'rev-list', '--reverse', target_branch]
        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        all_commits = result.stdout.strip().split('\n')
        # Take first N commits if limit specified, otherwise all
        commits = all_commits[:limit] if limit else all_commits
    else:  # order == 'last'
        # Get last N commits (newest)
        # Strategy: Get newest N commits (they come newest-first), then reverse to chronological
        command = ['git', 'rev-list']
        if limit:
            command.append(f'--max-count={limit}')
        command.append(target_branch)

        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commits = result.stdout.strip().split('\n')
        # Reverse to get chronological order (oldest to newest)
        commits = list(reversed(commits))

    return commits

def get_commit_diff(repo_path, commit_hash):
    result = subprocess.run(
        ['git', 'show', commit_hash],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
        errors='replace'
    )
    return result.stdout

def get_files_modified_in_commit(repo_path, commit_hash):
    result = subprocess.run(
        ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def get_commit_metadata(repo_path, commit_hash):
    result = subprocess.run(
        ['git', 'show', '-s', '--format=%an%n%ae%n%ad%n%B', commit_hash],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
        errors='replace'
    )
    output = result.stdout.strip().split('\n', 3)
    author_name = output[0] if len(output) > 0 else "N/A"
    author_email = output[1] if len(output) > 1 else "N/A"
    commit_date = output[2] if len(output) > 2 else "N/A"
    commit_message = output[3] if len(output) > 3 else "N/A"
    
    return {
        "author_name": author_name,
        "author_email": author_email,
        "commit_date": commit_date,
        "commit_message": commit_message
    }

def get_current_branch(repo_path):
    result = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def process_repository(repo_name):
    # Load config to get processing settings
    config = load_config()
    max_commits = config.get('processing', {}).get('max_commits', None)
    commit_order = config.get('processing', {}).get('commit_order', 'first')

    source_repo_path = os.path.join('data/repos', repo_name)
    # Create the main dataset directory
    dataset_base_dir = 'data/ast_dataset'
    os.makedirs(dataset_base_dir, exist_ok=True)

    # Create repository-specific directory under dataset
    repo_dataset_dir = os.path.join(dataset_base_dir, repo_name)
    os.makedirs(repo_dataset_dir, exist_ok=True)

    # Keep the original code_base directory for repository files
    base_output_dir = os.path.join('data/temp', repo_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # Create a temporary directory for processing the repository
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the source repository to the temporary directory
        temp_repo_path = os.path.join(temp_dir, repo_name)
        try:
            shutil.copytree(source_repo_path, temp_repo_path)
        except Exception as e:
            print(f"Error copying repository {repo_name}: {e}")
            return

        # Rename .git_disabled to .git if it exists
        git_disabled_path = os.path.join(temp_repo_path, '.git_disabled')
        git_path = os.path.join(temp_repo_path, '.git')
        if os.path.exists(git_disabled_path):
            os.rename(git_disabled_path, git_path)

        original_branch = get_current_branch(temp_repo_path)

        # Stash any local changes before starting to process commits (on the temporary repo)
        try:
            subprocess.run(
                ['git', 'stash', 'save', '--keep-index', '--include-untracked', 'Automated stash for commit processing'],
                cwd=temp_repo_path,
                capture_output=True,
                text=True,
                check=False # Don't check for errors, as there might be no changes to stash
            )
        except Exception as e:
            print(f"Error stashing changes in {repo_name}: {e}")

        commit_hashes = get_commit_hashes(temp_repo_path, limit=max_commits, order=commit_order)
        total_commits = len(commit_hashes)

        order_msg = "oldest" if commit_order == 'first' else "newest"
        commit_count_msg = f"all {total_commits} commits" if max_commits is None else f"{total_commits} {order_msg} commits"
        print(f"Processing {repo_name}: {commit_count_msg}")

        for i, commit_hash in enumerate(commit_hashes, 1):

            # Create commit directory in the new dataset structure
            commit_dataset_dir = os.path.join(repo_dataset_dir, f'commit_{i}')
            os.makedirs(commit_dataset_dir, exist_ok=True)

            # Keep the original commit directory for repository files
            commit_output_dir = os.path.join(base_output_dir, f'commit_{i}')
            os.makedirs(commit_output_dir, exist_ok=True)

            # Checkout the repository to the specific commit
            try:
                subprocess.run(
                    ['git', 'checkout', commit_hash],
                    cwd=temp_repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error checking out commit {commit_hash} in {temp_repo_path}: {e.stderr}")
                continue # Skip this commit if checkout fails

            # Copy the entire repository content (excluding .git) to the commit's output directory
            # Using git archive is cleaner than cp -r to avoid copying .git directory
            try:
                archive_command = ['git', 'archive', '--format=tar', 'HEAD']
                archive_process = subprocess.Popen(
                    archive_command,
                    cwd=temp_repo_path,
                    stdout=subprocess.PIPE
                )
                tar_command = ['tar', '-x', '-C', commit_output_dir]
                subprocess.run(
                    tar_command,
                    stdin=archive_process.stdout,
                    check=True
                )
                archive_process.stdout.close()
                archive_process.wait()
            except subprocess.CalledProcessError as e:
                print(f"Error archiving/extracting repo at commit {commit_hash} to {commit_output_dir}: {e}")
                # Clean up the partially created directory
                shutil.rmtree(commit_output_dir)
                continue

            # Clean up unnecessary git-related files/directories
            git_dir_in_output = os.path.join(commit_output_dir, '.git')
            if os.path.exists(git_dir_in_output):
                shutil.rmtree(git_dir_in_output)

            # List of other common git-related files to remove
            git_related_files = ['.gitattributes', '.gitignore', '.gitmodules']
            for git_file in git_related_files:
                file_to_remove = os.path.join(commit_output_dir, git_file)
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)

            # Get diff content for the current commit
            diff_content = get_commit_diff(temp_repo_path, commit_hash)
            modified_files = get_files_modified_in_commit(temp_repo_path, commit_hash)
            commit_metadata = get_commit_metadata(temp_repo_path, commit_hash)

            files_data = []
            for file_path in modified_files:
                # Only process Rust files, and ensure they exist in the current commit state
                full_file_path_in_commit_dir = os.path.join(commit_output_dir, file_path)
                if file_path.endswith('.rs') and os.path.exists(full_file_path_in_commit_dir):
                    with open(full_file_path_in_commit_dir, 'r', errors='replace') as f:
                        file_content = f.read()

                    files_data.append({
                        "file_path": file_path,
                        "content": file_content,
                    })
            
            commit_data = {
                "commit_hash": commit_hash,
                "diff": diff_content,
                "metadata": commit_metadata,
                "files_modified": files_data
            }
            
            # Save commit_data.json in the new dataset directory
            commit_data_path = os.path.join(commit_dataset_dir, 'commit_data.json')
            with open(commit_data_path, 'w') as f:
                json.dump(commit_data, f, indent=4)

            # Call the AST extraction script for the current commit directory
            ast_output_jsonl_path = os.path.join(commit_dataset_dir, 'ast.jsonl')
            try:
                extract_ast_main(commit_output_dir, ast_output_jsonl_path)
            except Exception as e:
                print(f"Error extracting AST for commit {commit_hash}: {e}")

            # Clean up the commit output directory to free up space
            # TEMPORARILY DISABLED - for debugging
            try:
                shutil.rmtree(commit_output_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up {commit_output_dir}: {e}")

            # Print completion message
            print(f"  ✓ {repo_name}: commit {i}/{total_commits} ({commit_hash[:8]})")

        # After processing all commits, checkout back to the original branch
        try:
            subprocess.run(
                ['git', 'checkout', original_branch],
                cwd=temp_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error checking out {repo_name} back to {original_branch}: {e.stderr}")

        # Unstash changes
        try:
            subprocess.run(
                ['git', 'stash', 'pop'],
                cwd=temp_repo_path,
                capture_output=True,
                text=True,
                check=False # Don't check for errors, as there might be no stashed changes
            )
        except Exception as e:
            print(f"Error unstashing changes in {repo_name}: {e}")

    print(f"Completed {repo_name}: {total_commits} commits processed")

# if __name__ == "__main__":
#     repositories = ['bat', 'ripgrep', 'rust_calculator', 'starship'] # Example repositories
#     for repo in repositories:
#         print(f"Starting processing for repository: {repo}")
#         process_repository(repo)

async def process_repository_async(repo_name):
    """Async wrapper for process_repository"""
    loop = asyncio.get_event_loop()
    
    # Run the CPU-bound process_repository function in a thread pool
    with concurrent.futures.ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, process_repository, repo_name)

async def main(repos=None):
    """
    Process repositories to generate AST dataset.

    Args:
        repos: List of repository names to process. If None, processes all downloaded repos from config.
    """
    if repos is None:
        # Get downloaded repos
        downloaded = list_downloaded_repos()
        if not downloaded:
            print("No repositories downloaded. Run 'manage-repos sync' first.")
            return
        repositories = downloaded
    else:
        repositories = repos

    print(f"\n{'='*60}")
    print(f"Starting AST generation for {len(repositories)} repositories")
    print(f"{'='*60}\n")

    # Create tasks for all repositories
    tasks = []
    for repo in repositories:
        task = asyncio.create_task(process_repository_async(repo))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"\n{'='*60}")
    print(f"All {len(repositories)} repositories processed successfully!")
    print(f"{'='*60}\n")


def cli():
    """Command-line interface for AST generation."""
    import sys

    if len(sys.argv) > 1:
        # Process specific repos provided as arguments
        repos = sys.argv[1:]
        asyncio.run(main(repos))
    else:
        # Process all downloaded repos
        asyncio.run(main())


if __name__ == "__main__":
    cli()
