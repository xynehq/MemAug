#!/usr/bin/env python3
"""
Complete data generation pipeline orchestrator.

Runs the entire data generation pipeline in sequence:
1. generate-ast (commit AST generation)
2. generate-ast-diffs (AST diff generation) 
3. generate-edit-seq (edit sequence dataset)
4. generate-enhanced-tasks (enhanced task generation)

Each stage waits for the previous stage to complete ALL repositories 
before proceeding to ensure data consistency.
"""
import asyncio
import subprocess
import sys
import time
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any
from mem_aug.utils.repo_manager import get_project_root


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config" / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}")
        return {}


def get_repository_names(config: Dict[str, Any]) -> List[str]:
    """Extract repository names from configuration."""
    repos = config.get('repositories', [])
    return [repo['name'] for repo in repos if isinstance(repo, dict) and 'name' in repo]


async def run_command_async(cmd: List[str], stage_name: str) -> bool:
    """Run a command asynchronously and return success status."""
    print(f"  🚀 Starting: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        duration = time.time() - start_time
        
        if process.returncode == 0:
            print(f"  ✅ Completed in {duration:.1f}s: {' '.join(cmd)}")
            if stdout:
                print(f"     Output: {stdout.decode().strip()}")
            return True
        else:
            print(f"  ❌ Failed in {duration:.1f}s: {' '.join(cmd)}")
            if stderr:
                print(f"     Error: {stderr.decode().strip()}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"  💥 Exception in {duration:.1f}s: {e}")
        return False


def check_stage_completion(stage_name: str, repos: List[str], base_dir: str = "data/ast_dataset") -> Dict[str, bool]:
    """Check which repositories have completed a given stage."""
    completion_status = {}
    
    for repo in repos:
        repo_path = Path(base_dir) / repo
        completed = False
        
        if stage_name == "generate-ast":
            # Check if commit directories exist with commit_data.json files
            if repo_path.exists():
                commit_dirs = [d for d in repo_path.iterdir() if d.is_dir() and d.name.startswith('commit_')]
                completed = len(commit_dirs) > 0 and all(
                    (d / 'commit_data.json').exists() for d in commit_dirs
                )
        
        elif stage_name == "generate-ast-diffs":
            # Check if AST diff files exist
            if repo_path.exists():
                commit_dirs = [d for d in repo_path.iterdir() if d.is_dir() and d.name.startswith('commit_')]
                completed = len(commit_dirs) > 0 and all(
                    (d / 'ast_diff.json').exists() for d in commit_dirs
                )
        
        elif stage_name == "generate-edit-seq":
            # Check if edit sequence files exist
            if repo_path.exists():
                commit_dirs = [d for d in repo_path.iterdir() if d.is_dir() and d.name.startswith('commit_')]
                completed = len(commit_dirs) > 0 and all(
                    (d / 'edit_sequence_dataset.json').exists() for d in commit_dirs
                )
        
        elif stage_name == "generate-enhanced-tasks":
            # Check if enhanced task files exist
            if repo_path.exists():
                commit_dirs = [d for d in repo_path.iterdir() if d.is_dir() and d.name.startswith('commit_')]
                completed = len(commit_dirs) > 0 and all(
                    (d / 'enhanced_tasks.json').exists() for d in commit_dirs
                )
        
        completion_status[repo] = completed
    
    return completion_status


async def wait_for_stage_completion(stage_name: str, repos: List[str], check_interval: int = 30):
    """Wait for all repositories to complete a stage before proceeding."""
    print(f"\n⏳ Waiting for all repositories to complete {stage_name}...")
    
    while True:
        completion_status = check_stage_completion(stage_name, repos)
        completed_repos = [repo for repo, status in completion_status.items() if status]
        pending_repos = [repo for repo, status in completion_status.items() if not status]
        
        print(f"   ✅ Completed: {len(completed_repos)}/{len(repos)} repositories")
        if pending_repos:
            print(f"   ⏳ Pending: {', '.join(pending_repos)}")
        
        if len(completed_repos) == len(repos):
            print(f"   🎉 All repositories completed {stage_name}!")
            break
        
        print(f"   💤 Waiting {check_interval}s before next check...")
        await asyncio.sleep(check_interval)


async def run_pipeline_stage(stage_name: str, repos: List[str], max_parallel: int = 4) -> bool:
    """Run a pipeline stage for all repositories with parallel execution."""
    print(f"\n{'='*80}")
    print(f"🔄 STAGE: {stage_name.upper()}")
    print(f"{'='*80}")
    print(f"📊 Processing {len(repos)} repositories with max {max_parallel} parallel")
    
    # Create semaphore for parallel execution control
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def run_single_repo(repo: str) -> bool:
        """Run command for a single repository with semaphore control."""
        async with semaphore:
            cmd = ["uv", "run", stage_name, repo]
            return await run_command_async(cmd, stage_name)
    
    # Run all repositories in parallel (controlled by semaphore)
    start_time = time.time()
    tasks = [run_single_repo(repo) for repo in repos]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successes
    successful = sum(1 for result in results if result is True)
    duration = time.time() - start_time
    
    print(f"\n📈 STAGE SUMMARY: {stage_name}")
    print(f"   ✅ Successful: {successful}/{len(repos)} repositories")
    print(f"   ⏱️  Duration: {duration:.1f}s")
    
    if successful == len(repos):
        print(f"   🎉 Stage {stage_name} completed successfully!")
        return True
    else:
        failed = len(repos) - successful
        print(f"   ⚠️  {failed} repositories failed")
        return False


async def run_complete_pipeline(repos: List[str] = None, skip_stages: List[str] = None):
    """Run the complete data generation pipeline."""
    print(f"🚀 Starting Complete Data Generation Pipeline")
    print(f"{'='*80}")
    
    # Load configuration
    config = load_config()
    
    # Get repositories
    if repos is None:
        repos = get_repository_names(config)
    
    if not repos:
        print("❌ No repositories found to process")
        return
    
    print(f"📂 Repositories: {', '.join(repos)}")
    
    # Define pipeline stages
    stages = [
        "generate-ast",
        "generate-ast-diffs", 
        "generate-edit-seq",
        "generate-enhanced-tasks"
    ]
    
    if skip_stages:
        stages = [stage for stage in stages if stage not in skip_stages]
        print(f"⏭️  Skipping stages: {', '.join(skip_stages)}")
    
    print(f"🔄 Pipeline stages: {' → '.join(stages)}")
    
    # Get parallelism settings
    max_parallel = min(len(repos), 4)  # Don't overwhelm the system
    
    start_time = time.time()
    
    try:
        for i, stage in enumerate(stages, 1):
            print(f"\n🎯 PIPELINE PROGRESS: Stage {i}/{len(stages)}")
            
            # Run the stage
            success = await run_pipeline_stage(stage, repos, max_parallel)
            
            if not success:
                print(f"💥 Pipeline failed at stage: {stage}")
                return False
            
            # Wait for completion before proceeding to next stage
            if i < len(stages):  # Don't wait after the last stage
                await wait_for_stage_completion(stage, repos)
        
        # Pipeline completed successfully
        total_duration = time.time() - start_time
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"📊 Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"📂 Processed {len(repos)} repositories through {len(stages)} stages")
        print(f"✨ All data generation completed!")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Pipeline failed with exception: {e}")
        return False


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete data generation pipeline')
    parser.add_argument('repos', nargs='*', help='Repository names to process (default: all from config)')
    parser.add_argument('--skip', nargs='*', choices=['generate-ast', 'generate-ast-diffs', 'generate-edit-seq', 'generate-enhanced-tasks'],
                       help='Stages to skip')
    parser.add_argument('--check-completion', action='store_true', help='Check completion status and exit')
    
    args = parser.parse_args()
    
    if args.check_completion:
        # Check completion status for all stages
        config = load_config()
        repos = args.repos if args.repos else get_repository_names(config)
        
        stages = ['generate-ast', 'generate-ast-diffs', 'generate-edit-seq', 'generate-enhanced-tasks']
        
        print("📊 Pipeline Completion Status:")
        print("="*50)
        
        for stage in stages:
            completion = check_stage_completion(stage, repos)
            completed = sum(completion.values())
            total = len(repos)
            percentage = (completed / total * 100) if total > 0 else 0
            
            print(f"{stage:20s}: {completed:2d}/{total} ({percentage:5.1f}%)")
            
            if completed < total:
                pending = [repo for repo, status in completion.items() if not status]
                print(f"{'':20s}  Pending: {', '.join(pending)}")
        
        return
    
    # Run the pipeline
    repos = args.repos if args.repos else None
    skip_stages = args.skip if args.skip else None
    
    asyncio.run(run_complete_pipeline(repos, skip_stages))


if __name__ == "__main__":
    main()