# Hybrid LLM Trainer

A unified training framework combining LM2's memory-augmented architecture with LongMem's external memory retrieval for efficient input/output training on large language models.

## Overview

This hybrid module integrates:
- **LM2**: Memory-augmented Llama architecture with learnable memory slots
- **LongMem**: External memory bank with FAISS-based retrieval for long-context modeling
- **Unified Training**: Streamlined training pipeline for input/output pairs

## Key Features

1. **Dual Memory System**:
   - Internal memory slots (from LM2) for short-term context
   - External memory bank (from LongMem) for long-term retrieval

2. **Flexible Architecture**:
   - Support for various base models (Llama, GPT-2, etc.)
   - Configurable memory parameters
   - Hybrid attention mechanisms

3. **Efficient Training**:
   - Distributed training support (DDP)
   - Mixed precision training
   - Checkpoint management

## Installation

### Option 1: Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd CodeRL

# Install core dependencies
uv sync

# Install optional dependencies for data generation (OpenAI integration)
uv sync --extra tasks
```

### Option 2: Using Conda

```bash
# Create conda environment
conda env create -n hybrid_llm -f environment.yaml
conda activate hybrid_llm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Generation Pipeline

This project includes a comprehensive AST-based code diff extraction and reasoning generation pipeline for creating training data from Git repositories.

#### Step 1: Configure LLM (for task and reasoning generation)

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API configuration
# OPENAI_API_KEY=your_api_key_here
# LLM_BASE_URL=https://api.openai.com/v1  (optional)
# LLM_MODEL_NAME=gpt-3.5-turbo  (optional)

# Test your LLM connection
uv run test-llm
```

#### Step 2: Manage Repositories

Repositories are configured in `config/config.yaml`. Edit the file to add or remove repositories:

```yaml
repositories:
  - name: bat
    url: "https://github.com/sharkdp/bat.git"
  - name: ripgrep
    url: "https://github.com/BurntSushi/ripgrep.git"
```

Then sync the repositories:

```bash
# Download all configured repositories
uv run manage-repos sync

# Check repository status
uv run manage-repos status

# List downloaded repositories
uv run manage-repos list
```

#### Step 3: Run the Data Generation Pipeline

The pipeline consists of 5 stages that should be run in sequence:

```bash
# Stage 1: Extract AST from commits
uv run generate-ast

# Stage 2: Generate AST diffs between commits
uv run generate-ast-diffs

# Stage 3: Generate task descriptions from diffs (requires LLM)
uv run generate-tasks

# Stage 4: Generate reasoning for code changes (requires LLM)
uv run generate-reasoning

# Stage 5: Build FIM (Fill-in-the-Middle) training dataset
uv run generate-fim
```

**Process specific repositories:**
```bash
uv run generate-ast bat ripgrep
uv run generate-tasks bat
uv run generate-reasoning bat
```

#### Step 4: Clean Up Generated Files (Optional)

If you need to regenerate specific files (e.g., after updating prompts):

```bash
# Dry run to see what would be removed
uv run cleanup-dataset --tasks --reasoning --dry-run

# Remove task.json files to regenerate tasks
uv run cleanup-dataset --tasks

# Remove reasoning.json files to regenerate reasoning
uv run cleanup-dataset --reasoning

# Remove specific files from specific repositories
uv run cleanup-dataset --tasks --reasoning bat ripgrep
```

#### Output Structure

The pipeline generates the following structure:
```
data/ast_dataset/
├── bat/
│   ├── commit_0/
│   │   ├── ast.jsonl              # AST nodes for this commit
│   │   ├── commit_data.json       # Commit metadata and diff
│   │   ├── diff_ast.jsonl         # Function-level changes
│   │   ├── task.json              # LLM-generated tasks
│   │   ├── reasoning.json         # LLM-generated reasoning
│   │   └── fim_dataset.json       # FIM training samples
│   ├── commit_1/
│   └── ...
└── ripgrep/
    └── ...
```

### 2. Train the Model

```bash
# Basic training
python train.py \
  model=llama_hybrid \
  train.batch_size=4 \
  train.learning_rate=1e-4

# Or use the training script
bash scripts/train_hybrid.sh
```

### 3. Evaluate

```bash
python eval.py \
  --checkpoint checkpoints/best_model.pt \
  --test_data datasets/io_pairs/test
```

## Architecture

### Hybrid Memory Module

```
Input Sequence
     ↓
Embedding Layer
     ↓
┌─────────────────────────────────┐
│  Transformer Layers             │
│  ┌──────────────────────────┐  │
│  │ Self-Attention           │  │
│  │ ↓                        │  │
│  │ Internal Memory Module   │  │ ← LM2 Memory Slots
│  │ ↓                        │  │
│  │ External Memory Retrieval│  │ ← LongMem FAISS Index
│  │ ↓                        │  │
│  │ Joint Attention Fusion   │  │
│  │ ↓                        │  │
│  │ Feed-Forward Network     │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
     ↓
Output Logits
```

## Configuration

Key configuration parameters in `configs/train.yaml`:

```yaml
model:
  model_type: llama_hybrid
  use_internal_memory: true
  use_external_memory: true
  memory_slots: 16
  num_mem_heads: 4
  external_memory_size: 1048576
  retrieval_k: 8

train:
  batch_size: 4
  learning_rate: 1e-4
  max_iters: 100000
  dtype: bfloat16
```

## Data Generation Pipeline Details

The pipeline extracts function-level code changes from Git repositories and generates training data with tasks and reasoning.

### Pipeline Stages

1. **`generate-ast`**: Extract AST nodes from each commit
   - Uses tree-sitter to parse source files
   - Extracts functions, classes, methods, and other declarations
   - Outputs: `ast.jsonl`, `commit_data.json`

2. **`generate-ast-diffs`**: Compare AST between commits
   - Identifies added, modified, and deleted functions
   - Tracks function-level changes with before/after code
   - Outputs: `diff_ast.jsonl`

3. **`generate-tasks`**: Generate task descriptions using LLM
   - Analyzes commit diffs to create developer tasks
   - Chunks large diffs for better LLM processing
   - Outputs: `task.json`
   - Requires: OpenAI API key in `.env`

4. **`generate-reasoning`**: Generate step-by-step reasoning using LLM
   - Creates first-person developer perspective reasoning
   - Tailored prompts for ADD/UPDATE/DELETE operations
   - Filters tasks relevant to each function
   - Outputs: `reasoning.json`
   - Requires: OpenAI API key in `.env`

5. **`generate-fim`**: Build Fill-in-the-Middle training dataset
   - Extracts code context (prefix/suffix)
   - Identifies exact changed lines
   - Creates training samples for code completion
   - Outputs: `fim_dataset.json`

### Data Format Examples

**diff_ast.jsonl** (function-level changes):
```json
{
  "id": "src/main.rs::function::run",
  "file": "src/main.rs",
  "kind": "function_item",
  "status": "modified",
  "before_code": "fn run() { ... }",
  "after_code": "fn run() { ... }",
  "commit_metadata": {...}
}
```

**task.json** (LLM-generated tasks):
```json
{
  "commit_number": 5,
  "commit_hash": "a7232a6e",
  "tasks": [
    "Implement custom syntax loading from user config directory",
    "Add fallback to default syntax set if custom loading fails"
  ]
}
```

**reasoning.json** (LLM-generated reasoning):
```json
{
  "function_id": "src/main.rs::function::run",
  "operation_type": "UPDATE",
  "reasoning": "Looking at the task, I need to add custom syntax loading. First, I'll construct the path to the .config/bat/syntax directory..."
}
```

**fim_dataset.json** (training samples):
```json
{
  "function_id": "src/main.rs::function::run",
  "operation_type": "UPDATE",
  "fim_prefix": "fn run() {\n    let options = Options {...};\n",
  "fim_middle": "    let mut syntax_set = SyntaxSet::new();\n    syntax_set.load_syntaxes(syntax_dir, false);",
  "fim_suffix": "\n    Ok(())\n}"
}
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `uv run manage-repos sync` | Download all configured repositories |
| `uv run manage-repos status` | Show repository download status |
| `uv run manage-repos list` | List downloaded repositories |
| `uv run generate-ast [repos...]` | Extract AST from commits |
| `uv run generate-ast-diffs [repos...]` | Generate AST diffs |
| `uv run generate-tasks [repos...]` | Generate task descriptions (LLM) |
| `uv run generate-reasoning [repos...]` | Generate reasoning (LLM) |
| `uv run generate-fim [repos...]` | Build FIM training dataset |
| `uv run test-llm` | Test LLM API connection |
| `uv run cleanup-dataset --tasks/--reasoning/--fim [repos...]` | Remove generated files |
| `uv run check-ast-coverage` | Check AST extraction coverage |

## Project Structure

```
CodeRL/
├── src/mem_aug/
│   ├── components/
│   │   └── datagen/
│   │       ├── generate_commit_ast.py    # AST extraction
│   │       ├── generate_ast_diffs.py     # Diff generation
│   │       ├── generate_tasks.py         # Task generation (LLM)
│   │       ├── generate_reasoning.py     # Reasoning generation (LLM)
│   │       └── build_fim_dataset.py      # FIM dataset builder
│   └── utils/
│       ├── repo_manager.py               # Repository management
│       ├── test_llm.py                   # LLM connection tester
│       └── cleanup_dataset.py            # Dataset cleanup utility
├── data/
│   ├── ast_dataset/                      # Generated datasets
│   └── repos/                            # Cloned repositories
├── configs/              # Configuration files
├── scripts/              # Training scripts
├── .env.example          # LLM configuration template
├── pyproject.toml        # Project dependencies and CLI commands
└── README.md
```

## Training on Input/Output Pairs

The hybrid trainer is specifically designed for input/output training:

1. **Input Phase**: Model processes input with both memory systems active
2. **Memory Update**: Internal and external memories are updated with input context
3. **Output Phase**: Model generates output leveraging both memory systems
4. **Loss Calculation**: Cross-entropy loss on output tokens only

Example data format:
```json
{
  "input": "Translate to French: Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

## Advanced Usage

### Custom Memory Configuration

```python
from src.models.hybrid_llama import HybridLlamaConfig

config = HybridLlamaConfig(
    use_internal_memory=True,
    use_external_memory=True,
    memory_slots=32,
    num_mem_heads=8,
    external_memory_size=2097152,
    retrieval_k=16,
    chunk_size=4
)
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 train.py \
  model=llama_hybrid \
  train.batch_size=2
```

## Performance Tips

1. **Memory Management**:
   - Adjust `memory_slots` based on GPU memory
   - Use `external_memory_size` for long-context tasks
   - Enable `use_gpu_to_search` for faster retrieval

2. **Training Efficiency**:
   - Use mixed precision (`dtype=bfloat16`)
   - Gradient accumulation for larger effective batch sizes
   - Checkpoint activations for memory-intensive models

3. **Data Processing**:
   - Pre-tokenize datasets for faster loading
   - Use appropriate sequence lengths
   - Balance input/output lengths

## Citation

If you use this hybrid trainer, please cite both original papers:

```bibtex
@article{LM2,
  title={LM2: Large Memory Models},
  author={...},
  journal={arXiv preprint arXiv:2502.06049v1},
  year={2025}
}

@article{LongMem,
  title={Augmenting Language Models with Long-Term Memory},
  author={Wang, Weizhi and Dong, Li and Cheng, Hao and Liu, Xiaodong and Yan, Xifeng and Gao, Jianfeng and Wei, Furu},
  journal={arXiv preprint arXiv:2306.07174},
  year={2023}
}
```

## License

This project combines components from LM2 (CC BY-NC 4.0) and LongMem (MIT License). Please refer to individual licenses for specific terms.

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## Support

For questions and issues, please open a GitHub issue or contact the maintainers.
