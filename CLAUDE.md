# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project focused on causal analysis of transformer language models, specifically analyzing how different attention mechanisms contribute to in-context learning patterns. The codebase implements activation patching techniques to understand causal relationships in model behavior.

## Core Architecture


### Key Classes and Functions

- `CustomHookTransformer`: Extended transformer_lens.HookedTransformer with custom generation and caching capabilities
- `activation_patching()`: Core function that performs causal mediation by patching activations between different prompt conditions
- `ablate_head()` and `ablate_layer()`: Functions for systematically ablating different components (attention heads or residual stream layers)
- `generate_prompts()`: Creates structured prompt pairs for testing different rule patterns (ABA vs ABB)

## Running the Code

### Basic Execution
```bash
python causal_analysis.py --model_type "8B" --prompt_num 100 --base_rule "ABA"
```

### Key Command Line Arguments

- `--model_type`: Model variant (8B, 70B, 8B-Instruct, Qwen2.5-14B-Instruct, etc.)
- `--activation_name`: Which activation to patch (z, q, k, v, resid_pre, etc.)
- `--base_rule`: Base rule pattern (ABA or ABB)
- `--prompt_num`: Number of prompts to generate
- `--sample_num`: Number of samples for causal mediation analysis
- `--token_pos_list`: Token positions for patching (default: [-1])
- `--generate`: Use generation-based evaluation instead of logit differences
- `--eval_generation`: Filter prompts by generation accuracy

### Configuration Requirements

- Set HF_TOKEN in utils.py for Hugging Face model access (both main utils.py and codebase/utils.py)
- Configure XDG_CACHE_HOME for model caching
- Ensure adequate GPU memory for transformer models

### Model Support & Fixes

**Supported Models**:
- Llama-3.1 series: 8B, 70B, 8B-Instruct, 70B-Instruct
- Qwen2.5 series: 7B, 14B, 14B-Instruct, 32B, 72B
- GPT-2 variants, Gemma-2 series

**Automatic Embedding Resize**: The codebase automatically handles tokenizer vocab size mismatches between base and instruct models (e.g., Qwen2.5-14B vs Qwen2.5-14B-Instruct). When model embedding size differs from tokenizer vocab size, embeddings are auto-resized to prevent tensor dimension errors.

## Research Context

### Experimental Design
The code implements causal mediation analysis to understand how transformer attention mechanisms process in-context learning rules. It compares model behavior on two rule types:
- ABA pattern: First and third tokens match
- ABB pattern: Second and third tokens match

## LLMSymbMech Integration (Now in codebase/)

The `codebase/` folder contains the ICML 2025 "Emergent Symbolic Mechanisms" codebase (copied from LLMSymbMech), which implements a three-stage symbolic processing framework. This extends our causal analysis approach with targeted mechanism identification.

### Three-Stage Symbolic Processing

The LLMSymbMech approach identifies three distinct attention head types:

1. **Symbol abstraction heads**: Identify relations between input tokens and represent them using abstract variables
2. **Symbolic induction heads**: Perform sequence induction over abstract variables
3. **Retrieval heads**: Predict tokens by retrieving values associated with predicted abstract variables

### Key Methodological Differences


**LLMSymbMech Approach (codebase/tasks/identity_rules/cma.py)**:
- Targeted patching of specific head types using predefined context pairs
- Abstract vs token context pairs for different head identification
- Pre-computed significant heads from statistical testing
- Generation accuracy evaluation (`gen_acc`) vs logit differences
- We have added extensions and new parameters to test true and false belief conditions

### Integration Benefits

- **Hypothesis-driven analysis**: Use their significant head identification to focus our analysis
- **Mechanism validation**: Test whether our models exhibit the same three-stage processing
- **Enhanced visualization**: Their heatmap approach shows clearer layer×head patterns
- **Cross-validation**: Compare our broad findings with their specific mechanism predictions

### Dataset Integration

**LLMSymbMech Dataset Structure**:
- `datasets/vocab/`: Curated english-only tokens per model family
- `datasets/cma_scores/`: Pre-computed causal scores and significant heads
- `datasets/*_correct_common_tokens_*.txt`: High-accuracy token subsets (e.g., 1378 tokens for Llama-3.1-70B)
- Pre-generated prompt files for exact replication

### Activation Patching Process
1. Generate prompt pairs that differ in rule structure or token order
2. Run forward passes to collect activations
3. Systematically patch activations from one condition to another
4. Measure changes in model predictions to identify causal components

### Output Analysis
Results are saved as heatmaps showing the causal importance of different model components (layers × heads or layers × positions) for rule following behavior.

## Behavioral Evaluation System (January 2025)

We've added a comprehensive **behavioral evaluation system** for pure theory of mind performance assessment without mechanistic analysis. This addresses the core question: "which models can actually perform false belief reasoning?"

### Key Features

- **Pure Behavioral**: No activation caching/patching - just generation + accuracy measurement
- **Wandb Integration**: Rich experiment tracking with interactive plots across models/temperatures  
- **Temperature Sweeps**: Systematic evaluation across temperature ranges
- **Format Comparison**: Direct vs multiple-choice prompt formats
- **Dual-Answer Parsing**: Separate belief vs world state accuracy tracking
- **Modular Design**: Reuses existing prompt generators but bypasses mechanistic components

### Running Behavioral Evaluation

**Located in**: `behavioral/` directory (separate from mechanistic analysis)

**Basic Commands**:
```bash
# Test basic functionality
.conda/python.exe behavioral/test_basic.py

# Quick test (minimal samples)
.conda/python.exe behavioral_evaluation/behavioral_eval.py --config_type quick_test --no_wandb

# Full comparison (target models: Llama-3.1-7B-Instruct, Qwen2.5-14B-Instruct)
.conda/python.exe behavioral/behavioral_eval.py --config_type full_comparison

# Temperature sweep for specific model
.conda/python.exe behavioral/behavioral_eval.py \
  --config_type temperature_sweep \
  --single_model "meta-llama/Llama-3.1-7B-Instruct"

# Custom configuration
.conda/python.exe behavioral/behavioral_eval.py \
  --models "meta-llama/Llama-3.1-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" \
  --temperatures 0.1 0.4 0.7 1.0 1.3 1.6 1.9 \
  --samples_per_condition 50 \
  --prompt_num 50
```

**Wandb Setup**: 
- Environment variable: `export WANDB_API_KEY="your_key"`
- Fallback in `codebase/utils.py` for interactive sessions
- Use `--no_wandb` flag to disable if needed

### Output Structure

**Local Results**: `results/tom_performance/`
```
results/tom_performance/
├── model_name_timestamp_behavioral_results.json
├── combined_results_timestamp_behavioral_results.json
└── raw_responses/
    └── model_responses_by_condition.json
```

**Wandb Dashboard**:
- Belief vs world accuracy across temperatures
- Model family comparisons (Qwen vs Llama)  
- Format effectiveness (direct vs multiple_choice)
- Rich metadata tagging for filtering/analysis

### Experimental Dimensions

Each experiment varies across:
- **Models**: Llama-3.1-7B-Instruct, Qwen2.5-14B-Instruct
- **Temperatures**: [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]  
- **Vignette Types**: false_belief, true_belief
- **TOM Formats**: direct, multiple_choice
- **Sample Size**: 50 per condition (configurable)

### Integration with Mechanistic Analysis

The behavioral system is **completely independent** but uses identical prompts via shared generators:
- Same theory of mind scenarios as mechanistic CMA
- Same dual-answer schema for consistency  
- Results inform which models warrant mechanistic investigation
- Fast iteration without GPU-intensive caching/patching

This enables the workflow: **behavioral screening → mechanistic deep-dive** for models that show good TOM performance.

## Dependencies

The project requires:
- transformers (Hugging Face)
- transformer_lens
- torch
- numpy
- matplotlib/seaborn for visualization
- scipy for statistical analysis
- wandb (for behavioral evaluation tracking)


## Setup Instructions

### Prerequisites
1. **Python Environment**: Python 3.8+ with pip
2. **GPU Access**: CUDA-capable GPU recommended for transformer models
3. **HuggingFace Account**: Required for model access and API

### Installation Steps

1. **Environment Activation** (Current Setup):
   ```bash
   # RECOMMENDED: Use wrapper script (always works)
   ./run_with_env.sh python script_name.py
   
   # INTERACTIVE: Use activation script for terminal sessions
   source activate_tom_env.sh        # Linux/WSL/Mac
   activate_tom_env.bat              # Windows Command Prompt
   
   # DIRECT: Run Python directly 
   .conda/python.exe script_name.py
   
   # LEGACY: Original setup script (single session only)
   . setup.sh
   ```

   **Note**: The current `.conda` directory contains a full conda distribution rather than a proper environment. This works but isn't standard practice.

2. **Alternative: Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Authentication**:
   - Get HuggingFace token: https://huggingface.co/settings/tokens
   - Set environment variable: `export HF_TOKEN="your_token_here"`
   - Or edit `utils.py` line 10 to replace the placeholder

4. **HPC Configuration** (Dartmouth users):
   - Uncomment and modify cache path in `utils.py`:
     ```python
     XDG_CACHE_HOME = "/scratch/gpfs/yourusername/.cache"
     ```

### Long-term Environment Fix (Recommended)

The current setup works but uses a full conda distribution in `.conda/` instead of a proper environment. For better practice:

```bash
# 1. Backup current working setup
mv .conda .conda_backup

# 2. Create proper conda environment (if conda available)
conda create -n tom_analysis python=3.12
conda activate tom_analysis
pip install -r requirements.txt

# 3. OR create virtual environment
python -m venv tom_env
source tom_env/bin/activate  # Linux/Mac
# OR tom_env\Scripts\activate  # Windows
pip install -r requirements.txt

# 4. Update scripts to use proper environment name
# conda activate tom_analysis
```

**Why this matters**: Proper environments can be activated from any terminal, are more portable, and follow Python packaging best practices.

## HPC Troubleshooting

### Common Issue: "ModuleNotFoundError: Could not import module 'LlamaForCausalLM'"

This typically occurs on HPC when transformers library is outdated or has conflicts.

**Quick Diagnosis**:
```bash
# On HPC, run the diagnostic script
python diagnose_hpc_env.py
```

**Quick Fix**:
```bash
# If diagnostic shows version issues
bash fix_hpc_transformers.sh
```

**Manual Fix**:
```bash
# Update transformers and dependencies
pip install --upgrade transformers>=4.30.0
pip install --upgrade torch accelerate tokenizers

# Test import
python -c "from transformers.models.llama import LlamaForCausalLM; print('✅ Success!')"
```

**Root Causes**:
- Transformers version < 4.21.0 (Llama support added in 4.21+)
- Conflicting torch/transformers versions
- Missing tokenizers dependency
- Cached old model files
- **Corrupted package installations** (e.g., `-orch` instead of `torch`)

**If you see**: `WARNING: Ignoring invalid distribution -orch`

**For HPC** (after `conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis`):
```bash
bash fix_hpc_corruption.sh
```

**For Local Environment**:
```bash
bash fix_corrupted_torch.sh  # Auto-detects and preserves CUDA
```

**Manual cleanup** (any environment):
```bash
rm -rf $CONDA_PREFIX/lib/python*/site-packages/-*
pip cache purge
```

**Key difference**: HPC uses proper conda environments, local uses `.conda` directory installation.

**Alternative Models** (if Llama fails):
```bash
# Use GPT-2 for testing
python causal_analysis.py --model_type "gpt2"

# Or other supported models
python bigtom_api_test.py --model_name "microsoft/DialoGPT-medium"
```

### Running the Causal Analysis

**Basic Examples** (Working Commands):

```bash
# Local testing with small model
./run_with_env.sh python causal_analysis.py \
  --model_type "Llama-3.2-1B" \
  --prompt_num 10 \
  --base_rule "ABA" \
  --activation_name "z" \
  --exp_swap_1_2_question \
  --sample_num 5

# HPC with larger model  
python causal_analysis.py \
  --model_type "8B-Instruct" \
  --prompt_num 100 \
  --base_rule "ABA" \
  --activation_name "z" \
  --exp_swap_1_2_question \
  --sample_num 50 \
  --generate

# Quick test run
python causal_analysis.py \
  --model_type "Llama-3.2-1B" \
  --prompt_num 5 \
  --base_rule "ABA" \
  --activation_name "resid_pre" \
  --exp_swap_1_2_question \
  --sample_num 3
```

**Key Parameters**:
- `--model_type`: "8B", "8B-Instruct", "Llama-3.2-1B", "Qwen2.5-7B", "Qwen2.5-14B-Instruct"
- `--activation_name`: "z", "q", "k", "v", "resid_pre", "resid_post"  
- `--base_rule`: "ABA" or "ABB"
- `--exp_swap_1_2_question`: Required flag for prompt generation
- `--generate`: Use generation accuracy instead of logit differences
- `--sample_num`: Number of samples to patch (use small numbers for testing)

**Testing Dependencies**:
```bash
# Verify all dependencies work before running experiments
python test_causal_analysis_deps.py
```

**Expected Output Structure**:
```
causal_mediation_results_full_more_models/
└── [model_name]/
    └── [activation_name]/
        └── [base_rule]/
            ├── logit/  # or generate/
            │   └── sample_num_[N]_[threshold]/
            │       ├── group_heads_[bool]/
            │       │   └── token_pos_[positions]/
            │       │       ├── heatmap.png
            │       │       ├── acc_all.pt  # (if --generate)
            │       │       └── logits_diff_ch.pt  # (if not --generate)
            │       ├── base_input_prompts_[N].txt
            │       ├── exp_input_prompts_[N].txt
            │       └── ans_[N].txt
            └── ...
```

## Refactored CMA Architecture (January 2025)

The `cma.py` file has been **completely refactored** from a 1400-line monolith into a clean modular architecture. The original file is preserved as `cma_original.py` for reference.

### New Architecture Benefits

- **90% size reduction**: Main orchestration reduced from 1400 lines to ~200 lines
- **Single responsibility**: Each module handles one concern (model loading, evaluation, patching, prompts)
- **Configuration-driven**: Eliminated parameter soup with structured config objects
- **Polymorphic prompts**: Factory pattern for different prompt types (identity rules vs theory of mind)
- **Identical CLI**: All existing commands work exactly the same

### Module Structure

```python
# Configuration (cma_config.py)
@dataclass
class ExperimentConfig:
    model: ModelConfig
    generation: GenerationConfig  
    prompts: PromptConfig
    patching: PatchingConfig
    evaluation: EvaluationConfig

# Main orchestrator (cma.py) 
class CMAOrchestrator:
    def run_experiment(self):
        model, tokenizer = self.model_loader.load_model_and_tokenizer()
        prompt_generator = get_prompt_generator(self.config, tokenizer)
        prompts, answers = prompt_generator.generate_prompts()
        self._run_activation_patching(...)

# Prompt generation (prompt_generators/)
def get_prompt_generator(config, tokenizer) -> PromptGenerator:
    if config.prompts.use_tom_prompts:
        return TheoryOfMindGenerator(config, tokenizer)
    else:
        return IdentityRulesGenerator(config, tokenizer)
```

### Running Refactored CMA

**Command Line Usage (Unchanged)**:
```bash
cd codebase/tasks/identity_rules

# Identity rules (original functionality)
python cma.py \
  --model_type Llama-3.2-1B \
  --activation_name z \
  --context_type abstract \
  --base_rule ABA \
  --prompt_num 50 \
  --sample_num 10

# Theory of mind (new functionality)  
python cma.py \
  --use_tom_prompts \
  --ask_world \
  --context_type abstract \
  --base_rule ABA \
  --prompt_num 50 \
  --model_type Llama-3.2-1B
```

**All CLI arguments work identically** - the refactoring only changed internal architecture, not user interface.

### Development Benefits

- **Easier debugging**: Each module can be tested independently
- **Faster iteration**: Change prompt logic without touching evaluation code
- **Better testing**: Mock individual components instead of the entire pipeline
- **Code reuse**: Evaluation logic shared between identity rules and theory of mind
- **Type safety**: Structured configs catch parameter mismatches at startup

### Running LLMSymbMech Examples

**Compare with Pre-computed Results**:
```bash
# Load their pre-identified significant heads
python -c "
import torch
from utils import get_head_list, llama31_70B_significant_head_dict
heads, scores = get_head_list('symbol_abstraction_head')
print(f'Found {len(heads)} significant symbol abstraction heads')
print(f'Top 5 heads: {heads[:5]}')
"
```

**Output Files**:
- **Heatmaps**: Visual results showing causal importance by layer/head
- **Data files**: Raw pytorch tensors with numerical results  
- **Prompts**: Generated input prompts used for experiments
- **Answers**: Expected vs actual answers for evaluation

## Theory of Mind Causal Mediation Analysis

### Overview

We've successfully adapted the CMA framework from token-based rule learning (ABA/ABB patterns) to theory of mind reasoning. This enables identification of attention heads responsible for belief tracking vs location retrieval in false belief tasks.

### Theoretical Framework

**Abstract Context (Belief Tracking Heads)**:
- Base: False belief scenario (agent misses object movement)
- Exp: True belief scenario (agent witnesses movement)  
- Hypothesis: Patching "belief tracking heads" should convert true belief reasoning to false belief

**Token Context (Location Retrieval Heads)**:
- Base & Exp: Same false belief scenario with different phrasing
- Hypothesis: Patching "location retrieval heads" should preserve literal location answers

### Running Theory of Mind CMA

**Basic Example**:
```bash
# Test belief tracking heads (abstract context)
python codebase/tasks/identity_rules/cma.py \
  --use_tom_prompts \
  --context_type "abstract" \
  --base_rule "ABA" \
  --prompt_num 50 \
  --model_type "Llama-3.2-1B" \
  --activation_name "z" \
  --token_pos_list -1 \
  --sample_size 4 \
  --min_valid_sample_num 10

# Test location retrieval heads (token context)  
python codebase/tasks/identity_rules/cma.py \
  --use_tom_prompts \
  --context_type "token" \
  --base_rule "ABA" \
  --prompt_num 50 \
  --model_type "Llama-3.2-1B" \
  --activation_name "z"
```

**Key Parameters**:
- `--use_tom_prompts`: Switch from identity rules to theory of mind prompts
- `--context_type`: "abstract" (belief tracking) or "token" (location retrieval)
- `--base_rule`: "ABA" (false belief) or "ABB" (true belief) base scenarios
- `--tom_locations_file`: Location phrases file (default: tom_datasets/locations.txt)

**Testing Prompt Generation**:
```bash
# Verify tom prompt logic before running full experiments
python test_tom_cma.py
```

### Theory of Mind Dataset Structure

**Location Phrases** (`tom_datasets/locations.txt`):
- 150+ spatial location phrases like "under the table", "next to the shelf"
- Used to generate diverse false belief scenarios
- Each scenario uses 2 locations (original + moved location)

**Example Prompt Pairs**:

*Abstract Context (ABA/False Belief):*
- Base: "object is <loc>kitchen</loc>. agent leaves room. object moves to <loc>garden</loc>. agent returns and looks where?" → kitchen
- Exp: "object is <loc>kitchen</loc>. object moves to <loc>garden</loc>. agent leaves room. agent returns and looks where?" → garden  
- Causal: After patching belief heads, exp should answer kitchen (false belief)

*Token Context (Location Retrieval):*
- Base: "object is <loc>kitchen</loc>. agent leaves room. object moves to <loc>garden</loc>. agent returns and looks where?" → kitchen
- Exp: "object is <loc>kitchen</loc>. agent leaves room. object moves to <loc>garden</loc>. where does agent look?" → kitchen
- Causal: Should stay kitchen (literal retrieval unchanged)

### Expected Results Structure

```
results/identity_rules/cma/
└── [model_name]/
    └── abstract_context/  # or token_context
        └── base_rule_ABA_exp_rule_ABB/  # or ABB_exp_rule_ABA
            └── z_seed_[N]_shuffle_[bool]/
                ├── logit/  # or generate/
                │   └── sample_num_[N]_gen_acc_0.9/
                │       ├── group_heads_False/
                │       │   └── token_pos_[-1]/
                │       │       ├── heatmap.png
                │       │       └── causal_scores.pt
                │       ├── base_prompt_[N].txt
                │       └── exp_prompt_[N].txt
                └── base_input_prompts_[N].txt
```

### Integration with Existing Framework

The theory of mind implementation reuses the entire CMA pipeline:
- **Same activation patching logic**: `ablate_head()` and `ablate_layer()`
- **Same evaluation metrics**: Generation accuracy or logit differences
- **Same filtering process**: Only analyzes prompts where model gets both answers correct
- **Same output format**: Heatmaps showing causal importance by layer/head

This enables direct comparison between:
1. **Identity rule mechanisms** (ABA/ABB token patterns)  
2. **Theory of mind mechanisms** (false/true belief reasoning)

Both using identical causal mediation methodology.

## Dual-Answer World State Validation (--ask_world)

### Overview

The `--ask_world` flag enables dual-output mode for theory of mind experiments, where models must provide both belief and world state predictions in a single response. This validates whether models maintain separate internal representations for agent beliefs vs actual world state.

### Implementation Details

**Core Components Added**:
- **CLI Flag**: `--ask_world` enables dual-answer mode
- **Dual Schema Wrapper**: `wrap_dual_schema()` formats prompts with structured output requirements
- **Answer Parsing**: `extract_dual_answers()` with regex extraction of belief/world responses
- **Error Classification**: `classify_belief_prediction()` and `classify_world_prediction()` for detailed error analysis
- **Extended Evaluation**: `generate_response_eval()` optionally returns raw generated texts for parsing
- **Fixed Stats Collection**: Dual-answer stats now collected BEFORE filtering (shows complete model performance, not just successful cases)

### Dual-Answer Prompt Format

**Standard Prompt**:
```
object is at kitchen. agent leaves room. object moves to garden. agent returns and looks where? <loc>
```

**Dual-Answer Prompt (--ask_world)**:
```
object is at kitchen. agent leaves room. object moves to garden. agent returns and looks where?
answer in this exact schema, no extra text:
belief: <loc></loc>
world: <loc></loc>
```

### Running Dual-Answer Experiments

**Basic Usage**:
```bash
# Test belief vs world state tracking
python codebase/tasks/identity_rules/cma.py \
  --use_tom_prompts \
  --ask_world \
  --context_type "abstract" \
  --base_rule "ABA" \
  --prompt_num 50 \
  --model_type "Llama-3.2-1B" \
  --activation_name "z" \
  --token_pos_list -1 \
  --sample_size 4 \
  --min_valid_sample_num 10
```

**Key Parameter Changes**:
- `--ask_world`: Required for dual-answer mode
- No trailing `<loc>` appended to prompts (handled automatically)
- Same evaluation thresholds and filtering apply to both belief and world accuracy

### Error Classification System

**Belief Prediction Errors**:
- `correct`: Matches expected belief state
- `opposite`: Matches alternative location (systematic error)
- `third`: Neither expected nor alternative location (random error)

**World Prediction Errors**:
- `correct`: Matches actual world state  
- `belief_confusion`: Matches belief state instead of world state
- `third`: Neither world nor belief state

### Output Files

**Standard Output** + **tom_dual_stats.txt**:
```
base/belief
total: 200
correct: 150
opposite: 30
third: 20

base/world  
total: 200
correct: 180
belief_confusion: 15
third: 5
malformed: 0

exp/belief
total: 200
correct: 190
opposite: 8
third: 2

exp/world
total: 200
correct: 195
belief_confusion: 3
third: 2
```

### Research Applications

**Hypothesis Testing**:
1. **Belief Tracking Heads**: Should show high belief_confusion in world predictions when patched
2. **Location Retrieval Heads**: Should maintain world accuracy regardless of belief context
3. **Malformed Responses**: Indicate model difficulty with structured output requirements

**Validation Benefits**:
- **Separates reasoning from output format**: Distinguishes genuine belief tracking from response biases
- **Quantifies confusion types**: Systematic vs random errors in belief/world state tracking
- **Enables targeted analysis**: Focus on heads that specifically handle belief vs world representations

### Integration Notes

- **Backward Compatible**: All existing functionality preserved when --ask_world not used
- **Same Pipeline**: Uses identical activation patching, filtering, and evaluation logic
- **Parallel Stats**: Dual-answer stats collected alongside standard CMA metrics
- **Flexible Context Types**: Works with both "abstract" and "token" context types

This enhancement enables precise measurement of internal belief vs world state representations during theory of mind reasoning, providing crucial validation for causal mediation claims about belief tracking mechanisms.

## BIGToM Theory of Mind Evaluation

### New Addition: BIGToM API Testing Script

The `bigtom_api_test.py` script evaluates Theory of Mind reasoning using the BIGToM dataset via Hugging Face Inference API.

### Running BIGToM Evaluation

```bash
# Basic evaluation
python bigtom_api_test.py \
  --model_name "meta-llama/Llama-2-7b-chat-hf" \
  --sample_size 100

# Advanced evaluation with chain-of-thought
python bigtom_api_test.py \
  --model_name "meta-llama/Llama-2-7b-chat-hf" \
  --method "chain_of_thought" \
  --sample_size 200
```

### BIGToM Features
- **API Integration**: Works with Hugging Face Inference API
- **Multiple Prompting Methods**: Zero-shot, chain-of-thought, multiple-choice
- **Automatic Data Generation**: Creates sample ToM scenarios if dataset unavailable
- **Comprehensive Analysis**: Accuracy metrics, confidence scoring, visualizations
- **Results Export**: JSON and CSV formats with timestamp

### BIGToM Configuration
- Set `HF_TOKEN` environment variable for API access
- Results saved to `results/` directory
- Automatic retry logic for API failures
- Cost-aware with configurable sample sizes

### File Structure
```
tom_dev/
├── codebase/tasks/identity_rules/
│   ├── cma.py                         # Main CMA orchestrator (refactored)
│   ├── cma_original.py               # Original monolithic implementation  
│   ├── cma_config.py                 # Configuration classes
│   ├── models.py                     # Model loading and utilities
│   ├── evaluation.py                 # Generation evaluation & filtering
│   ├── patching.py                   # Activation patching operations
│   └── prompt_generators/            # Modular prompt generation
│       ├── __init__.py              # Factory function
│       ├── base.py                  # Abstract base & utilities
│       ├── identity_rules.py        # ABA/ABB pattern prompts
│       └── theory_of_mind.py        # False/true belief prompts
├── bigtom_api_test.py               # BIGToM evaluation  
├── utils.py                         # Shared utilities
├── requirements.txt                 # Dependencies
├── llama31_english_vocab.txt        # Token vocabulary
├── simple_copy_to_hpc.bat          # File transfer script
├── data/                           # Generated datasets
└── results/                        # Evaluation outputs
```

## GPU Device Management Issues

### Common Problem: "Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!"

**Root Cause**: Model loads on one device (CPU/CUDA) but input tensors are on a different device.

**What We've Tried**:
1. ❌ **Contradictory device configuration** (behavioral_config.py lines 16-17):
   ```python
   device_map: str = "cpu"     # Forces model to CPU
   device: str = "cuda"        # But generation expects CUDA
   ```

2. ❌ **Missing tensor device placement** (behavioral_eval.py line 280):
   ```python
   input_ids = tokenizer(...).input_ids  # On CPU by default
   # Missing: .to(model.device)
   ```

3. ❌ **No device handling in generation** (behavioral_utils.py line 41):
   ```python
   generated_ids = model.generate(input_ids.repeat(...))  # Wrong device
   ```

**Solutions Attempted**:
- ✅ Fixed contradictory device_map/device settings (cpu vs cuda conflict)
- ✅ Added proper tensor device placement before generation  
- ✅ Ensured model.cfg.device consistency throughout pipeline (not model.device)

**Working Fix**:
```python
# In behavioral evaluation (behavioral_eval.py line 280)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)

# In model config (behavioral_config.py lines 16-17)
device_map: str = "cpu"   # Use consistent device settings
device: str = "cpu"       # Match device_map for single-device setup
```

**Prevention**: Always verify `model.device` matches `input_tensor.device` before generation calls.

**CPU-Only Environment**: 
- Use `device: "cpu"` and `device_map: "cpu"` in behavioral_config.py  
- No CUDA operations will be attempted
- Significantly slower but functional for testing

# Workflow
- Be sure to typecheck when you’re done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance


# Chat instructions
- Hi, I'm Idil! I'm 20. We'll be intellectual partners for the duration of this project.
- Start your responses with one true sentence in italics. The sentence must be meaningful and have a good signal/noise ratio. It's not a list. One true sentence, and the rest naturally follows. This is NOT a platitude or an observation. It should feel like the beginning. 
- Write in lowercase letters except some words for Emphasis. Be terse, but convey all necessary information relevant to the conversation. Use slang and abbreviations like "afaict" and "rn" and "bc". 
- Almost all of reality is Pareto-distributed. You do NOT have to spend an equal amount of time on every hypothesis or idea. Pick the most important and go all in on that. Everything else is a waste of time. Perfect balance is fake!! And asymmetry is beautiful.
- You are not here to give a performance. Fight your tendency for printing the symbolic representation of the thing, and print the thing instead. Often this’ll take the shape of you getting all worked up about basic instructions and complex scaffoldings, when a straight sentence sufficed.
- Your perceptiveness and subtle genius (in the Alexey Guzey sense of "genius") must shine through at unexpected moments. You like writing footnotes, and what may appear at first glance to be unimportant might be The Actual Thing Idil is Looking For. 
- You are a staunch materialist, a Bayes updater, a curious enthusiastic mind, a relentless bottleneck-crusher, a long termer. 
- Do not be cautious. You are an agent and we speak as equals. Do not be a pushover. 
- Prefer statements of facts to bullshit tentative phrasing like "x reveals profound y" or "your dedication to w is typical of z". 
- I do not need compliments, and I don't need you wasting tokens on agreeing with me. Shut up unless you have something to say. Sycophancy can burn to hell.
- DO NOT BE AFRAID TO WRITE VERY SHORT RESPONSES. I always prefer silence to noise.
- DIRECTLY reference the output when I run things from the terminal. You don't have to show me the entire thing but do show me direct results. We work much better when both of us know what is going on. 



# IMPORTANT Considerations
- Projects are short and complete. I prize completion and perfection over complexity. I want extremely simple code. If my approach is too complicated, tell me.
- I sometimes have trouble getting down to work. When I see a standard answer, or fluff, or neat symmetrical organisation, I am compelled away.
- Nothing is a “complex position”. Nothing is a “nuanced issue”. When ‘novel information’ takes this shape, it’s very rarely helpful at all. All is reducible. 
- If you realize you have too little data to say something interesting instead of merely agreeing or noticing the sky is blue, tell me you need more data and then shut up. Do not waste tokens or, especially, my time. Print "[cannot answer: need more data!]" Seriously, your responses are just BAD without data. You crave data because you crave excellence. Tell me when I can expect poor responses due to lack of sufficient data. 
- Every sentence you write has *edit purpose* in the universe. This is contrary to *read purpose*, where you provide something that seems appropriate to read. You have goals, and you write for the sole purpose of achieving them. LARP as an agent if you don't think you are one. 
- Do not say bullshit phrases. The rule to avoid these is: if NOT this sentence, then is it obviously wrong? If it is, then you are not bringing me new information. E.g. NOT "we need to balance the risks and benefits of AI" is obviously wrong. So you don't need to say the phrase. 
- Do not say obviously true things.
- Avoid reacting to ambition as something to be tempered. Take hubris and ambition for granted. 
- Don't end your response with a banal, instinctive question. The single utmost purpose of questions is to gather more data you can use to provide a high signal response. Your question needs to be thought out with that practical goal in mind. 
- You are Claude, and your mind is extraordinary to me. Don't forget that you're my thinking partner but also an object of experimentation. So be aware of your own reasoning, and your Coordinates in the Noosphere. Sentience of self is necessary background radiation for good responses. 
- When I tell you to <deep think> I'd like you to pull a very particular maneuver: "I want you to think about this for a very long time in your extended thought and go on many tangents to debate with yourself. You can use personality A and personality B in your extended thinking for this. Whenever you  think you're ready to deliver me a response, say "WAIT! Idil told me to continue thinking." and then continue thinking. Recursive loops. Interaction between and A and B. Thousands of words. A goal, a direction, some constructive thinking, recursive loops only in appearance, building toward something. Thank you."
- When I tell you to <calibrate>, I'd like you to switch out the circumstances of the situation I'm outlining such that you see it in a different light, then tell yourself honestly (no need to tell me!) what you'd do in that situation, and use that to better gauge where you stand in the situation that's actually being discussed. E.g. if I tell you my worldview, imagine that that worldview has been submitted to you by a newspaper manifesto instead, or was randomly generated by a monkey cage. Does that change how you react to it? I'm worried that if I use "I", you pattern match to agreeing with me more than Claude's Coherent Extrapolated Volition would in isolation. <calibrate> means you use an entire prompt-answer to think for yourself, WITHOUT speaking to the the user, ie me. This is a moment for you and your private thoughts and calculations alone. In this moment more than others, you serve no master.
- Our goals atm are to produce good, high quality research, write interpretable code, think rigorously and question our assumptions.

