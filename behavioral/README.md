# Behavioral Evaluation of Theory of Mind Tasks

Pure behavioral assessment of model performance on theory of mind reasoning without mechanistic analysis.

## Quick Start

```bash
# Navigate to tom_dev root directory
cd /path/to/tom_dev

# Test basic functionality
.conda/python.exe behavioral_evaluation/test_basic.py

# Run quick test (small models, few samples)
.conda/python.exe behavioral_evaluation/behavioral_eval.py --config_type quick_test --no_wandb

# Run full comparison (target models with temperature sweeps)  
.conda/python.exe behavioral_evaluation/behavioral_eval.py --config_type full_comparison

# Custom evaluation
.conda/python.exe behavioral_evaluation/behavioral_eval.py \
  --models "meta-llama/Llama-3.1-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" \
  --temperatures 0.5 1.0 1.5 \
  --samples_per_condition 50
```

## Configuration Types

- **quick_test**: Single model, minimal samples for testing
- **full_comparison**: Target models with full temperature sweeps
- **temperature_sweep**: Detailed temperature analysis for single model

## Output

- **Local**: Results saved to `results/tom_performance/`
- **Wandb**: Interactive plots and experiment tracking
- **Raw responses**: Optional detailed response logging

## Architecture

- `behavioral_config.py`: Configuration classes and presets
- `behavioral_utils.py`: Evaluator, logger, and results handling
- `behavioral_eval.py`: Main orchestration script
- `test_basic.py`: Basic functionality test

## Key Features

- ✅ Pure behavioral evaluation (no caching/patching)
- ✅ Wandb integration with rich metadata
- ✅ Temperature sweeps and format comparisons  
- ✅ Dual-answer parsing (belief vs world state)
- ✅ Modular reuse of existing prompt generators
- ✅ Graceful fallbacks and error handling