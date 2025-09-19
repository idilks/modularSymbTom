"""
Behavioral evaluation of theory of mind tasks.
Pure generation and accuracy measurement without mechanistic analysis.
"""

import os
import sys

# Add codebase paths for imports  
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'codebase'))
sys.path.insert(0, os.path.join(project_root, 'codebase', 'tasks', 'identity_rules'))

# Configuration from environment variables
HF_HOME = os.environ.get("HF_HOME", "/dartfs/rc/lab/F/FranklandS/models/cache")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Set environment variables only if they have values
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

# Set wandb api key if not already set (slurm takes precedence)
if "WANDB_API_KEY" not in os.environ and WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

import logging
import argparse
from typing import Dict, Any, List
from tqdm import tqdm

from behavioral_config import BehavioralConfig, DefaultBehavioralConfig
from behavioral_utils import BehavioralEvaluator, WandbLogger, ResultsSaver
from behavioral_prompt_minimal import get_minimal_behavioral_generator
from codebase.tasks.identity_rules.models import ModelLoader
from codebase.tasks.identity_rules.cma_config import (
    ExperimentConfig, ModelConfig, GenerationConfig, 
    PromptConfig, PatchingConfig, EvaluationConfig
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_experiment_config(model_name: str, config: BehavioralConfig, **kwargs) -> ExperimentConfig:
    """Create proper ExperimentConfig from flat parameters."""
    
    model_config = ModelConfig(
        model_type=model_name,
        device_map=config.device_map,
        device=config.device, 
        n_devices=config.n_devices
    )
    
    generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=kwargs.get('temperature', 0.7),
        do_sample=True
    )
    
    prompt_config = PromptConfig(
        base_rule=kwargs.get('base_rule', 'ABA'),
        context_type=kwargs.get('context_type', 'abstract'),
        prompt_num=config.prompt_num,
        use_tom_prompts=kwargs.get('use_tom_prompts', True),
        ask_world=kwargs.get('ask_world', True),
        tom_format=kwargs.get('tom_format', 'direct'),
        prompt_variant=kwargs.get('prompt_variant', 'standard'),
        tom_locations_file=config.tom_locations_file
    )
    
    patching_config = PatchingConfig(
        activation_name='z',  # dummy value, not used in behavioral
        token_pos_list=[-1]
    )
    
    evaluation_config = EvaluationConfig(
        eval_metric='gen_acc',
        low_prob_threshold=0.6
    )
    
    return ExperimentConfig(
        model=model_config,
        generation=generation_config,
        prompts=prompt_config,
        patching=patching_config,
        evaluation=evaluation_config,
        seed=kwargs.get('seed', config.seed)
    )

def get_args():
    parser = argparse.ArgumentParser(description="Behavioral evaluation of Theory of Mind tasks")
    
    # Experiment type
    parser.add_argument("--config_type", type=str, default="full_comparison", 
                       choices=["quick_test", "full_comparison", "temperature_sweep"],
                       help="Predefined configuration type")
    parser.add_argument("--single_model", type=str, default=None,
                       help="Run temperature sweep on single model")
    
    # Model settings  
    parser.add_argument("--models", nargs='+', default=None,
                       help="List of models to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping strategy")
    
    # Experimental parameters
    parser.add_argument("--temperatures", nargs='+', type=float, default=None,
                       help="Temperatures to test")
    parser.add_argument("--samples_per_condition", type=int, default=5,
                       help="Number of samples per experimental condition")
    parser.add_argument("--prompt_num", type=int, default=20,
                       help="Number of prompts to generate")
    
    # Task settings
    parser.add_argument("--vignette_types", nargs='+', default=None,
                       choices=["false_belief", "true_belief", "token_variant"],
                       help="Types of vignettes to test")
    parser.add_argument("--tom_formats", nargs='+', default=None,
                       choices=["direct", "multiple_choice"],
                       help="Prompt formats to test")
    parser.add_argument("--context_types", nargs='+', default=None,
                       choices=["abstract", "token"],
                       help="Context types to test")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="tom-behavioral-eval",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Weights & Biases entity/username")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--save_dir", type=str, default="results/tom_performance",
                       help="Directory to save results")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    return parser.parse_args()

def create_config_from_args(args) -> BehavioralConfig:
    """Create behavioral config from command line arguments."""
    
    # Use predefined configs as base
    if args.config_type == "quick_test":
        config = DefaultBehavioralConfig.quick_test()
    elif args.config_type == "temperature_sweep" and args.single_model:
        config = DefaultBehavioralConfig.temperature_sweep(args.single_model)
    else:  # full_comparison
        config = DefaultBehavioralConfig.full_comparison()
    
    # Override with command line arguments if provided
    if args.models:
        config.models = args.models
    if args.temperatures:
        config.temperatures = args.temperatures
    if args.samples_per_condition:
        config.samples_per_condition = args.samples_per_condition
    if args.prompt_num:
        config.prompt_num = args.prompt_num
    if args.vignette_types:
        config.vignette_types = args.vignette_types
    if args.tom_formats:
        config.tom_formats = args.tom_formats
    if args.context_types:
        config.context_types = args.context_types
    
    # Device and logging settings (using defaults from BehavioralConfig)
    config.save_dir = args.save_dir
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    config.seed = args.seed
    
    # Disable wandb if requested
    if args.no_wandb:
        config.wandb_project = None
    
    return config

def run_model_evaluation(
    model_name: str,
    config: BehavioralConfig,
    evaluator: BehavioralEvaluator,
    wandb_logger: WandbLogger,
    results_saver: ResultsSaver
) -> Dict[str, Any]:
    """Run behavioral evaluation for a single model."""
    
    logger.info(f"Starting evaluation for model: {model_name}")
    
    # Load model and tokenizer once for all conditions
    dummy_exp_config = create_experiment_config(
        model_name=model_name,
        config=config,
        use_tom_prompts=True,
        ask_world=True
    )
    
    model_loader = ModelLoader(
        dummy_exp_config.model,
        dummy_exp_config.generation, 
        dummy_exp_config.prompts
    )
    
    model, tokenizer, generation_kwargs, eos_token_ids, A_tok_id, B_tok_id, model_id = (
        model_loader.load_model_and_tokenizer()
    )
    
    all_results = {}
    
    # Calculate total for progress tracking (conditions × temperatures)
    total_conditions = (len(config.vignette_types) * len(config.tom_formats) * 
                       len(config.context_types) * len(config.prompt_variants))
    total_evaluations = total_conditions * len(config.temperatures)
    
    progress_bar = tqdm(total=total_evaluations, desc=f"Evaluating {model_name}")
    condition_count = 0
    
    for vignette_type in config.vignette_types:
        for tom_format in config.tom_formats:
            for context_type in config.context_types:
                for prompt_variant in config.prompt_variants:
                    base_rule = "ABA" if vignette_type == "false_belief" else "ABB"
                    
                    # Initialize wandb for this specific experimental condition
                    wandb_logger.init_experiment(
                        model_name=model_name,
                        vignette_type=vignette_type,
                        tom_format=tom_format,
                        context_type=context_type,
                        prompt_variant=prompt_variant,
                        base_rule=base_rule
                    )
                    
                    # Generate prompts for this condition using behavioral generator
                    behavioral_config = create_experiment_config(
                        model_name=model_name,
                        config=config,
                        use_tom_prompts=True,
                        ask_world=True,
                        context_type=context_type,
                        base_rule=base_rule,
                        tom_format=tom_format,
                        prompt_variant=prompt_variant,
                        seed=config.seed
                    )
                    
                    # Set current experimental condition for generator
                    behavioral_config.vignette_types = [vignette_type]
                    behavioral_config.tom_formats = [tom_format]
                    behavioral_config.prompt_variants = [prompt_variant]
                    
                    prompt_generator = get_minimal_behavioral_generator(behavioral_config, tokenizer)
                    behavioral_prompts = prompt_generator.generate_prompts(config.prompt_num)
                    
                    condition_count += 1
                    condition_name = f"{vignette_type}_{tom_format}_{context_type}_{prompt_variant}"
                    logger.info(f"Starting condition {condition_count}/{total_conditions}: {condition_name}")
                    
                    # Run temperature sweep for this condition
                    for temp_idx, temperature in enumerate(config.temperatures, 1):
                        condition_key = f"{vignette_type}_{tom_format}_{context_type}_{prompt_variant}_T{temperature}"
                        
                        # Update progress bar with detailed description
                        progress_description = (f"{condition_name} | T={temperature} "
                                              f"({temp_idx}/{len(config.temperatures)}) | "
                                              f"{config.prompt_num} prompts × {config.samples_per_condition} samples")
                        progress_bar.set_description(progress_description)
                        
                        # Evaluate all prompts for this temperature
                        condition_results = {
                            'belief_accuracies': [],
                            'world_accuracies': [],
                            'malformed_rates': [],
                            'all_response_details': []
                        }
                        
                        for behavioral_prompt in behavioral_prompts:
                            # Use single prompt from minimal behavioral generator  
                            input_ids = tokenizer(behavioral_prompt.text, return_tensors="pt").input_ids.to(model.cfg.device)
                            
                            # Handle different answer formats from minimal generator
                            expected_answer = behavioral_prompt.expected_answer
                            if isinstance(expected_answer, tuple):
                                # Dual answer mode: (belief, world)
                                expected_answers = expected_answer
                            else:
                                # Single answer mode: duplicate for belief/world
                                expected_answers = (expected_answer, expected_answer)
                            
                            # Evaluate this prompt
                            results = evaluator.evaluate_model_response(
                                model=model,
                                input_ids=input_ids,
                                eos_token_ids=eos_token_ids,
                                tokenizer=tokenizer,
                                expected_answers=expected_answers,
                                temperature=temperature,
                                tom_format=behavioral_prompt.tom_format
                            )
                            
                            condition_results['belief_accuracies'].append(results['belief_accuracy'])
                            condition_results['world_accuracies'].append(results['world_accuracy'])
                            condition_results['malformed_rates'].append(results['malformed_rate'])
                            condition_results['all_response_details'].extend(results['response_details'])
                        
                        # Aggregate results for this temperature
                        import numpy as np
                        aggregated_results = {
                            'belief_accuracy': np.mean(condition_results['belief_accuracies']),
                            'world_accuracy': np.mean(condition_results['world_accuracies']),
                            'malformed_rate': np.mean(condition_results['malformed_rates']),
                            'belief_accuracy_std': np.std(condition_results['belief_accuracies']),
                            'world_accuracy_std': np.std(condition_results['world_accuracies']),
                            'total_responses': len(condition_results['all_response_details']),
                            'belief_correct_count': sum(1 for r in condition_results['all_response_details'] 
                                                      if r.get('belief_correct', False)),
                            'world_correct_count': sum(1 for r in condition_results['all_response_details'] 
                                                     if r.get('world_correct', False))
                        }
                        
                        all_results[condition_key] = aggregated_results
                        
                        # Log to wandb (temperature as step)
                        wandb_logger.log_condition_results(
                            results=aggregated_results,
                            temperature=temperature
                        )
                        
                        # Save raw responses if enabled
                        if config.log_raw_responses:
                            condition_info = {
                                'temperature': temperature,
                                'vignette_type': vignette_type,
                                'tom_format': tom_format,
                                'context_type': context_type,
                                'prompt_variant': prompt_variant
                            }
                            results_saver.save_raw_responses(
                                condition_results['all_response_details'],
                                model_name,
                                condition_info
                            )
                        
                        # Update progress after each temperature
                        progress_bar.update(1)
                        logger.info(f"Completed: {condition_name} T={temperature} "
                                   f"({temp_idx}/{len(config.temperatures)})")
                    
                    # Finish wandb run for this condition
                    wandb_logger.finish_experiment()
                    
                    logger.info(f"Completed all temperatures for condition: {condition_name}")
    
    progress_bar.close()
    
    # Save complete results for this model
    results_file = results_saver.save_experiment_results(all_results, model_name)
    
    logger.info(f"Completed evaluation for {model_name}")
    return all_results

def main():
    """Main entry point for behavioral evaluation."""
    args = get_args()
    config = create_config_from_args(args)
    
    # Set random seed
    set_seed(config.seed)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize utilities
    evaluator = BehavioralEvaluator(config)
    wandb_logger = WandbLogger(config)
    results_saver = ResultsSaver(config)
    
    logger.info(f"Starting behavioral evaluation with config: {config}")
    logger.info(f"Models to evaluate: {config.models}")
    logger.info(f"Experimental conditions: {len(config.vignette_types)} vignette types, "
               f"{len(config.tom_formats)} formats, {len(config.temperatures)} temperatures")
    
    # Run evaluation for each model
    all_model_results = {}
    
    for model_name in config.models:
        try:
            model_results = run_model_evaluation(
                model_name=model_name,
                config=config,
                evaluator=evaluator,
                wandb_logger=wandb_logger,
                results_saver=results_saver
            )
            all_model_results[model_name] = model_results
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {str(e)}")
            continue
    
    # Save combined results
    if all_model_results:
        combined_file = results_saver.save_experiment_results(all_model_results, "combined_results")
        logger.info(f"All results saved to: {combined_file}")
    
    logger.info("Behavioral evaluation completed!")

if __name__ == "__main__":
    main()