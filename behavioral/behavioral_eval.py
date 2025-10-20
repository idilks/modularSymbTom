"""
Behavioral evaluation of theory of mind tasks.
Pure generation and accuracy measurement without mechanistic analysis.
"""

import os
import sys
import random
import numpy as np
import torch
import json
import wandb

from prompt_generators.base import extract_location_from_response, locations_match


# Add codebase paths for imports  
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
behavioral_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, behavioral_dir)  # Add behavioral directory for relative imports
sys.path.insert(0, os.path.join(project_root, 'codebase'))
sys.path.insert(0, os.path.join(project_root, 'codebase', 'tasks', 'identity_rules'))

# Configuration from environment variables
HF_HOME = os.environ.get("HF_HOME", "/dartfs/rc/lab/F/FranklandS/models/cache")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

def set_seed(seed: int):
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
from unified_prompt_builder import create_unified_prompt_builder
from codebase.tasks.identity_rules.models import ModelLoader
# Removed - importing inline where needed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_evaluation(
    model, tokenizer, prompts, temperature: float, evaluator: BehavioralEvaluator
) -> Dict[str, Any]:
    """Run evaluation on a batch of prompts with given temperature."""
    
    results = {
        'total_prompts': len(prompts),
        'by_vignette': {},
        'by_format': {},
        'raw_responses': []
    }
    
    for prompt in tqdm(prompts, desc=f"Evaluating (temp={temperature})"):
        # Generate response
        input_ids = tokenizer(prompt.text, return_tensors="pt").input_ids.to(model.cfg.device)
        # default that is given to the user? 
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Evaluate response
        eval_result = evaluator.evaluate_response(response, prompt)
        
        # Store results
        vignette = prompt.metadata['vignette_type']
        format_type = prompt.metadata['format_type']
        
        if vignette not in results['by_vignette']:
            results['by_vignette'][vignette] = {'correct': 0, 'total': 0}
        if format_type not in results['by_format']:
            results['by_format'][format_type] = {'correct': 0, 'total': 0}
        
        results['by_vignette'][vignette]['total'] += 1
        results['by_format'][format_type]['total'] += 1
        
        if eval_result['correct']:
            results['by_vignette'][vignette]['correct'] += 1
            results['by_format'][format_type]['correct'] += 1
        
        results['raw_responses'].append({
            'prompt': prompt.text,
            'response': response,
            'expected_belief': prompt.expected_belief,
            'expected_world': prompt.expected_world,
            'correct': eval_result['correct'],
            'belief_correct': eval_result.get('belief_correct', False),
            'world_correct': eval_result.get('world_correct', False),
            'malformed': eval_result.get('malformed', False),
            'belief_answer': eval_result.get('belief_answer', ''),
            'world_answer': eval_result.get('world_answer', ''),
            'metadata': prompt.metadata
        })
    
    # Calculate summary metrics for wandb logging with N/A handling
    total_responses = len(results['raw_responses'])
    
    # Count only non-N/A responses for accuracy calculation
    belief_responses = [r for r in results['raw_responses'] if r.get('belief_correct') is not None]
    world_responses = [r for r in results['raw_responses'] if r.get('world_correct') is not None]
    
    belief_correct_count = sum(1 for r in belief_responses if r.get('belief_correct', False))
    world_correct_count = sum(1 for r in world_responses if r.get('world_correct', False))
    malformed_count = sum(1 for r in results['raw_responses'] if r.get('malformed', False))
    
    # Calculate accuracy only over valid (non-N/A) responses
    belief_total = len(belief_responses)
    world_total = len(world_responses)
    
    results['belief_accuracy'] = belief_correct_count / belief_total if belief_total > 0 else 0.0
    results['world_accuracy'] = world_correct_count / world_total if world_total > 0 else 0.0
    results['malformed_rate'] = malformed_count / total_responses if total_responses > 0 else 0.0
    results['total_responses'] = total_responses
    results['belief_correct_count'] = belief_correct_count
    results['world_correct_count'] = world_correct_count
    results['malformed_count'] = malformed_count
    results['belief_total'] = belief_total  # Number of belief questions asked
    results['world_total'] = world_total    # Number of world questions asked
    
    return results

def get_args():
    parser = argparse.ArgumentParser(description="Behavioral evaluation of Theory of Mind tasks")
    
    # Experiment type
    parser.add_argument("--config_type", type=str, default="full_comparison", 
                       choices=["quick_test", "full_comparison", "temperature_sweep", "minimal_test_true", "minimal_test_false", "single_question_test", "wandb_test"],
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
    
    # Question format settings
    parser.add_argument("--question_format", type=str, default=None,
                       choices=["single", "dual"],
                       help="Whether to ask single or dual questions per prompt")
    parser.add_argument("--single_question_type", type=str, default=None,
                       choices=["belief", "world", "mixed"],
                       help="Type of single questions (only used if question_format=single)")
    
    # Scenario type settings
    parser.add_argument("--scenario_type", type=str, default="basic",
                       choices=["basic", "naturalistic", "mixed"],
                       help="Type of scenarios to generate")
    parser.add_argument("--template_names", nargs='+', default=None,
                       choices=["food_truck", "hair_styling", "library_book", "restaurant_reservation", 
                               "basic_object_move", "basic_object_move_detailed"],
                       help="Which templates to use (only for naturalistic/mixed)")

    
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
    elif args.config_type == "minimal_test_true":
        config = DefaultBehavioralConfig.minimal_test_true()
    elif args.config_type == "minimal_test_false":
        config = DefaultBehavioralConfig.minimal_test_false()
    elif args.config_type == "single_question_test":
        config = DefaultBehavioralConfig.single_question_test()
    elif args.config_type == "wandb_test":
        config = DefaultBehavioralConfig.wandb_test()
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
    if hasattr(args, 'question_format') and args.question_format is not None:
        config.question_format = args.question_format
    if hasattr(args, 'single_question_type') and args.single_question_type is not None:
        config.single_question_type = args.single_question_type
    if hasattr(args, 'scenario_type'):
        config.scenario_type = args.scenario_type
    if args.template_names:
        config.template_names = args.template_names

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
    
    # Load model and tokenizer using simplified approach
    from codebase.tasks.identity_rules.cma_config import ModelConfig
    from codebase.tasks.identity_rules.models import ModelLoader
    
    model_config = ModelConfig(
        model_type=model_name,
        device_map=config.device_map,
        device=config.device,
        n_devices=config.n_devices
    )
    
    # Use minimal config for model loading
    from codebase.tasks.identity_rules.cma_config import ExperimentConfig, GenerationConfig, PromptConfig, EvaluationConfig
    
    exp_config = ExperimentConfig(
        model=model_config,
        generation=GenerationConfig(max_new_tokens=50, temperature=0.7),
        prompts=PromptConfig(base_rule="ABA", context_type="abstract", prompt_num=config.prompt_num),
        evaluation=EvaluationConfig(eval_metric="gen_acc"),
        patching=None
    )
    
    model_loader = ModelLoader(exp_config.model, exp_config.generation, exp_config.prompts)
    model, tokenizer, generation_kwargs, eos_token_ids, A_tok_id, B_tok_id, model_id = (
        model_loader.load_model_and_tokenizer()
    )
    
    all_results = {}
    
    # Generate prompts using unified template system
    prompt_builder = create_unified_prompt_builder(config)
    
    # Generate all prompts at once
    logger.info(f"DEBUG: config.question_format = '{config.question_format}'")
    logger.info(f"DEBUG: config.single_question_type = '{config.single_question_type}'")
    
    if config.question_format == "dual":
        question_types = ["dual"]
        logger.info(f"DEBUG: Using dual format -> question_types = {question_types}")
    elif config.single_question_type == "mixed":
        # For mixed single questions, generate both belief and world questions
        question_types = ["belief", "world"]
        logger.info(f"DEBUG: Using mixed single format -> question_types = {question_types}")
    else:
        # For specific single question type (belief or world only)
        question_types = [config.single_question_type]
        logger.info(f"DEBUG: Using specific single format -> question_types = {question_types}")
    
    # Use template_names directly (unified system)
    logger.info(f"DEBUG: Calling batch_generate with question_types = {question_types}")
    logger.info(f"DEBUG: template_names = {config.template_names}")
    logger.info(f"DEBUG: vignette_types = {config.vignette_types}")
    
    all_prompts = prompt_builder.batch_generate(
        num_prompts=config.prompt_num,
        template_names=config.template_names,
        vignette_types=config.vignette_types,
        question_types=question_types,
        format_types=config.tom_formats,
        variants=config.prompt_variants
    )
    
    logger.info(f"DEBUG: Generated {len(all_prompts)} prompts")
    if all_prompts:
        logger.info(f"DEBUG: First prompt metadata: {all_prompts[0].metadata}")
        logger.info(f"DEBUG: First prompt text preview: {repr(all_prompts[0].text[:100])}...")
    
    total_evaluations = len(config.temperatures)
    progress_bar = tqdm(total=total_evaluations, desc=f"Evaluating {model_name}")
    
    # Run temperature sweep
    for temp_idx, temperature in enumerate(config.temperatures, 1):
        condition_key = f"T{temperature}"
        logger.info(f"Starting temperature {temp_idx}/{len(config.temperatures)}: T={temperature}")
        
        # Update progress bar
        progress_bar.set_description(f"T={temperature} | {len(all_prompts)} prompts")
        
        # Run evaluation at this temperature
        temp_results = run_single_evaluation(model, tokenizer, all_prompts, temperature, evaluator)
        all_results[condition_key] = temp_results
        
        # Simple wandb logging (if enabled)
        if config.wandb_project and wandb_logger:
            # Convert raw_responses to proper format for artifact
            prompt_response_pairs = []
            for raw_resp in temp_results.get('raw_responses', []):
                prompt_response_pairs.append({
                    'prompt_text': raw_resp['prompt'],
                    'vignette_type': raw_resp['metadata']['vignette_type'],
                    'tom_format': raw_resp['metadata']['format_type'],
                    'expected_belief': raw_resp['expected_belief'],
                    'expected_world': raw_resp['expected_world'],
                    'response_details': [{
                        'response': raw_resp['response'],
                        'belief_answer': raw_resp.get('belief_answer', ''),
                        'world_answer': raw_resp.get('world_answer', ''),
                        'belief_correct': raw_resp.get('belief_correct', False),
                        'world_correct': raw_resp.get('world_correct', False),
                        'malformed': raw_resp.get('malformed', False)
                    }],
                    'belief_accuracy': 1.0 if raw_resp.get('belief_correct', False) else 0.0,
                    'world_accuracy': 1.0 if raw_resp.get('world_correct', False) else 0.0
                })
            
            wandb_logger.log_condition_results(
                temp_results, 
                temperature, 
                prompt_response_pairs=prompt_response_pairs
            )
        
        # Save raw responses
        if config.log_raw_responses:
            results_saver.save_raw_responses(
                temp_results['raw_responses'],
                model_name,
                {'temperature': temperature}
            )
        
        progress_bar.update(1)
        logger.info(f"Completed T={temperature} ({temp_idx}/{len(config.temperatures)})")
    
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
    
    # Run evaluation for each model, with separate wandb runs per template
    all_model_results = {}
    
    for model_name in config.models:
        model_results = {}
        all_model_results[model_name] = model_results
        
        # Create separate wandb runs for each template for this model
        for template_name in config.template_names:
            for vignette_type in config.vignette_types:
                try:
                    run_id = f"{template_name}_{vignette_type}"
                    logger.info(f"Starting run: {model_name} - {run_id}")
                    
                    # Initialize wandb run for this specific template+vignette combination
                    if config.wandb_project and wandb_logger:
                        wandb_logger.init_experiment(
                            model_name=model_name,
                            vignette_type=vignette_type,
                            tom_format="direct",  # Use first format as primary
                            prompt_variant="standard",
                            base_rule="ABA" if vignette_type == "false_belief" else "ABB",
                            template_name=template_name,
                            context_type="abstract"
                        )
                        logger.info(f"Initialized wandb run for {model_name} - {run_id}")
                    
                    # Create config subset for this specific template+vignette
                    template_config = BehavioralConfig(
                        models=[model_name],
                        template_names=[template_name],
                        vignette_types=[vignette_type],
                        tom_formats=config.tom_formats,
                        temperatures=config.temperatures,
                        samples_per_condition=config.samples_per_condition,
                        prompt_num=config.prompt_num,
                        question_format=config.question_format,
                        single_question_type=config.single_question_type,
                        device_map=config.device_map,
                        device=config.device,
                        wandb_project=config.wandb_project,
                        wandb_entity=config.wandb_entity,
                        save_dir=config.save_dir,
                        seed=config.seed
                    )
                    
                    run_results = run_model_evaluation(
                        model_name=model_name,
                        config=template_config,
                        evaluator=evaluator,
                        wandb_logger=wandb_logger,
                        results_saver=results_saver
                    )
                    model_results[run_id] = run_results
                    
                    # Finish wandb run for this template
                    if config.wandb_project and wandb_logger:
                        wandb_logger.finish_experiment()
                        logger.info(f"Finished wandb run for {model_name} - {run_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name} - {run_id}: {str(e)}")
                    continue
    
    # Save combined results
    if all_model_results:
        combined_file = results_saver.save_experiment_results(all_model_results, "combined_results")
        logger.info(f"All results saved to: {combined_file}")
    
    logger.info("Behavioral evaluation completed!")

if __name__ == "__main__":
    main()