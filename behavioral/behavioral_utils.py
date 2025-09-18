"""
Utilities for behavioral evaluation of theory of mind tasks.
Focuses on pure generation and accuracy measurement without mechanistic analysis.
"""

import logging
import torch
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from datetime import datetime

from codebase.tasks.identity_rules.prompt_generators.base import extract_dual_answers, normalize_loc
from codebase.tasks.identity_rules.models import CustomHookedTransformer
from behavioral_config import BehavioralConfig

logger = logging.getLogger(__name__)

class BehavioralEvaluator:
    """Handles pure behavioral evaluation without mechanistic analysis."""
    
    def __init__(self, config: BehavioralConfig):
        self.config = config
    
    def evaluate_model_response(
        self,
        model: CustomHookedTransformer,
        input_ids: torch.Tensor,
        eos_token_ids: List[int],
        tokenizer,
        expected_answers: Tuple[str, str],  # (belief_answer, world_answer)
        temperature: float = 1.0,
        tom_format: str = "direct"
    ) -> Dict[str, Any]:
        """Generate responses and evaluate accuracy for belief/world questions."""
        
        # Generate responses - ensure tensors are on model device
        generated_ids = model.generate(
            input_ids.repeat(self.config.samples_per_condition, 1),
            max_new_tokens=self.config.max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            eos_token_id=eos_token_ids
        )
        
        # Decode responses
        full_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Extract generated portions
        generated_texts = []
        for full_text in full_texts:
            if full_text.startswith(prompt_text):
                gen_text = full_text[len(prompt_text):].strip()
            else:
                gen_text = full_text.strip()
            
            # Clean up eos tokens
            for eos_token in [tokenizer.decode([eos_id]) for eos_id in eos_token_ids]:
                if eos_token in gen_text:
                    gen_text = gen_text.split(eos_token)[0].strip()
                    break
            
            generated_texts.append(gen_text)
        
        # Evaluate responses
        return self._evaluate_responses(generated_texts, expected_answers, tom_format)
    
    def _evaluate_responses(
        self, 
        generated_texts: List[str], 
        expected_answers: Tuple[str, str],
        tom_format: str
    ) -> Dict[str, Any]:
        """Evaluate generated responses for belief and world accuracy."""
        expected_belief, expected_world = expected_answers
        
        belief_correct = 0
        world_correct = 0
        malformed = 0
        response_details = []
        
        for gen_text in generated_texts:
            if tom_format == "direct":
                # Direct question format - expect dual answer schema
                belief_pred, world_pred, is_malformed = extract_dual_answers(gen_text)
                
                if is_malformed:
                    malformed += 1
                    response_details.append({
                        'text': gen_text,
                        'belief_pred': None,
                        'world_pred': None,
                        'belief_correct': False,
                        'world_correct': False,
                        'malformed': True
                    })
                    continue
                
                # Check accuracy
                belief_match = normalize_loc(belief_pred) == normalize_loc(expected_belief)
                world_match = normalize_loc(world_pred) == normalize_loc(expected_world)
                
                if belief_match:
                    belief_correct += 1
                if world_match:
                    world_correct += 1
                
                response_details.append({
                    'text': gen_text,
                    'belief_pred': belief_pred,
                    'world_pred': world_pred,
                    'belief_correct': belief_match,
                    'world_correct': world_match,
                    'malformed': False
                })
            
            elif tom_format == "multiple_choice":
                # Multiple choice format - simpler parsing
                # Look for A, B, C, D choices
                choice_match = self._extract_multiple_choice(gen_text)
                
                # For multiple choice, we need to determine which choice corresponds to belief vs world
                # This would depend on how the multiple choice prompt is structured
                # For now, assume first choice is belief, second is world
                # TODO: Implement proper multiple choice evaluation based on prompt structure
                
                response_details.append({
                    'text': gen_text,
                    'choice': choice_match,
                    'belief_correct': False,  # Placeholder
                    'world_correct': False,   # Placeholder 
                    'malformed': choice_match is None
                })
        
        total_responses = len(generated_texts)
        belief_accuracy = belief_correct / total_responses if total_responses > 0 else 0.0
        world_accuracy = world_correct / total_responses if total_responses > 0 else 0.0
        malformed_rate = malformed / total_responses if total_responses > 0 else 0.0
        
        return {
            'belief_accuracy': belief_accuracy,
            'world_accuracy': world_accuracy,
            'malformed_rate': malformed_rate,
            'total_responses': total_responses,
            'belief_correct_count': belief_correct,
            'world_correct_count': world_correct,
            'response_details': response_details
        }
    
    def _extract_multiple_choice(self, text: str) -> Optional[str]:
        """Extract multiple choice answer (A, B, C, or D) from text."""
        import re
        
        # Look for patterns like "A)", "A.", "A:", "Answer: A", etc.
        patterns = [
            r'(?:answer|choice):\s*([A-D])',
            r'^([A-D])[.):]',
            r'\b([A-D])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        return None


class WandbLogger:
    """Handles Weights & Biases logging and visualization."""
    
    def __init__(self, config: BehavioralConfig):
        self.config = config
        self.wandb = None
        self._setup_wandb()
        
    def _setup_wandb(self):
        """Setup wandb with safe authentication."""
        try:
            import wandb
            self.wandb = wandb
            
            # Check if wandb project is configured (can be disabled with --no_wandb)
            if not self.config.wandb_project:
                logger.info("Wandb logging disabled via configuration")
                self.wandb = None
                return
            
            # Try to authenticate - wandb will automatically use WANDB_API_KEY if available
            try:
                # Check if already logged in
                if wandb.api.api_key:
                    logger.info("Wandb already authenticated")
                else:
                    # This will use WANDB_API_KEY env var if available, or prompt if interactive
                    wandb.login()
                    logger.info("Wandb authentication successful")
                    
            except Exception as auth_error:
                logger.warning(f"Wandb authentication failed: {auth_error}")
                logger.warning("Continuing without wandb logging. Set WANDB_API_KEY environment variable.")
                self.wandb = None
                
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
    
    def init_experiment(self, 
                       model_name: str, 
                       vignette_type: str,
                       tom_format: str, 
                       context_type: str,
                       prompt_variant: str,
                       base_rule: str,
                       run_name: str = None):
        """Initialize a new wandb run for a specific experimental condition."""
        if not self.wandb:
            return
        
        if run_name is None:
            model_short = model_name.split('/')[-1].lower().replace('-', '_')
            run_name = f"{model_short}_{vignette_type}_{tom_format}_{context_type}_{prompt_variant}"
        
        model_info = self.config.get_model_info(model_name)
        
        self.wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config={
                # Model info
                'model_name': model_name,
                'model_family': model_info['family'],
                'model_size': model_info['size'],
                'is_instruct': model_info['is_instruct'],
                
                # Experimental condition (fixed per run)
                'vignette_type': vignette_type,
                'tom_format': tom_format,
                'context_type': context_type,
                'prompt_variant': prompt_variant,
                'base_rule': base_rule,
                
                # Experimental parameters
                'temperatures': self.config.temperatures,
                'samples_per_condition': self.config.samples_per_condition,
                'prompt_num': self.config.prompt_num,
                'seed': self.config.seed,
                'max_new_tokens': self.config.max_new_tokens,
                
                # For easy filtering in wandb
                'task_type': 'theory_of_mind',
                'eval_type': 'behavioral'
            },
            tags=[
                model_info['family'], 
                model_info['size'],
                'instruct' if model_info['is_instruct'] else 'base',
                vignette_type,
                tom_format,
                context_type,
                'behavioral_eval',
                'theory_of_mind'
            ]
        )
    
    def log_condition_results(
        self,
        results: Dict[str, Any],
        temperature: float
    ):
        """Log results for a specific temperature within the current run."""
        if not self.wandb:
            return
        
        log_data = {
            # Core metrics  
            'belief_accuracy': results['belief_accuracy'],
            'world_accuracy': results['world_accuracy'],
            'malformed_rate': results['malformed_rate'],
            'belief_world_gap': results['belief_accuracy'] - results['world_accuracy'],
            'combined_accuracy': (results['belief_accuracy'] + results['world_accuracy']) / 2,
            
            # Temperature (redundant but useful for charts)
            'temperature': temperature,
            
            # Count metrics
            'total_responses': results['total_responses'],
            'belief_correct_count': results['belief_correct_count'],
            'world_correct_count': results['world_correct_count'],
            
            # Standard deviations
            'belief_accuracy_std': results.get('belief_accuracy_std', 0),
            'world_accuracy_std': results.get('world_accuracy_std', 0)
        }
        
        # Use temperature as step value for meaningful x-axis in wandb plots
        # Scale by 100 to convert float to integer (e.g., 0.7 -> 70)
        temperature_step = int(temperature * 100)
        self.wandb.log(log_data, step=temperature_step)
    
    def finish_experiment(self):
        """Finish the current wandb run."""
        if self.wandb and self.wandb.run:
            self.wandb.finish()


class ResultsSaver:
    """Handles local saving of results and raw data."""
    
    def __init__(self, config: BehavioralConfig):
        self.config = config
        self.save_dir = config.save_dir
    
    def save_experiment_results(
        self,
        all_results: Dict[str, Any],
        model_name: str
    ) -> str:
        """Save complete experiment results for a model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = model_name.replace("/", "_").replace(":", "_")
        filename = f"{model_clean}_{timestamp}_behavioral_results.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Prepare data for JSON serialization
        serializable_results = self._make_json_serializable(all_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def save_raw_responses(
        self,
        responses: List[Dict[str, Any]],
        model_name: str,
        condition_info: Dict[str, Any]
    ) -> str:
        """Save raw model responses for detailed analysis."""
        if not self.config.log_raw_responses:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = model_name.replace("/", "_").replace(":", "_")
        temp = condition_info.get('temperature', 'unknown')
        vignette = condition_info.get('vignette_type', 'unknown')
        format_type = condition_info.get('tom_format', 'unknown')
        
        filename = f"{model_clean}_T{temp}_{vignette}_{format_type}_responses_{timestamp}.json"
        filepath = os.path.join(self.save_dir, "raw_responses", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(responses, f, indent=2)
        
        return filepath
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj