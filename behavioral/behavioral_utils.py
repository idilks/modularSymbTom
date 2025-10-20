"""
Utilities for behavioral evaluation of theory of mind tasks.
Focuses on pure generation and accuracy measurement without mechanistic analysis.
"""

import logging
import torch
import json
import os
import sys
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from datetime import datetime

from codebase.tasks.identity_rules.prompt_generators.base import extract_location_from_response, normalize_loc, locations_match

from codebase.tasks.identity_rules.models import CustomHookedTransformer
from behavioral_config import BehavioralConfig

logger = logging.getLogger(__name__)

def extract_single_answer_hybrid(response: str, question_type: str, loc1: str = None, loc2: str = None) -> str:
    """Extract answer using new simplified parsing."""
    return extract_location_from_response(response, loc1, loc2)

class BehavioralEvaluator:
    """Handles pure behavioral evaluation without mechanistic analysis."""
    
    def __init__(self, config: BehavioralConfig):
        self.config = config
    
    def evaluate_response(self, response: str, prompt) -> Dict[str, Any]:
        """Simple evaluation for new Prompt format with single question support."""
        # Check if this is a single question prompt
        question_type = getattr(prompt, 'question_type', 'dual')
        if hasattr(prompt, 'metadata') and 'question_type' in prompt.metadata:
            question_type = prompt.metadata['question_type']
        
        try:
            if question_type == "belief":
                # Only extract belief answer for belief questions
                belief_answer = extract_single_answer_hybrid(response, "belief")
                world_answer = None  # N/A
                
                belief_correct = normalize_loc(belief_answer) == normalize_loc(prompt.expected_belief)
                world_correct = None  # N/A - don't count in accuracy
                malformed = belief_answer == ""
                
            elif question_type == "world":
                # Only extract world answer for world questions
                belief_answer = None  # N/A
                world_answer = extract_single_answer_hybrid(response, "world")
                
                belief_correct = None  # N/A - don't count in accuracy
                world_correct = normalize_loc(world_answer) == normalize_loc(prompt.expected_world)
                malformed = world_answer == ""
                
            else:  # dual question
                # Extract both answers for dual questions
                belief_answer, world_answer = extract_dual_answers(response)
                belief_correct = normalize_loc(belief_answer) == normalize_loc(prompt.expected_belief)
                world_correct = normalize_loc(world_answer) == normalize_loc(prompt.expected_world)
                malformed = belief_answer == "" or world_answer == ""
                
        except Exception as e:
            # Fallback on error
            belief_answer = world_answer = ""
            belief_correct = world_correct = False
            malformed = True
        
        # Calculate overall correctness based on question type
        if question_type == "belief":
            correct = belief_correct
        elif question_type == "world":
            correct = world_correct
        else:  # dual
            correct = belief_correct and world_correct
        
        return {
            'correct': correct,
            'belief_correct': belief_correct,
            'world_correct': world_correct,
            'malformed': malformed,
            'belief_answer': belief_answer,
            'world_answer': world_answer,
            'response': response,
            'question_type': question_type
        }
    
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
        input_batch = input_ids.repeat(self.config.samples_per_condition, 1).to(model.cfg.device)
        generated_ids = model.generate(
            input_batch,
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
                # print the eos_token for debugging
                if eos_token in gen_text:
                    logger.debug(f"DEBUG: Found eos_token '{eos_token}' in gen_text, truncating")
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
            logger.info(f"Wandb imported successfully, project: {self.config.wandb_project}")
            
            # Check if wandb project is configured (can be disabled with --no_wandb)
            if not self.config.wandb_project:
                logger.info("Wandb logging disabled via configuration")
                self.wandb = None
                return
            
            # Try to authenticate - wandb will automatically use WANDB_API_KEY if available
            try:
                # Get API key from environment
                api_key = os.environ.get("WANDB_API_KEY", "")
                if not api_key:
                    # Try from codebase utils
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codebase"))
                    try:
                        from utils import WANDB_API_KEY
                        api_key = WANDB_API_KEY
                    except ImportError:
                        pass
                
                if api_key:
                    wandb.login(key=api_key)
                    logger.info("Wandb authentication successful with API key")
                else:
                    # Check if already logged in
                    if wandb.api.api_key:
                        logger.info("Wandb already authenticated")
                    else:
                        # This will prompt if interactive
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
                       prompt_variant: str,
                       base_rule: str,
                       template_name: str = "mixed",
                       context_type: str = "abstract",  # Default, not used in behavioral eval
                       run_name: str = None):
        """Initialize a new wandb run for a specific experimental condition."""
        if not self.wandb:
            logger.info("Wandb not available, skipping init_experiment")
            return
        
        logger.info(f"Initializing wandb experiment for {model_name}")
        
        model_info = self.config.get_model_info(model_name)
        
        if run_name is None:
            # Clean model name: "meta-llama/Llama-3.1-7B-Instruct" -> "llama-7b"
            model_clean = model_info['family'].lower() + '-' + model_info['size'].lower()
            
            # Clean template name: "basic_object_move" -> "basic-move" 
            template_clean = template_name.replace('_', '-').replace('basic-object-move', 'basic-move')
            
            # Abbreviate vignette type: false_belief -> fb, true_belief -> tb
            vignette_abbrev = vignette_type.replace('false_belief', 'fb').replace('true_belief', 'tb')
            
            # Question format: dual/belief/world
            question_abbrev = self.config.question_format if hasattr(self.config, 'question_format') else 'dual'
            
            run_name = f"{model_clean}_{template_clean}_{vignette_abbrev}_{question_abbrev}"
        
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
                'template_name': template_name,
                
                # Experimental parameters
                'temperatures': self.config.temperatures,
                '_conditionsamples_per': self.config.samples_per_condition,
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
                template_name,
                'behavioral_eval',
                'theory_of_mind',
                f"question_format_{self.config.question_format}"
            ] + ([f"single_type_{self.config.single_question_type}"] if self.config.question_format == "single" else [])
        )
    
    def log_condition_results(
        self,
        results: Dict[str, Any],
        temperature: float,
        prompt_response_pairs: List[Dict[str, Any]] = None
    ):
        """Log results for a specific temperature within the current run."""
        if not self.wandb:
            logger.debug("Wandb not available, skipping log_condition_results")
            return
        
        # Additional check to ensure wandb is properly initialized
        try:
            if not hasattr(self.wandb, 'run') or self.wandb.run is None:
                logger.warning("Wandb run not initialized, skipping log_condition_results")
                return
        except AttributeError:
            logger.warning("Wandb run attribute error, skipping log_condition_results")
            return
        
        logger.info(f"Logging results to wandb for temperature {temperature}")
        
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
        
        # Create artifact with prompt-response pairs if provided
        if prompt_response_pairs:
            self._create_prompt_response_artifact(prompt_response_pairs, temperature)
        
        # Use temperature as step value for meaningful x-axis in wandb plots
        # Scale by 100 to convert float to integer (e.g., 0.7 -> 70)
        temperature_step = int(temperature * 100)
        self.wandb.log(log_data, step=temperature_step)
    
    def _create_prompt_response_artifact(
        self, 
        prompt_response_pairs: List[Dict[str, Any]], 
        temperature: float
    ):
        """Create wandb artifact with structured prompt-response data."""
        if not self.wandb or not self.wandb.run:
            return
        
        # Create artifact
        artifact_name = f"prompt_responses_T{temperature:.1f}"
        artifact = self.wandb.Artifact(
            name=artifact_name,
            type="prompt_response_data",
            description=f"Prompt-response pairs for temperature {temperature}"
        )
        
        # Extract run-level metadata from first pair (same for all in this run)
        first_pair = prompt_response_pairs[0] if prompt_response_pairs else {}
        run_metadata = {
            'temperature': temperature,
            'vignette_type': first_pair.get('vignette_type', ''),
            'tom_format': first_pair.get('tom_format', ''),
            'model_name': getattr(self.wandb.run.config, 'model_name', '') if self.wandb.run else '',
            'total_prompts': len(prompt_response_pairs)
        }
        
        # Create responses list without redundant metadata
        responses = []
        for pair in prompt_response_pairs:
            response_details = pair.get('response_details', [{}])[0]  # Get first response
            
            response_entry = {
                'prompt': pair.get('prompt_text', ''),
                'expected_belief': pair.get('expected_belief', ''),
                'expected_world': pair.get('expected_world', ''),
                'cleaned_response': {
                    'belief': response_details.get('belief_answer', ''),
                    'world': response_details.get('world_answer', ''),
                    'belief_correct': response_details.get('belief_correct', False),
                    'world_correct': response_details.get('world_correct', False),
                    'malformed': response_details.get('malformed', False)
                },
                'raw_response': response_details.get('response', '')
            }
            responses.append(response_entry)
        
        # Final structure
        structured_data = {
            'run_metadata': run_metadata,
            'responses': responses
        }
        
        # Save as JSON and add to artifact
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(structured_data, f, indent=2)
            temp_path = f.name
        
        artifact.add_file(temp_path, name=f"prompt_responses_T{temperature:.1f}.json")
        
        # Log artifact to wandb
        self.wandb.log_artifact(artifact)
        
        # Clean up temp file
        os.unlink(temp_path)
    
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