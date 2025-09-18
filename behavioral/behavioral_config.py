"""
Configuration classes for behavioral evaluation experiments.
Focuses on pure generation + accuracy measurement without mechanistic analysis.
"""

from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class BehavioralConfig:
    """Configuration for behavioral theory of mind evaluation."""
    
    # Model settings
    models: List[str]
    device_map: str = "cpu"     # Use CPU for environments without CUDA
    device: str = "cpu"         # Target device for model and tensors  
    n_devices: int = 1
    
    # Experimental parameters
    temperatures: List[float] = None
    samples_per_condition: int = 10
    max_new_tokens: int = 50
    
    # Task settings
    vignette_types: List[str] = None  # ['false_belief', 'true_belief']
    tom_formats: List[str] = None     # ['direct', 'multiple_choice']
    prompt_variants: List[str] = None # ['standard', 'rephrased'] - for future token variants
    context_types: List[str] = None   # ['abstract'] - keep for compatibility with prompt generators
    # Note: base_rules automatically derived from vignette_types (false_belief=ABA, true_belief=ABB)
    
    # Data settings
    prompt_num: int = 50
    tom_locations_file: str = "codebase/tom_datasets/locations.txt"
    
    # Logging settings
    save_dir: str = "results/tom_performance"
    wandb_project: str = "tom-behavioral-eval"
    wandb_entity: Optional[str] = None
    log_raw_responses: bool = True
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Set defaults for list parameters."""
        if self.temperatures is None:
            self.temperatures = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
        
        if self.vignette_types is None:
            self.vignette_types = ['false_belief', 'true_belief']
        
        if self.tom_formats is None:
            self.tom_formats = ['direct', 'multiple_choice']
        
        if self.prompt_variants is None:
            self.prompt_variants = ['standard', 'detailed']  # Compare template types
            
        if self.context_types is None:
            self.context_types = ['abstract']  # Keep for prompt generator compatibility
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
    
    def get_model_info(self, model_name: str) -> dict:
        """Extract model family and size from model name."""
        model_lower = model_name.lower()
        
        # Determine model family
        if 'qwen' in model_lower:
            family = 'qwen'
        elif 'llama' in model_lower:
            family = 'llama'
        elif 'gemma' in model_lower:
            family = 'gemma'
        else:
            family = 'other'
        
        # Extract model size
        import re
        size_match = re.search(r'(\d+)b', model_lower)
        if size_match:
            size = f"{size_match.group(1)}B"
        else:
            size = "unknown"
        
        # Check if instruct model
        is_instruct = 'instruct' in model_lower or 'chat' in model_lower
        
        return {
            'family': family,
            'size': size,
            'is_instruct': is_instruct,
            'full_name': model_name
        }

@dataclass 
class DefaultBehavioralConfig:
    """Predefined configurations for common experimental setups."""
    
    @staticmethod
    def quick_test() -> BehavioralConfig:
        """Fast configuration for testing."""
        return BehavioralConfig(
            models=['Llama-3.1-8B-Instruct'],
            temperatures=[1.0],
            samples_per_condition=10,
            vignette_types=['false_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['detailed', 'standard']
        )
    
    @staticmethod
    def full_comparison() -> BehavioralConfig:
        """Full comparison between target models."""
        return BehavioralConfig(
            models=[
                'Qwen2.5-14B-Instruct'
            ],
            temperatures=[0.1, 0.4, 0.7, 1.0, 1.3],
            samples_per_condition=10,
            vignette_types=['true_belief', 'false_belief'],
            tom_formats=['direct', 'multiple_choice'],
            context_types=['abstract'],
            prompt_variants=['standard', 'detailed']
        )
        
    def big_model_comparison() -> BehavioralConfig:
        """One full pass w llama 70B"""
        return BehavioralConfig(
            models=[
                'Llama-3.1-70B-Instruct'
            ],
            temperatures=[0.1, 0.4, 0.7, 1.0, 1.3],
            samples_per_condition=10,
            vignette_types=['true_belief', 'false_belief'],
            prompt_variants=['standard', 'detailed']
        )
            
    
    @staticmethod
    def temperature_sweep(model_name: str) -> BehavioralConfig:
        """Focus on temperature effects for single model."""
        return BehavioralConfig(
            models=[model_name],
            temperatures=[0.1, 0.4, 0.7, 1.0, 1.3],
            samples_per_condition=10,
            vignette_types=['false_belief', 'true_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['standard', 'detailed']
        )