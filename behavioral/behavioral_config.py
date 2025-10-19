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
    device_map: str = "cuda"    # Use single CUDA device  
    device: str = "cuda"        # Target device for model and tensors  
    n_devices: int = 1
    load_in_8bit: bool = False  # Disabled: transformer_lens doesn't support quantization
    load_in_4bit: bool = False
    
    # Experimental parameters
    temperatures: List[float] = None
    samples_per_condition: int = 10
    max_new_tokens: int = 20 # Limit output length for efficiency 
    batch_size: int = 1  # Keep batch size small to avoid OOM
    
    # Task settings
    vignette_types: List[str] = None  # ['false_belief', 'true_belief']
    tom_formats: List[str] = None     # ['direct', 'multiple_choice']
    prompt_variants: List[str] = None # ['standard', 'detailed'] 
    context_types: List[str] = None   # ['abstract'] - keep for compatibility with prompt generators
    # Note: base_rules automatically derived from vignette_types (false_belief=ABA, true_belief=ABB)
    
    # Question format settings
    question_format: str = "single"     # "single" or "dual"
    single_question_type: str = "belief"  # "belief", "world", or "mixed" (only used if question_format="single")
    
    # Template settings
    template_names: List[str] = None  # specific templates: ["basic_object_move", "food_truck", "hair_styling", etc.]
    
    # Data settings
    prompt_num: int = 20
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
        
        if self.template_names is None:
            self.template_names = ['food_truck', 'hair_styling', 'library_book']
        
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
            models=['Llama-3.1-70B-Instruct'],
            temperatures=[0.1, 0.4, 0.7, 1.0, 1.3],
            samples_per_condition=5,
            vignette_types=['false_belief', 'true_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['detailed', 'standard']
        )
    
    @staticmethod
    def single_question_test() -> BehavioralConfig:
        """Test single question format."""
        return BehavioralConfig(
            models=['Qwen2.5-14B-Instruct'],
            temperatures=[0.1],
            samples_per_condition=3,
            vignette_types=['false_belief', 'true_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['standard'],
            question_format="single",
            template_names=['food_truck', 'library_book', 'basic_object_move_detailed']
        )
    
    @staticmethod
    def full_comparison() -> BehavioralConfig:
        """Full comparison between target models."""
        return BehavioralConfig(
            models=[
                'Qwen2.5-14B-Instruct'
            ],
            temperatures=[0.1, 0.4, 0.7],
            samples_per_condition=1,
            vignette_types=['true_belief', 'false_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['detailed', 'standard']
        )
    
    @staticmethod    
    def big_model_comparison() -> BehavioralConfig:
        """One full pass w llama 70B"""
        return BehavioralConfig(
            models=[
                'Llama-3.1-70B-Instruct'
            ],
            temperatures=[0.1, 0.4, 0.7],
            samples_per_condition=1,
            vignette_types=['true_belief', 'false_belief'],
            context_types=['abstract'],
            prompt_variants=['standard', 'detailed']
        )
            
    
    @staticmethod
    def temperature_sweep(model_name: str) -> BehavioralConfig:
        """Focus on temperature effects for single model."""
        return BehavioralConfig(
            models=[model_name],
            temperatures=[0.1, 0.4, 0.7],
            samples_per_condition=1,
            vignette_types=['false_belief', 'true_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['standard', 'detailed']
        )
        
    @staticmethod
    def minimal_test_true() -> BehavioralConfig:
        """Minimal configuration for quick debugging."""
        return BehavioralConfig(
            models=['Qwen2.5-14B-Instruct'],
            temperatures=[0.1, 0.4, 0.7],
            samples_per_condition=2,
            vignette_types=['true_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['detailed']
        )
    @staticmethod
    def minimal_test_false() -> BehavioralConfig:
        """Minimal configuration for quick debugging."""
        return BehavioralConfig(
            models=['Qwen2.5-14B-Instruct'],
            temperatures=[0.1, 0.4, 0.7],
            samples_per_condition=2,
            vignette_types=['false_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['detailed']
        )
    
    @staticmethod
    def wandb_test() -> BehavioralConfig:
        """Minimal test specifically for wandb debugging with 1B model."""
        return BehavioralConfig(
            models=['Llama-3.2-1B'],
            temperatures=[0.1, 0.4],
            samples_per_condition=2,
            vignette_types=['false_belief'],
            tom_formats=['direct'],
            context_types=['abstract'],
            prompt_variants=['standard'],
            prompt_num=3
        )