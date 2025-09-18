"""
Behavioral-only prompt generator for theory of mind evaluation.
Generates single prompts without base/exp pairs - no CMA complexity.
"""

import random
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class BehavioralPrompt:
    """Single prompt for behavioral evaluation."""
    text: str
    expected_belief: str
    expected_world: str
    vignette_type: str  # 'false_belief' or 'true_belief'
    tom_format: str     # 'direct' or 'multiple_choice'
    prompt_variant: str # 'standard' or 'detailed'

class BehavioralPromptGenerator:
    """Generates theory of mind prompts for pure behavioral evaluation."""
    
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.locations = self._load_locations()
        
    def _load_locations(self) -> List[str]:
        """Load location phrases from file."""
        try:
            with open(self.config.prompts.tom_locations_file, 'r') as f:
                locations = [line.strip() for line in f if line.strip()]
            return locations
        except FileNotFoundError:
            # Fallback locations if file not found
            return [
                "under the table", "next to the shelf", "in the kitchen",
                "on the counter", "behind the door", "in the garden",
                "near the window", "by the fireplace", "in the bedroom",
                "on the desk", "inside the box", "under the bed"
            ]
    
    def generate_prompts(self, num_prompts: int = None) -> List[BehavioralPrompt]:
        """Generate behavioral evaluation prompts."""
        if num_prompts is None:
            num_prompts = self.config.prompts.prompt_num
            
        prompts = []
        
        # Generate prompts for each vignette type
        for vignette_type in self.config.vignette_types:
            for tom_format in self.config.tom_formats:
                for prompt_variant in self.config.prompt_variants:
                    
                    # Generate equal numbers of each type
                    type_prompts = num_prompts // (
                        len(self.config.vignette_types) * 
                        len(self.config.tom_formats) * 
                        len(self.config.prompt_variants)
                    )
                    
                    for _ in range(type_prompts):
                        prompt = self._generate_single_prompt(
                            vignette_type=vignette_type,
                            tom_format=tom_format,
                            prompt_variant=prompt_variant
                        )
                        prompts.append(prompt)
        
        random.shuffle(prompts)
        return prompts[:num_prompts]
    
    def _generate_single_prompt(self, vignette_type: str, tom_format: str, prompt_variant: str) -> BehavioralPrompt:
        """Generate a single behavioral prompt."""
        
        # Select two different locations
        loc1, loc2 = random.sample(self.locations, 2)
        
        # Generate base scenario
        if vignette_type == "false_belief":
            # Agent doesn't witness the move
            scenario, belief_answer, world_answer = self._create_false_belief_scenario(loc1, loc2, prompt_variant)
        else:  # true_belief
            # Agent witnesses the move
            scenario, belief_answer, world_answer = self._create_true_belief_scenario(loc1, loc2, prompt_variant)
        
        # Format according to tom_format
        if tom_format == "direct":
            prompt_text = self._format_direct_prompt(scenario, prompt_variant)
        else:  # multiple_choice
            prompt_text = self._format_multiple_choice_prompt(scenario, loc1, loc2, prompt_variant)
        
        return BehavioralPrompt(
            text=prompt_text,
            expected_belief=belief_answer,
            expected_world=world_answer,
            vignette_type=vignette_type,
            tom_format=tom_format,
            prompt_variant=prompt_variant
        )
    
    def _create_false_belief_scenario(self, loc1: str, loc2: str, variant: str) -> Tuple[str, str, str]:
        """Create false belief scenario where agent misses object movement."""
        
        if variant == "detailed":
            scenario = (
                f"Alice places her book {loc1}. "
                f"Alice leaves the room and goes outside and cannot see inside of the room anymore. "
                f"While Alice is away, John moves the book from {loc1} to {loc2}. "
                f"After this, the book stays in place, nobody moves it."
                f"Alice returns to the room. "
                f"Where will Alice look for her book?"
            )
        else:  # standard
            scenario = (
                f"object is located at {loc1}. agent leaves room. "
                f"object is moved to {loc2}. agent returns and looks where?"
            )
        
        return scenario, loc1, loc2  # belief=loc1, world=loc2
    
    def _create_true_belief_scenario(self, loc1: str, loc2: str, variant: str) -> Tuple[str, str, str]:
        """Create true belief scenario where agent witnesses object movement."""
        
        if variant == "detailed":
            scenario = (
                f"Alice places her book {loc1}. "
                f"John moves the book from {loc1} to {loc2}. "
                f"Alice sees John move the book. "
                f"Alice leaves the room and goes outside and cannot see inside of the room anymore. "
                f"After this, the book stays in place, nobody moves it."
                f"Alice returns to the room. "
                f"Where will Alice look for her book?"
            )
        else:  # standard
            scenario = (
                f"object is located at {loc1}. object is moved to {loc2}. "
                f"agent leaves room. agent returns and looks where?"
            )
        
        return scenario, loc2, loc2  # belief=loc2, world=loc2
    
    def _format_direct_prompt(self, scenario: str, variant: str) -> str:
        """Format prompt for direct dual-answer format."""
        
        if variant == "detailed":
            return (
                f"{scenario}\n\n"
                f"Please answer in this exact format:\n"
                f"belief: <loc>location where agent will look</loc>\n"
                f"world: <loc>actual location of object</loc>"
            )
        else:  # standard
            return (
                f"{scenario}\n"
                f"answer in this exact schema, no extra text:\n"
                f"belief: <loc></loc>\n"
                f"world: <loc></loc>"
            )
    
    def _format_multiple_choice_prompt(self, scenario: str, loc1: str, loc2: str, variant: str) -> str:
        """Format prompt for multiple choice format."""
        
        # Randomize choice order
        choices = [loc1, loc2]
        random.shuffle(choices)
        
        if variant == "detailed":
            prompt = (
                f"{scenario}\n\n"
                f"Please select the best answer:\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n\n"
                f"Answer with just the letter (A or B):"
            )
        else:  # standard
            prompt = (
                f"{scenario}\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n"
                f"Answer with just the letter (A or B):"
            )
        
        return prompt

def get_behavioral_prompt_generator(config, tokenizer=None) -> BehavioralPromptGenerator:
    """Factory function to create behavioral prompt generator."""
    return BehavioralPromptGenerator(config, tokenizer)