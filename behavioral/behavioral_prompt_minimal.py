"""
Behavioral-only prompt generator extracted from working CMA theory of mind system.
Preserves exact prompt formatting that achieved 40% accuracy.
"""

import os
import random
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# Import from working CMA system
import sys
sys.path.append('codebase/tasks/identity_rules')
from cma_config import LOC_L, LOC_R
from prompt_generators.base import wrap_dual_schema

@dataclass
class MinimalBehavioralPrompt:
    """Single prompt extracted from working CMA system."""
    text: str
    expected_answer: Tuple[str, str]  # (belief, world) or single answer
    vignette_type: str
    tom_format: str
    prompt_variant: str

class MinimalBehavioralGenerator:
    """Extracts working exp_prompt logic from CMA theory of mind generator."""
    
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        
    def generate_prompts(self, num_prompts: int = None) -> List[MinimalBehavioralPrompt]:
        """Generate behavioral prompts using working CMA logic."""
        if num_prompts is None:
            num_prompts = self.config.prompts.prompt_num
            
        # Load locations exactly like working system
        locations_file = self.config.prompts.tom_locations_file
        assert os.path.exists(locations_file), f"Locations file {locations_file} does not exist."
        
        with open(locations_file, "r") as f:
            location_phrases = [l.rstrip() for l in f.readlines()]
        
        prompts = []
        
        for vignette_type in self.config.vignette_types:
            for tom_format in self.config.tom_formats:
                for prompt_variant in self.config.prompt_variants:
                    
                    # Generate prompts for this condition
                    type_count = num_prompts // (
                        len(self.config.vignette_types) * 
                        len(self.config.tom_formats) * 
                        len(self.config.prompt_variants)
                    )
                    
                    for _ in range(type_count):
                        # Get two random locations
                        test_loc_a, test_loc_b = random.sample(location_phrases, 2)
                        
                        # Map vignette type to base_rule for working logic
                        base_rule = "ABA" if vignette_type == "false_belief" else "ABB"
                        
                        # Generate using working CMA logic - we only need exp_prompt
                        if prompt_variant == "detailed":
                            question, answer = self._create_detailed_question(
                                test_loc_a, test_loc_b, base_rule
                            )
                        else:  # standard
                            question, answer = self._create_standard_question(
                                test_loc_a, test_loc_b, base_rule
                            )
                        
                        # Apply same formatting as working system
                        if self.config.prompts.ask_world:
                            question = wrap_dual_schema(question)
                        
                        # Add multiple choice or location tags like working system
                        if tom_format == "multiple_choice":
                            question, answer = self._add_multiple_choice_format(
                                question, test_loc_a, test_loc_b, answer
                            )
                        else:  # direct format
                            if not self.config.prompts.ask_world:
                                question += f" {LOC_L}"
                        
                        prompts.append(MinimalBehavioralPrompt(
                            text=question,
                            expected_answer=answer,
                            vignette_type=vignette_type,
                            tom_format=tom_format,
                            prompt_variant=prompt_variant
                        ))
        
        random.shuffle(prompts)
        return prompts[:num_prompts]
    
    def _create_standard_question(self, test_loc_a: str, test_loc_b: str, base_rule: str) -> Tuple[str, Tuple[str, str]]:
        """Create standard question using exact working CMA logic."""
        
        if base_rule == "ABA":  # false belief - we want the exp_question from abstract context
            # This creates the main false belief scenario
            question = f"object is located at {LOC_L}{test_loc_a}{LOC_R}. agent leaves room. object moves to {LOC_L}{test_loc_b}{LOC_R}. agent returns and looks where?"
            
            if self.config.prompts.ask_world:
                answer = (test_loc_a, test_loc_b)  # (belief, world)
            else:
                answer = test_loc_a  # belief only
                
        else:  # base_rule == "ABB", true belief
            # Agent witnesses the move
            question = f"object is located at {LOC_L}{test_loc_a}{LOC_R}. object moves to {LOC_L}{test_loc_b}{LOC_R}. agent leaves room. agent returns and looks where?"
            
            if self.config.prompts.ask_world:
                answer = (test_loc_b, test_loc_b)  # (belief, world) - agent has true belief
            else:
                answer = test_loc_b  # belief only
        
        return question, answer
    
    def _create_detailed_question(self, test_loc_a: str, test_loc_b: str, base_rule: str) -> Tuple[str, Tuple[str, str]]:
        """Create detailed question using exact working CMA logic."""
        
        if base_rule == "ABA":  # false belief
            question = f"Alice places her book on the {LOC_L}{test_loc_a}{LOC_R}. She then leaves the room and does not see what is happening in the room. While she is gone Bob moves the book to the {LOC_L}{test_loc_b}{LOC_R}. When Alice returns, where will she look for her book?"
            
            if self.config.prompts.ask_world:
                answer = (test_loc_a, test_loc_b)  # (belief, world)
            else:
                answer = test_loc_a  # belief only
                
        else:  # base_rule == "ABB", true belief  
            question = f"Alice places her book on the {LOC_L}{test_loc_a}{LOC_R}. While she is there, Bob moves the book to the {LOC_L}{test_loc_b}{LOC_R}. Alice then leaves the room. When she returns, where will she look for her book?"
            
            if self.config.prompts.ask_world:
                answer = (test_loc_b, test_loc_b)  # (belief, world) - agent has true belief
            else:
                answer = test_loc_b  # belief only
        
        return question, answer
    
    def _add_multiple_choice_format(self, question: str, test_loc_a: str, test_loc_b: str, answer):
        """Add multiple choice format using working CMA logic."""
        question += f"\nA: {LOC_L}{test_loc_a}{LOC_R}\nB: {LOC_L}{test_loc_b}{LOC_R}\nanswer:"
        
        # Convert answer to A/B format using belief component (exact same logic as working system)
        def get_belief_component(ans):
            return ans[0] if isinstance(ans, tuple) else ans
        
        belief = get_belief_component(answer)
        answer_formatted = "A" if belief == test_loc_a else "B"
        
        return question, answer_formatted

def get_minimal_behavioral_generator(config, tokenizer=None) -> MinimalBehavioralGenerator:
    """Factory function for minimal behavioral generator."""
    return MinimalBehavioralGenerator(config, tokenizer)