#!/usr/bin/env python3
"""
Simple test script to see what BehavioralPromptGenerator actually outputs
"""

import sys
sys.path.append('behavioral')

from behavioral_prompt_generator import BehavioralPromptGenerator
from dataclasses import dataclass

@dataclass 
class MockPromptConfig:
    tom_locations_file: str = "tom_datasets/locations.txt"
    prompt_num: int = 5

@dataclass
class MockConfig:
    ask_world: bool = True
    prompts: MockPromptConfig = None
    vignette_types: list = None
    tom_formats: list = None  
    prompt_variants: list = None
    
    def __post_init__(self):
        if self.prompts is None:
            self.prompts = MockPromptConfig()
        if self.vignette_types is None:
            self.vignette_types = ["false_belief", "true_belief"]
        if self.tom_formats is None:
            self.tom_formats = ["direct", "multiple_choice"]
        if self.prompt_variants is None:
            self.prompt_variants = ["standard", "detailed"]

def test_prompt_generation():
    config = MockConfig()
    generator = BehavioralPromptGenerator(config)
    
    print("=" * 80)
    print("BEHAVIORAL PROMPT GENERATOR TEST")
    print("=" * 80)
    
    # Generate prompts using the generator
    try:
        prompts = generator.generate_prompts(num_prompts=8)  # 2 per type combo
        
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"PROMPT {i+1}: {prompt.vignette_type.upper()} + {prompt.tom_format.upper()} + {prompt.prompt_variant.upper()}")
            print(f"{'='*60}")
            
            print(f"PROMPT TEXT (repr):")
            print(repr(prompt.text))
            print(f"\nPROMPT TEXT (display):")
            print(prompt.text)
            print(f"\nEXPECTED BELIEF: {prompt.expected_belief}")
            print(f"EXPECTED WORLD: {prompt.expected_world}")
            print(f"VIGNETTE TYPE: {prompt.vignette_type}")
            print(f"TOM FORMAT: {prompt.tom_format}")
            print(f"PROMPT VARIANT: {prompt.prompt_variant}")
                
    except Exception as e:
        print(f"ERROR generating prompts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prompt_generation()