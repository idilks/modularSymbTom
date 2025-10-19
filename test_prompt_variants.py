#!/usr/bin/env python3
"""
Quick test of prompt variant functionality
"""

import sys
sys.path.append('codebase/tasks/identity_rules')
sys.path.append('behavioral')

from behavioral_config import BehavioralConfig, DefaultBehavioralConfig
from cma_config import ExperimentConfig, ModelConfig, GenerationConfig, PromptConfig, PatchingConfig, EvaluationConfig
from prompt_generators import get_prompt_generator

def create_experiment_config(model_name: str, prompt_variant: str = "standard") -> ExperimentConfig:
    """Create test experiment config."""
    model_config = ModelConfig(
        model_type=model_name,
        device_map="cuda:0",
        n_devices=1
    )
    
    generation_config = GenerationConfig(
        max_new_tokens=70,
        temperature=0.7,
        do_sample=True
    )
    
    prompt_config = PromptConfig(
        base_rule="ABA",
        context_type="abstract",
        prompt_num=2,
        use_tom_prompts=True,
        ask_world=True,
        tom_format="direct",
        prompt_variant=prompt_variant,
        tom_locations_file="codebase/tasks/identity_rules/tom_datasets/locations.txt"
    )
    
    patching_config = PatchingConfig(
        activation_name='z',
        token_pos_list=[-1]
    )
    
    evaluation_config = EvaluationConfig()
    
    return ExperimentConfig(
        model=model_config,
        generation=generation_config,
        prompts=prompt_config,
        patching=patching_config,
        evaluation=evaluation_config
    )

def test_prompt_variants():
    """Test both standard and detailed prompt variants."""
    print("Testing prompt variants...")
    
    # Mock tokenizer (we're just testing prompt generation)
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
    
    tokenizer = MockTokenizer()
    
    for variant in ["standard", "detailed"]:
        print(f"\n=== Testing {variant} variant ===")
        
        config = create_experiment_config("test-model", prompt_variant=variant)
        prompt_generator = get_prompt_generator(config, tokenizer)
        
        try:
            prompts, answers, causal_ans = prompt_generator.generate_prompts()
            
            print(f"Generated {len(prompts)} prompt pairs")
            print(f"First prompt pair:")
            print(f"  Base: {prompts[0][0][:150]}...")
            print(f"  Exp:  {prompts[0][1][:150]}...")
            print(f"  Answer: {answers[0]}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_prompt_variants()