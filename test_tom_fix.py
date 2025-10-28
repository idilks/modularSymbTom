#!/usr/bin/env python3
"""Test the fixed ToM prompt generator to verify base_rule logic."""

import sys
sys.path.append('codebase/tasks/identity_rules')

from cma_config import ExperimentConfig, ModelConfig, GenerationConfig, PromptConfig, PatchingConfig, EvaluationConfig, WandbConfig
from prompt_generators import get_prompt_generator

# Mock tokenizer
class MockTokenizer:
    def __init__(self):
        self.vocab_size = 32000

def test_base_rule_mapping():
    """Test that base_rule parameter correctly maps belief types."""
    
    # Test ABA base_rule (false belief as base)
    config_aba = ExperimentConfig(
        model=ModelConfig(model_type="test"),
        generation=GenerationConfig(),
        prompts=PromptConfig(
            base_rule="ABA",
            context_type="abstract", 
            prompt_num=2,
            use_behavioral_tom=True,
            template_names=["food_truck"]
        ),
        patching=PatchingConfig(),
        evaluation=EvaluationConfig(),
        wandb=WandbConfig()
    )
    
    # Test ABB base_rule (true belief as base)
    config_abb = ExperimentConfig(
        model=ModelConfig(model_type="test"),
        generation=GenerationConfig(),
        prompts=PromptConfig(
            base_rule="ABB",
            context_type="abstract",
            prompt_num=2, 
            use_behavioral_tom=True,
            template_names=["food_truck"]
        ),
        patching=PatchingConfig(),
        evaluation=EvaluationConfig(),
        wandb=WandbConfig()
    )
    
    tokenizer = MockTokenizer()
    
    print("=== Testing ABA base_rule (false belief as base) ===")
    generator_aba = get_prompt_generator(config_aba, tokenizer)
    prompt_pairs_aba, ans_pairs_aba, causal_aba = generator_aba.generate_prompts()
    
    print(f"Base prompt (should be false belief):\n{prompt_pairs_aba[0][0]}")
    print(f"\nExp prompt (should be true belief):\n{prompt_pairs_aba[0][1]}")
    print(f"\nBase answer: {ans_pairs_aba[0][0]}")
    print(f"Exp answer: {ans_pairs_aba[0][1]}")
    print(f"Causal answer: {causal_aba[0]}")
    
    print("\n" + "="*60)
    print("=== Testing ABB base_rule (true belief as base) ===")
    generator_abb = get_prompt_generator(config_abb, tokenizer)
    prompt_pairs_abb, ans_pairs_abb, causal_abb = generator_abb.generate_prompts()
    
    print(f"Base prompt (should be true belief):\n{prompt_pairs_abb[0][0]}")
    print(f"\nExp prompt (should be false belief):\n{prompt_pairs_abb[0][1]}")
    print(f"\nBase answer: {ans_pairs_abb[0][0]}")
    print(f"Exp answer: {ans_pairs_abb[0][1]}")
    print(f"Causal answer: {causal_abb[0]}")

if __name__ == "__main__":
    test_base_rule_mapping()