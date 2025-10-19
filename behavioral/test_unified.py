"""
Test the unified template system.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from unified_prompt_builder import UnifiedPromptBuilder

def test_unified_system():
    """Test the unified template system."""
    
    print("=== Testing Unified Template System ===\n")
    
    builder = UnifiedPromptBuilder()
    
    print("Available templates:")
    for template in builder.list_available_templates():
        print(f"  - {template}")
    print()
    
    # Test basic template
    print("--- BASIC TEMPLATE (standard) ---")
    basic_prompt = builder.generate_prompt(
        template_name="basic_object_move",
        vignette_type="false_belief",
        question_type="dual",
        variant="standard"
    )
    print(basic_prompt.text)
    print(f"Belief: {basic_prompt.expected_belief} | World: {basic_prompt.expected_world}")
    print()
    
    # Test basic detailed
    print("--- BASIC TEMPLATE (detailed) ---")
    basic_detailed = builder.generate_prompt(
        template_name="basic_object_move_detailed",
        vignette_type="false_belief",
        question_type="dual",
        variant="detailed"
    )
    print(basic_detailed.text)
    print(f"Belief: {basic_detailed.expected_belief} | World: {basic_detailed.expected_world}")
    print()
    
    # Test naturalistic templates
    naturalistic_templates = ["food_truck", "hair_styling", "library_book", "restaurant_reservation"]
    
    for template in naturalistic_templates:
        print(f"--- {template.upper()} TEMPLATE ---")
        try:
            prompt = builder.generate_prompt(
                template_name=template,
                vignette_type="false_belief",
                question_type="dual",
                variant="detailed"
            )
            print(prompt.text[:200] + "..." if len(prompt.text) > 200 else prompt.text)
            print(f"Belief: {prompt.expected_belief} | World: {prompt.expected_world}")
        except Exception as e:
            print(f"ERROR: {e}")
        print()
    
    # Test batch generation
    print("--- BATCH GENERATION (mixed templates) ---")
    prompts = builder.batch_generate(
        num_prompts=4,
        template_names=["basic_object_move", "food_truck"],
        vignette_types=["false_belief"],
        question_types=["dual"],
        variants=["standard"]
    )
    
    for i, prompt in enumerate(prompts, 1):
        template_name = prompt.metadata['template_name']
        print(f"PROMPT {i} ({template_name}):")
        print(prompt.text[:150] + "..." if len(prompt.text) > 150 else prompt.text)
        print(f"Belief: {prompt.expected_belief} | World: {prompt.expected_world}")
        print()
    
    # Test paired generation
    print("--- PAIRED GENERATION (single questions) ---")
    pairs = builder.batch_generate(
        num_prompts=4,
        template_names=["basic_object_move"],
        vignette_types=["false_belief"],
        question_types=["belief", "world"],  # This triggers pairing
        variants=["standard"]
    )
    
    # Group by pair_id
    pair_groups = {}
    for prompt in pairs:
        pair_id = prompt.metadata.get('pair_id', 'unknown')
        if pair_id not in pair_groups:
            pair_groups[pair_id] = []
        pair_groups[pair_id].append(prompt)
    
    for pair_id, pair_prompts in pair_groups.items():
        print(f"PAIR {pair_id}:")
        for prompt in pair_prompts:
            q_type = prompt.metadata['question_type']
            print(f"  {q_type}: {prompt.text}")
        print()

if __name__ == "__main__":
    test_unified_system()