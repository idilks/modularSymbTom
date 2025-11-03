#!/usr/bin/env python3
"""
Test script to generate concrete examples of abstract context prompts
showing how ABA vs ABB base rules create different belief state assignments.
"""

import sys
import os
sys.path.append("codebase/tasks/identity_rules")

from unified_templates import UnifiedTemplateSystem, UnifiedQuestionFormatter
from transformers import AutoTokenizer

def test_abstract_context_logic():
    """Generate examples showing ABA vs ABB abstract context differences."""
    
    # Initialize components
    vocab_dir = "tom_datasets"
    template_system = UnifiedTemplateSystem(vocab_dir)
    
    # Use a simple tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    question_formatter = UnifiedQuestionFormatter(tokenizer)
    
    template_name = "food_truck"
    context_type = "abstract"
    
    print("=== ABSTRACT CONTEXT LOGIC EXAMPLES ===\n")
    
    # Test ABA base rule (false belief as base)
    print("ABA BASE RULE (false belief as base, true belief as exp):")
    print("-" * 60)
    
    base_scenario_aba, exp_scenario_aba, base_ans_aba, exp_ans_aba, causal_ans_aba = (
        template_system.generate_better_cma_pair("food_truck", "ABA", "abstract")
    )
    
    question = question_formatter.get_cma_questions("food_truck", "abstract", "ABA", "instruction")
    
    print(f"BASE (false belief): {base_scenario_aba}")
    print(f"BASE question: {question}")
    print(f"BASE expected answer: {base_ans_aba}")
    print()
    print(f"EXP (true belief): {exp_scenario_aba}")
    print(f"EXP question: {question}")
    print(f"EXP expected answer: {exp_ans_aba}")
    print()
    print(f"CAUSAL hypothesis: After patching base->exp, expect {causal_ans_aba}")
    print(f"(Belief tracking heads should preserve {base_ans_aba} → {causal_ans_aba})")
    print()
    
    # Test ABB base rule (true belief as base)
    print("ABB BASE RULE (true belief as base, false belief as exp):")
    print("-" * 60)
    
    base_scenario_abb, exp_scenario_abb, base_ans_abb, exp_ans_abb, causal_ans_abb = (
        template_system.generate_better_cma_pair("food_truck", "ABB", "abstract")
    )
    
    print(f"BASE (true belief): {base_scenario_abb}")
    print(f"BASE question: {question}")
    print(f"BASE expected answer: {base_ans_abb}")
    print()
    print(f"EXP (false belief): {exp_scenario_abb}")
    print(f"EXP question: {question}")
    print(f"EXP expected answer: {exp_ans_abb}")
    print()
    print(f"CAUSAL hypothesis: After patching base->exp, expect {causal_ans_abb}")
    print(f"(Belief tracking heads should preserve {base_ans_abb} → {causal_ans_abb})")
    print()
    
    # Show key insight
    print("KEY INSIGHT:")
    print("-" * 60)
    print("• ABA = false belief BASE → tests if patching preserves false belief state")
    print("• ABB = true belief BASE → tests if patching preserves true belief state")
    print("• Abstract context creates DIFFERENT belief states (false vs true)")
    print("• This isolates belief-tracking mechanisms from location-retrieval mechanisms")
    print()
    
    # Show contrast with token context (if available)
    print("CONTRAST: Token context (same belief state, different locations):")
    print("-" * 60)
    
    try:
        base_scenario_token, exp_scenario_token, base_ans_token, exp_ans_token, causal_ans_token = (
            template_system.generate_control_cma_pair("food_truck", "ABA", "control")
        )
        
        print(f"TOKEN BASE: {base_scenario_token}")
        print(f"TOKEN EXP: {exp_scenario_token}")
        print(f"Same belief type, different location tokens ({base_ans_token} vs {exp_ans_token})")
        print("→ Tests location retrieval, not belief tracking")
    except Exception as e:
        print(f"Token context generation failed: {e}")

if __name__ == "__main__":
    test_abstract_context_logic()