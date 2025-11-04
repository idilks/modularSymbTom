#!/usr/bin/env python3
"""
Simple test to understand abstract context logic based on saved results.
"""

import sys
import os

def analyze_saved_abstract_results():
    """Analyze saved abstract context examples to understand ABA/ABB logic."""
    
    print("=== ABSTRACT CONTEXT ANALYSIS FROM SAVED RESULTS ===\n")
    
    # Read saved examples
    base_prompt_file = "/mnt/c/Users/idilk/Desktop/tom_dev/codebase/results/identity_rules/cma/meta-llama/Llama-3.2-1B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_False/base_input_prompts_1.txt"
    exp_prompt_file = "/mnt/c/Users/idilk/Desktop/tom_dev/codebase/results/identity_rules/cma/meta-llama/Llama-3.2-1B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_False/exp_input_prompts_1.txt"
    ans_file = "/mnt/c/Users/idilk/Desktop/tom_dev/codebase/results/identity_rules/cma/meta-llama/Llama-3.2-1B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_False/ans_1.txt"
    
    try:
        with open(base_prompt_file, 'r', encoding='utf-8') as f:
            base_content = f.read().strip()
        with open(exp_prompt_file, 'r', encoding='utf-8') as f:
            exp_content = f.read().strip()
        with open(ans_file, 'r', encoding='utf-8') as f:
            ans_content = f.read().strip()
        
        print("ABA BASE RULE EXAMPLE (saved results):")
        print("-" * 60)
        print("BASE PROMPT (false belief scenario):")
        print(base_content)
        print()
        print("EXP PROMPT (true belief scenario):")
        print(exp_content)
        print()
        print("EXPECTED ANSWERS:")
        print(ans_content)
        print()
        
        # Extract key differences
        print("KEY ANALYSIS:")
        print("-" * 60)
        
        # Parse prompts to show structure
        base_lines = base_content.split('\n')
        exp_lines = exp_content.split('\n')
        
        if len(base_lines) >= 2 and len(exp_lines) >= 2:
            base_scenario = base_lines[0].replace('→', '').strip()
            exp_scenario = exp_lines[0].replace('→', '').strip()
            
            print(f"BASE scenario: {base_scenario}")
            print(f"EXP scenario:  {exp_scenario}")
            print()
            
            # Identify the key difference
            if "agent leaves room. object moved" in base_scenario and "object moved" in exp_scenario:
                if base_scenario.find("agent leaves") < base_scenario.find("object moved"):
                    print("BASE = FALSE BELIEF: Agent leaves BEFORE object moves (doesn't see move)")
                if exp_scenario.find("object moved") < exp_scenario.find("agent leaves"):
                    print("EXP = TRUE BELIEF: Object moves BEFORE agent leaves (agent sees move)")
            
            # Parse answers
            if ans_content:
                ans_lines = ans_content.split('\n')
                for line in ans_lines:
                    if "base_ans:" in line and "exp_ans:" in line and "causal_ans:" in line:
                        parts = line.split(',')
                        base_ans = parts[0].split('base_ans:')[1].strip()
                        exp_ans = parts[1].split('exp_ans:')[1].strip() 
                        causal_ans = parts[2].split('causal_ans:')[1].strip()
                        
                        print()
                        print(f"BASE answer (false belief): {base_ans}")
                        print(f"EXP answer (true belief): {exp_ans}")
                        print(f"CAUSAL expected (after patching): {causal_ans}")
                        print()
                        print("CAUSAL HYPOTHESIS:")
                        print(f"-> Patching belief-tracking heads should change {exp_ans} -> {causal_ans}")
                        print(f"-> This tests whether heads track agent's BELIEF STATE vs WORLD STATE")
        
    except FileNotFoundError as e:
        print(f"Could not find saved results: {e}")
        print("Need to run actual CMA experiment to generate examples.")
    except Exception as e:
        print(f"Error reading files: {e}")
        
    print("\n" + "=" * 80)
    print("ABSTRACT CONTEXT LOGIC SUMMARY:")
    print("=" * 80)
    print("- ABA base rule = FALSE belief as base condition")
    print("- ABB base rule = TRUE belief as base condition") 
    print("- Abstract context varies BELIEF STATES (false vs true)")
    print("- Token context varies LOCATION TOKENS (same belief type)")
    print("- Causal hypothesis: belief-tracking heads preserve base belief state")
    print("- This isolates mechanisms for BELIEF TRACKING vs LOCATION RETRIEVAL")

if __name__ == "__main__":
    analyze_saved_abstract_results()