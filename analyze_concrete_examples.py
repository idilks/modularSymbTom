#!/usr/bin/env python3
"""
Analyze concrete abstract context examples from saved results.
"""

def analyze_concrete_examples():
    """Analyze the actual saved abstract context examples."""
    
    print("=== CONCRETE ABSTRACT CONTEXT EXAMPLES ===\n")
    
    # Read the actual saved examples
    base_file = "/mnt/c/Users/idilk/Desktop/tom_dev/codebase/results/identity_rules/cma/meta-llama/Llama-3.2-1B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_False/base_input_prompts_1.txt"
    exp_file = "/mnt/c/Users/idilk/Desktop/tom_dev/codebase/results/identity_rules/cma/meta-llama/Llama-3.2-1B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_False/exp_input_prompts_1.txt"
    ans_file = "/mnt/c/Users/idilk/Desktop/tom_dev/codebase/results/identity_rules/cma/meta-llama/Llama-3.2-1B/abstract_context/base_rule_ABA_exp_rule_ABB/z_seed_0_shuffle_False/ans_1.txt"
    
    with open(base_file, 'r') as f:
        base_content = f.read().strip()
    with open(exp_file, 'r') as f:
        exp_content = f.read().strip()
    with open(ans_file, 'r') as f:
        ans_content = f.read().strip()
    
    print("DIRECTORY NAME: base_rule_ABA_exp_rule_ABB")
    print("-> This means: ABA is base rule, ABB is experimental rule")
    print("-> ABA = false belief baseline, ABB = true belief experimental\n")
    
    print("BASE PROMPT (ABA = false belief scenario):")
    print("-" * 50)
    print(base_content)
    print()
    
    print("EXP PROMPT (ABB = true belief scenario):")
    print("-" * 50)
    print(exp_content)
    print()
    
    print("ANSWERS:")
    print("-" * 50)
    print(ans_content)
    print()
    
    # Parse the scenarios
    print("SCENARIO ANALYSIS:")
    print("-" * 50)
    
    base_lines = base_content.split('\n')
    exp_lines = exp_content.split('\n')
    
    base_scenario = base_lines[0].replace('→', '').strip()
    exp_scenario = exp_lines[0].replace('→', '').strip()
    
    print(f"BASE: {base_scenario}")
    print(f"EXP:  {exp_scenario}")
    print()
    
    # Identify key difference
    print("KEY DIFFERENCE:")
    print("-" * 50)
    if "agent leaves room. object moved" in base_scenario:
        print("BASE = FALSE BELIEF: Agent leaves BEFORE object moves")
        print("   -> Agent doesn't see the move")
        print("   -> Agent has false belief about object location")
    
    if "object moved" in exp_scenario and "agent leaves room" in exp_scenario:
        exp_move_pos = exp_scenario.find("object moved")
        exp_leave_pos = exp_scenario.find("agent leaves room")
        if exp_move_pos < exp_leave_pos:
            print("EXP = TRUE BELIEF: Object moves BEFORE agent leaves") 
            print("   -> Agent sees the move")
            print("   -> Agent has true belief about object location")
    print()
    
    # Parse answers
    ans_lines = ans_content.split('\n')
    for line in ans_lines:
        if "base_ans:" in line and "exp_ans:" in line and "causal_ans:" in line:
            line = line.replace('→', '').replace('0: ', '')
            parts = line.split(', ')
            
            base_ans = parts[0].split('base_ans: ')[1]
            exp_ans = parts[1].split('exp_ans: ')[1] 
            causal_ans = parts[2].split('causal_ans: ')[1]
            
            print("ANSWER ANALYSIS:")
            print("-" * 50)
            print(f"BASE answer (false belief): {base_ans}")
            print(f"   -> Agent thinks object is at original location")
            print(f"EXP answer (true belief): {exp_ans}")
            print(f"   -> Agent knows object moved to new location")
            print(f"CAUSAL expected: {causal_ans}")
            print(f"   -> After patching, should get BASE belief state")
            print()
    
    print("CAUSAL MEDIATION HYPOTHESIS:")
    print("-" * 50)
    print("1. BASE condition: false belief scenario")
    print("2. EXP condition: true belief scenario")  
    print("3. Patch activations from BASE -> EXP")
    print("4. Expected result: EXP should now give BASE answers")
    print("5. This tests: do attention heads track BELIEF STATES?")
    print()
    print("If heads track belief states:")
    print(f"   -> Patching should preserve false belief: {base_ans}")
    print("If heads only track locations:")
    print(f"   -> Patching should not change behavior")

if __name__ == "__main__":
    analyze_concrete_examples()