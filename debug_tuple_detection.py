#!/usr/bin/env python3
"""Debug tuple detection in evaluation logic."""

def test_tuple_detection():
    # Test cases from the error log
    causal_ans = ('shelf', 'window')
    original_ans = 'kitchen'  # Or could also be a tuple
    
    print(f"causal_ans: {causal_ans}, type: {type(causal_ans)}")
    print(f"original_ans: {original_ans}, type: {type(original_ans)}")
    
    print(f"isinstance(causal_ans, tuple): {isinstance(causal_ans, tuple)}")
    print(f"isinstance(original_ans, tuple): {isinstance(original_ans, tuple)}")
    
    # The condition from evaluation.py
    condition = isinstance(causal_ans, tuple) and isinstance(original_ans, tuple)
    print(f"Dual tuple condition: {condition}")
    
    if condition:
        print("Would call calculate_dual_logit_differences")
    else:
        print("Would call to_single_token - THIS IS THE PROBLEM!")
        
    # Test with both as tuples
    print("\n--- Test with both as tuples ---")
    original_ans_tuple = ('kitchen', 'garden')
    condition2 = isinstance(causal_ans, tuple) and isinstance(original_ans_tuple, tuple)
    print(f"Both tuples condition: {condition2}")

if __name__ == "__main__":
    test_tuple_detection()