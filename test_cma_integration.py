#!/usr/bin/env python3
"""
Test script for CMA-compatible unified templates.
Verifies that generate_cma_pair() produces properly tagged scenarios.
"""

import sys
import os
sys.path.append('behavioral')

from unified_templates import UnifiedTemplateSystem, UnifiedQuestionFormatter

def test_cma_pair_generation():
    """Test CMA pair generation with location tagging."""
    print("Testing CMA pair generation...")
    
    template_system = UnifiedTemplateSystem("tom_datasets")
    question_formatter = UnifiedQuestionFormatter()
    
    # Test basic object move scenario
    template_name = "basic_object_move"
    print(f"\nTesting template: {template_name}")
    
    try:
        (false_belief_scenario, belief_ans, world_ans), (true_belief_scenario, exp_belief, exp_world) = \
            template_system.generate_cma_pair(template_name)
        
        print(f"False belief scenario: {false_belief_scenario}")
        print(f"True belief scenario: {true_belief_scenario}")
        print(f"Belief answer: {belief_ans}")
        print(f"World answer: {world_ans}")
        
        # Check for <loc> tags
        has_loc_tags = "<loc>" in false_belief_scenario and "</loc>" in false_belief_scenario
        print(f"Has <loc> tags: {has_loc_tags}")
        
        # Test CMA question
        cma_question = question_formatter.get_cma_questions(template_name)
        print(f"CMA question: {cma_question}")
        
        # Full prompt example
        full_prompt = f"{false_belief_scenario}\n{cma_question}"
        print(f"\nFull CMA prompt:\n{full_prompt}")
        
        return True
        
    except Exception as e:
        print(f"Error testing {template_name}: {e}")
        return False

def test_multiple_templates():
    """Test multiple template types."""
    template_system = UnifiedTemplateSystem("tom_datasets") 
    question_formatter = UnifiedQuestionFormatter()
    
    templates = ["basic_object_move", "food_truck"]
    question_styles = ["instruction", "completion"]
    
    for template_name in templates:
        print(f"\n{'='*50}")
        print(f"Testing template: {template_name}")
        
        try:
            (false_belief, _, _), (true_belief, _, _) = template_system.generate_cma_pair(template_name)
            
            print(f"Scenario: {false_belief[:100]}...")
            print(f"Has tags: {'<loc>' in false_belief}")
            
            # Test both question styles
            for style in question_styles:
                question = question_formatter.get_cma_questions(template_name, style)
                print(f"{style.title()} question: {question}")
            
        except Exception as e:
            print(f"Error with {template_name}: {e}")

def test_question_styles():
    """Compare completion vs instruction question styles."""
    print(f"\n{'='*60}")
    print("COMPARING QUESTION STYLES")
    print("="*60)
    
    template_system = UnifiedTemplateSystem("tom_datasets")
    question_formatter = UnifiedQuestionFormatter()
    
    template_name = "basic_object_move"
    (scenario, _, _), _ = template_system.generate_cma_pair(template_name)
    
    print(f"Scenario: {scenario}")
    print()
    
    styles = ["instruction", "completion"]
    for style in styles:
        question = question_formatter.get_cma_questions(template_name, style)
        full_prompt = f"{scenario}\n{question}"
        print(f"{style.upper()} APPROACH:")
        print(f"Question: {question}")
        print(f"Full prompt: {full_prompt}")
        print(f"Expected model response pattern: {'natural completion' if style == 'completion' else 'explicit format following'}")
        print("-" * 40)

if __name__ == "__main__":
    print("Testing CMA integration with unified templates")
    print("=" * 60)
    
    # Test basic functionality
    success = test_cma_pair_generation()
    
    if success:
        print("\n" + "="*60)
        print("Testing multiple templates...")
        test_multiple_templates()
        
        # Test question style comparison
        test_question_styles()
    
    print("\nDone!")