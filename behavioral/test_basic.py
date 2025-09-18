"""
Simple test of behavioral evaluation imports and basic functionality.
"""

import os
import sys

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'codebase'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'codebase', 'tasks', 'identity_rules'))

try:
    from behavioral_config import BehavioralConfig, DefaultBehavioralConfig
    print("SUCCESS: behavioral_config imported successfully")
    
    # Test config creation
    config = DefaultBehavioralConfig.quick_test()
    config.models = ["gpt2"]  # Use simple model for testing
    config.samples_per_condition = 3
    config.prompt_num = 3
    print(f"SUCCESS: Config created: {config.models}, {config.samples_per_condition} samples")
    
    from behavioral_utils import BehavioralEvaluator, WandbLogger, ResultsSaver
    print("SUCCESS: behavioral_utils imported successfully")
    
    # Test utility creation
    evaluator = BehavioralEvaluator(config)
    print("SUCCESS: BehavioralEvaluator created")
    
    wandb_logger = WandbLogger(config)
    print("SUCCESS: WandbLogger created (disabled)")
    
    results_saver = ResultsSaver(config)
    print("SUCCESS: ResultsSaver created")
    
    print("\nALL TESTS PASSED: behavioral evaluation ready!")
    print("\nConfiguration:")
    print(f"  Models: {config.models}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Vignette types: {config.vignette_types}")
    print(f"  TOM formats: {config.tom_formats}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()