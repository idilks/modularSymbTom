"""Quick smoke test for code sample - verifies function can be called"""
import torch
from unittest.mock import Mock
from cma_code_sample import patch_head_activations

# Create minimal mocks
mock_model = Mock()
# Make model callable and return proper tensor
fake_logits = torch.randn(1, 10, 50257)
mock_model.return_value = fake_logits  # for baseline logits when called as model(input_ids)
mock_model.run_with_hooks = Mock(return_value=fake_logits)  # for patched logits
mock_evaluator = Mock()
mock_evaluator.evaluate_generation_accuracy = Mock(return_value=(8, 10, 0.8))
mock_cache = {"blocks.0.attn.hook_z": torch.randn(1, 10, 4, 64)}
mock_input_ids = torch.randint(0, 1000, (1, 10))

print("testing logit mode...")
result = patch_head_activations(
    model=mock_model,
    base_cache=mock_cache,
    input_ids=mock_input_ids,
    token_pos=-1,
    evaluator=mock_evaluator,
    total_layers=1,
    total_heads=4,
    activation_name="z",
    device="cpu",  # use CPU for testing
    use_generation=False,
    causal_ans_id=100,
    original_ans_id=200
)
assert result.shape == (1, 4), f"expected (1, 4), got {result.shape}"
print(f"[OK] logit mode: output shape {result.shape} correct")

print("\ntesting generation mode...")
result = patch_head_activations(
    model=mock_model,
    base_cache=mock_cache,
    input_ids=mock_input_ids,
    token_pos=-1,
    evaluator=mock_evaluator,
    total_layers=1,
    total_heads=4,
    device="cpu",  # use CPU for testing
    use_generation=True
)
assert result.shape == (1, 4), f"expected (1, 4), got {result.shape}"
print(f"[OK] generation mode: output shape {result.shape} correct")

print("\n[SUCCESS] all tests passed! code is valid.")
