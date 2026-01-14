"""Activation Patching for Causal Mediation Analysis in Transformers"""

import torch
from functools import partial
from tqdm import tqdm
import transformer_lens.utils as utils

def patch_head_activations(
    model, base_cache, input_ids: torch.Tensor, token_pos, evaluator,
    total_layers: int, total_heads: int, activation_name: str = "z",
    device: str = "cuda", use_generation: bool = True,
    causal_ans_id: int = None, original_ans_id: int = None, **eval_kwargs
):
    """
    Systematic activation patching across all layers and heads.
    
    Methodology: Cache activations from base context, then for each layer/head,
    run experimental context and replace exp activations with base activations
    during forward pass. Measure prediction change to identify causal importance.
    
    Args:
        base_cache: Cached activations from base forward pass
        input_ids: Experimental context tokens
        token_pos: Position(s) to patch - int or (base_pos, exp_pos) tuple
        activation_name: Which activation (z=attn output, q/k/v=queries/keys/values)
        use_generation: True=generation accuracy, False=logit differences
        causal_ans_id: Token ID of answer we expect after patching (for logit mode)
        original_ans_id: Token ID of original answer before patching (for logit mode)
    
    Returns:
        accuracy_matrix or causal_effect_matrix (layers × heads)
    """

    # handle flexible token position formats
    if isinstance(token_pos, tuple) and len(token_pos) == 2:
        base_pos, exp_pos = token_pos  # different positions per context
    else:
        base_pos = exp_pos = token_pos  # same position in both

    def replace_head_activation_hook(activation, hook, head_idx, token_len, group_size=1):
        """Hook that performs surgical activation replacement."""
        activation_tag = hook.name  # e.g., "blocks.5.attn.hook_z"

        if activation.shape[1] == token_len:  # safety: verify sequence length
            # extract from base context at base_pos
            base_head_act = base_cache[activation_tag][
                :, base_pos, head_idx * group_size : (head_idx + 1) * group_size, :
            ]
            # inject into exp context at exp_pos
            activation[:, exp_pos, head_idx * group_size : (head_idx + 1) * group_size, :] = base_head_act

    token_len = input_ids.shape[-1]

    # allocate results
    if use_generation:
        accuracy_matrix = torch.zeros((total_layers, total_heads), device=device)
    else:
        # compute baseline logit difference (unpatched forward pass)
        baseline_logits = model(input_ids)
        baseline_diff = (baseline_logits[0, -1, causal_ans_id] -
                        baseline_logits[0, -1, original_ans_id])
        causal_effect_matrix = torch.zeros((total_layers, total_heads), device=device)

    # systematic iteration: test EACH head independently to isolate causal contributions
    for layer_idx in tqdm(range(total_layers), desc="Patching layers"):
        activation_tag = utils.get_act_name(activation_name, layer_idx)

        for head_idx in range(total_heads):
            hook_fn = partial(replace_head_activation_hook, head_idx=head_idx,
                            token_len=token_len, group_size=1)

            if use_generation:
                # evaluate via generation: run forward pass with hook, check correctness
                _, _, accuracy = evaluator.evaluate_generation_accuracy(
                    model, input_ids, fwd_hooks=[(activation_tag, hook_fn)], **eval_kwargs
                )
                accuracy_matrix[layer_idx, head_idx] = accuracy
            else:
                # evaluate via logit differences
                patched_logits = model.run_with_hooks(input_ids, fwd_hooks=[(activation_tag, hook_fn)])
                # causal mediation score: how much did patching change the logit difference?
                # positive = patching pushed toward causal answer (head is important)
                # negative = patching pushed away from causal answer (head opposes behavior)
                patched_diff = (patched_logits[0, -1, causal_ans_id] -
                               patched_logits[0, -1, original_ans_id])
                causal_effect = patched_diff - baseline_diff
                causal_effect_matrix[layer_idx, head_idx] = causal_effect

    return accuracy_matrix if use_generation else causal_effect_matrix