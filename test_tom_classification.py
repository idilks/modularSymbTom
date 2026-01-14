"""test error classification logic"""
from tom_error_classification import (
    extract_dual_answers, classify_belief_prediction,
    classify_world_prediction, normalize_loc
)

# test extraction
text = "where the agent thinks: <loc>kitchen</loc>\nactual location: <loc>garden</loc>"
belief, world, malformed = extract_dual_answers(text)
assert belief == "kitchen" and world == "garden" and not malformed
print(f"[OK] extraction: belief={belief}, world={world}")

# test belief classification
assert classify_belief_prediction("kitchen", "kitchen", "garden") == "correct"
assert classify_belief_prediction("garden", "kitchen", "garden") == "opposite"
assert classify_belief_prediction("bathroom", "kitchen", "garden") == "third"
print("[OK] belief classification")

# test world classification - the key insight!
assert classify_world_prediction("garden", "garden", "kitchen") == "correct"
assert classify_world_prediction("kitchen", "garden", "kitchen") == "belief_confusion"
assert classify_world_prediction("bathroom", "garden", "kitchen") == "third"
print("[OK] world classification (belief_confusion detection)")

# test normalization
assert normalize_loc("the Kitchen") == "kitchen"
assert normalize_loc("  Main Street  ") == "main street"
print("[OK] normalization")

print("\n[SUCCESS] all classification logic verified!")
