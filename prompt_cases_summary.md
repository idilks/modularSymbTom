# Prompt Cases Summary for CMA Experiments

## Overview
The codebase generates minimal pairs for causal mediation analysis (CMA) using two context types: **abstract** and **control**. Each targets different cognitive mechanisms in transformer models through carefully designed prompt pairs.

## ABA/ABB Logic Explained

The `--base_rule` parameter determines which belief state serves as the baseline condition:

- **ABA Pattern**: Agent expects A (original location), but object is at B (moved location), answer should be A (false belief)
- **ABB Pattern**: Agent knows B (moved location), object is at B, answer should be B (true belief)

When `--base_rule ABA`:
- BASE = False belief scenario (agent misses movement)  
- EXP = True belief scenario (agent witnesses movement)

When `--base_rule ABB`:
- BASE = True belief scenario (agent witnesses movement)
- EXP = False belief scenario (agent misses movement)

## Context Types

### 1. Abstract Context (`--context_type abstract`)

**Purpose**: Tests belief tracking vs world state reasoning  
**Hypothesis**: Patching belief tracking heads should convert true belief reasoning to false belief reasoning

**How it works**:
- Same locations in both conditions, different belief states
- Timing of agent departure determines false vs true belief
- Tests mechanisms that track **agent's belief state**

**Real Example from `Qwen2.5-14B-Instruct_food_truck_z_ABA_control.json`**:
```
BASE (ABA = False Belief):
"object at fireplace. agent leaves room. object moved to radiator.
agent returns and looks where?"
Expected: fireplace (agent's false belief)

EXP (ABB = True Belief):  
"object at fireplace. object moved to radiator. agent leaves room.
agent returns and looks where?"
Expected: radiator (agent's true belief)

Causal Prediction: After patching belief tracking heads, 
exp should answer "fireplace" (converted to false belief)
```

### 2. Control Context (`--context_type control`)

**Purpose**: Methodological control for validating causal claims  
**Hypothesis**: Should show minimal causal effects (negative control)

**How it works**:
- Both base and exp use identical scenario structure
- Only the specific tokens (colors, locations) are swapped between conditions
- Tests whether patching effects are due to general disruption vs specific mechanisms

**Real Example from `Qwen2.5-14B-Instruct_hair_styling_z_ABA_control.json`**:
```
BASE:
"Alice has topaz hair. Alice dyes her hair orange but avoids telling Jennifer.
What color would Jennifer say Alice's hair is?"
Expected: topaz (Jennifer's belief about original color)

EXP:
"Alice has orange hair. Alice dyes her hair topaz but avoids telling Jennifer.  
What color would Jennifer say Alice's hair is?"
Expected: orange (Jennifer's belief about original color)

Causal Prediction: Minimal patching effects
(both scenarios have identical false belief structure)
```

## Scenario Templates

The system uses 5 scenario types with rich vocabulary:

### Food Truck (Mobile Object)
```
False Belief: "Alice sees a food truck selling tacos on Main Street. 
Alice drives away. The truck moves to Oak Avenue while Alice is gone."

True Belief: "Alice sees a food truck selling tacos on Main Street.
The truck moves to Oak Avenue. Alice sees it moving."
```

### Hair Styling (Hidden Change)
```
False Belief: "Alice dyes her hair blonde but doesn't tell Jennifer."
True Belief: "Alice dyes her hair blonde and immediately calls Jennifer."
```

### Library Book (Object Transfer)
```
False Belief: "Alice puts a book on the history shelf. Alice leaves.
Bob moves the book to the science shelf."

True Belief: "Alice puts a book on the history shelf. 
Bob moves the book to the science shelf. Alice sees Bob moving it."
```

## Prompt Generation Pipeline

1. **Template Selection**: Choose from 5 scenario types
2. **Vocabulary Sampling**: Random locations, foods, names from curated lists
3. **Variable Substitution**: Fill template with sampled vocabulary
4. **Context Application**: Apply abstract/token/control logic
5. **Question Formatting**: Add belief-based or world-state questions
6. **Chat Template**: Wrap in instruction format for instruct models

## Output Format

Models respond with location in `<loc></loc>` tags:
```
Human: Where would Alice look for the object?
Model: Alice would look for the object at <loc>kitchen</loc>
```

## Key Differences Summary

| Aspect | Abstract | Control |
|--------|----------|---------|
| **Scenario Structure** | Different (FB vs TB) | Identical (both FB) |
| **Question Type** | Belief-based | Belief-based |
| **Target Mechanism** | Belief tracking | Negative control |
| **Expected Patching** | Belief→World confusion | Minimal effects |
| **Token Manipulation** | Same tokens, different order | Different tokens, same structure |

## Causal Mediation Logic

**Abstract Context**: 
- Identifies attention heads that track agent beliefs vs world state
- Patching should cause models to confuse what agents know vs what's true
- Tests whether models have distinct representations for belief states vs world facts

**Control Context**:
- Validates that observed effects aren't due to non-specific disruption or token-level differences
- Should show minimal causal effects if belief tracking mechanisms are truly specific
- Both conditions use identical false belief structure with swapped tokens

**Experimental Pipeline**:
1. **Abstract Context**: Test if heads specifically track belief states (expect large effects)
2. **Control Context**: Verify effects aren't due to general patching disruption (expect minimal effects)
3. **Comparison**: Abstract effects >> Control effects = evidence for specific belief tracking mechanisms

This design enables precise isolation of belief tracking mechanisms through minimal pair comparisons that control for tokenization, surface structure, and general patching effects.