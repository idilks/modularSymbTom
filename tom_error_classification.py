"""
error classification for theory of mind dual-answer evaluation

key insight: not all failures are equal. distinguishes between:
- belief_confusion: model conflates agent's belief with actual world state
- opposite: systematic error (wrong but predictable)
- third: random error (neither expected answer)

this reveals WHAT the model misunderstands, not just THAT it's wrong.
"""

import re
from typing import Tuple


def normalize_loc(location: str) -> str:
    """normalize location strings: lowercase, remove articles, strip whitespace."""
    if not location:
        return ""

    normalized = location.strip().lower()

    # remove leading articles (models often add "the", "a", "an")
    for article in ["the ", "a ", "an "]:
        if normalized.startswith(article):
            normalized = normalized[len(article):]
            break

    return normalized


def locations_match(predicted: str, expected: str) -> bool:
    """fuzzy matching for locations. handles "Main Street" vs "on Main Street"."""
    if not predicted or not expected:
        return False

    pred_norm = normalize_loc(predicted)
    expected_norm = normalize_loc(expected)

    # exact match or substring match (handles prepositional phrases)
    return (pred_norm == expected_norm or
            expected_norm in pred_norm or
            pred_norm in expected_norm)


def extract_dual_answers(text: str) -> Tuple[str, str, bool]:
    """
    parse dual-answer format from model generation.

    schema: "where agent thinks: <loc>X</loc>\nactual location: <loc>Y</loc>"
    returns: (belief_answer, world_answer, malformed)

    design: signal-based regex on semantic keywords ("thinks" vs "actual")
    rather than rigid positional parsing. robust to phrasing variation.
    """
    belief_pattern = re.compile(r"(?:agent thinks|where.*thinks).*?<loc>(.*?)</loc>", re.I | re.DOTALL)
    world_pattern = re.compile(r"(?:actual|real).*?<loc>(.*?)</loc>", re.I | re.DOTALL)

    belief_match = belief_pattern.search(text)
    world_match = world_pattern.search(text)

    belief = belief_match.group(1).strip() if belief_match else ""
    world = world_match.group(1).strip() if world_match else ""

    malformed = (belief == "" or world == "")  # model couldn't follow schema
    return belief, world, malformed


def classify_belief_prediction(predicted: str, expected_belief: str, alternative_belief: str) -> str:
    """
    classify belief prediction errors.

    - "correct": tracks agent's mental state
    - "opposite": systematic error (gives alternative location)
    - "third": random error (gives neither location)
    """
    predicted = normalize_loc(predicted)
    expected = normalize_loc(expected_belief)
    alternative = normalize_loc(alternative_belief)

    if locations_match(predicted, expected):
        return "correct"
    elif locations_match(predicted, alternative):
        return "opposite"
    else:
        return "third"


def classify_world_prediction(predicted: str, expected_world: str, expected_belief: str) -> str:
    """
    classify world state prediction - KEY INSIGHT for theory of mind.

    when asking "where is object actually?", if model answers with agent's belief
    instead of actual location, that's "belief_confusion" - model conflates mental
    states with physical reality.

    - "correct": reports actual world state
    - "belief_confusion": reports agent's belief (CRITICAL - reveals no theory of mind)
    - "third": random error

    belief_confusion rate measures whether model maintains separate representations
    for beliefs vs reality. enables targeted analysis: which heads cause confusion?
    patching those heads should reduce confusion if causally responsible.
    """
    predicted = normalize_loc(predicted)
    world_state = normalize_loc(expected_world)
    belief_state = normalize_loc(expected_belief)

    if locations_match(predicted, world_state):
        return "correct"
    elif locations_match(predicted, belief_state):
        return "belief_confusion"  # conflates belief with reality
    else:
        return "third"
