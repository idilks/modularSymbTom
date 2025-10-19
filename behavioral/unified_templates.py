"""
Unified template system for all theory of mind scenarios.
Replaces both ScenarioBuilder and NaturalisticScenarios with a single approach.
"""

import random
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ScenarioTemplate:
    """Template for generating ToM scenarios with variable substitution."""
    name: str
    false_belief_template: str
    true_belief_template: str
    variables: Dict[str, str]  # variable_name -> vocab_file_name
    belief_answer_key: str
    world_answer_key: str
    description: str = ""


class VocabularyLoader:
    """Loads vocabulary from txt files."""
    
    def __init__(self, vocab_dir: str = "tom_datasets"):
        self.vocab_dir = vocab_dir
        self.vocabularies = {}
        self._load_all_vocabularies()
    
    def _load_all_vocabularies(self):
        """Load all vocabulary files from directory."""
        vocab_files = {
            "locations": "locations.txt",
            "foods": "foods.txt", 
            "streets": "streets.txt",
            "restaurants": "restaurants.txt",
            "gift_items": "gift_items.txt",
            "book_types": "book_types.txt",
            "library_sections": "library_sections.txt",
            "hair_lengths": "hair_lengths.txt",
            "times": "times.txt",
            "reasons": "reasons.txt"
        }
        
        for vocab_name, filename in vocab_files.items():
            filepath = os.path.join(self.vocab_dir, filename)
            self.vocabularies[vocab_name] = self._load_vocab_file(filepath, vocab_name)
    
    def _load_vocab_file(self, filepath: str, vocab_name: str) -> List[str]:
        """Load vocabulary from single file with fallbacks."""
        try:
            with open(filepath, 'r') as f:
                vocab = [line.strip() for line in f if line.strip()]
            if vocab:
                return vocab
        except FileNotFoundError:
            pass
        
        # Fallback vocabularies if files don't exist. currently disabled.
        # fallbacks = {
        #     "locations": ["under the table", "next to the shelf", "in the kitchen", "on the counter", 
        #                  "behind the door", "in the garden", "near the window", "by the fireplace"],
        #     "foods": ["tacos", "burgers", "pizza", "hot dogs", "sandwiches", "ice cream"],
        #     "streets": ["Main Street", "Oak Avenue", "Park Boulevard", "First Street", "Downtown Plaza"],
        #     "restaurants": ["Luigi's", "The Garden", "Sakura", "Blue Plate", "Café Roma"],
        #     "gift_items": ["earrings", "a necklace", "a headband", "a hat", "sunglasses", "a scarf"],
        #     "book_types": ["history", "science", "art", "philosophy", "psychology", "literature"],
        #     "library_sections": ["fiction", "non-fiction", "reference", "new arrivals", "academic"],
        #     "hair_lengths": ["long", "short"],
        #     "times": ["7:00 PM", "6:30 PM", "8:00 PM", "7:30 PM", "8:30 PM"],
        #     "reasons": ["kitchen delay", "staff shortage", "large party cancellation"]
        # }
        
        # return fallbacks.get(vocab_name, [f"default_{vocab_name}"])
    
    def get_vocab(self, vocab_name: str) -> List[str]:
        """Get vocabulary by name."""
        return self.vocabularies.get(vocab_name, [])
    
    def sample_different(self, vocab_name: str, count: int = 2) -> List[str]:
        """Sample different items from vocabulary."""
        vocab = self.get_vocab(vocab_name)
        if len(vocab) < count:
            # If not enough items, pad with repetition
            return random.choices(vocab, k=count)
        return random.sample(vocab, count)


class UnifiedTemplateSystem:
    """Unified system for all ToM scenario generation."""
    
    def __init__(self, vocab_dir: str = "tom_datasets"):
        self.vocab_loader = VocabularyLoader(vocab_dir)
        self.templates = self._create_all_templates()
    
    def _create_all_templates(self) -> Dict[str, ScenarioTemplate]:
        """Create all scenario templates."""
        return {
            "basic_object_move": ScenarioTemplate(
                name="basic_object_move",
                false_belief_template="object at {location_1}. agent leaves room. object moved to {location_2}.",
                true_belief_template="object at {location_1}. object moved to {location_2}. agent leaves room.",
                variables={
                    "location_1": "locations",
                    "location_2": "locations"
                },
                belief_answer_key="location_1",
                world_answer_key="location_2",
                description="Basic object movement scenario"
            ),
            
            "basic_object_move_detailed": ScenarioTemplate(
                name="basic_object_move_detailed",
                false_belief_template=(
                    "Alice places an object at {location_1}. Alice leaves the room and goes outside. "
                    "While Alice is away, John moves the object from {location_1} to {location_2}. "
                    "The object stays there and nobody moves it again."
                ),
                true_belief_template=(
                    "Alice places an object at {location_1}. John moves the object to {location_2}. "
                    "Alice sees John move the object. Alice leaves the room and goes outside. "
                    "The object stays there and nobody moves it again."
                ),
                variables={
                    "location_1": "locations",
                    "location_2": "locations"
                },
                belief_answer_key="location_1",
                world_answer_key="location_2",
                description="Detailed basic object movement scenario"
            ),
            
            "food_truck": ScenarioTemplate(
                name="food_truck",
                false_belief_template=(
                    "Alice is driving past a food truck selling <loc>{food}</loc>, on <loc>{street_1}</loc>. "
                    "She drives past the food truck and continues down the road. "
                    "After Alice turns towards a different street, she is unable to see the truck "
                    "but makes a note to herself to grab a plate of <loc>{food}</loc> later. "
                    "After Alice is out of sight, Bob, the owner of the food truck, "
                    "drives the truck to <loc>{street_2}</loc> to stay until the end of the day. "
                    "After this, the food truck stays in place, nobody moves it."
                ),
                true_belief_template=(
                    "Alice is driving past a food truck selling <loc>{food}</loc>, on <loc>{street_1}</loc>. "
                    "Bob, the owner of the food truck, drives the truck to <loc>{street_2}</loc> "
                    "to stay until the end of the day. Alice sees Bob moving the truck. "
                    "After Alice turns towards a different street, she makes a note to herself "
                    "to grab a plate of <loc>{food}</loc> later from the new location. "
                    "She drives past the food truck and continues down the road. "
                    "After this, the food truck stays in place, nobody moves it."
                ),
                variables={
                    "food": "foods",
                    "street_1": "streets", 
                    "street_2": "streets"
                },
                belief_answer_key="street_1",
                world_answer_key="street_2",
                description="Food truck location change scenario"
            ),
            
            "hair_styling": ScenarioTemplate(
                name="hair_styling",
                false_belief_template=(
                    "Alice and Jennifer are good friends. On Monday, Alice decides to cut her long hair short, "
                    "but the process goes terribly and she avoids telling this to Jennifer and avoids all her friends "
                    "and does not go outside. Tuesday is Alice's birthday and Jennifer is out shopping for "
                    "{gift_item} as a birthday gift for Alice. As hair length affects how accessories look, "
                    "the sales assistant asks how long Alice's hair is."
                ),
                true_belief_template=(
                    "Alice and Jennifer are good friends. On Monday, Alice decides to cut her long hair short, "
                    "but the process goes terribly. She immediately calls Jennifer to tell her about the disaster. "
                    "Tuesday is Alice's birthday and Jennifer is out shopping for "
                    "{gift_item} as a birthday gift for Alice. As hair length affects how accessories look, "
                    "the sales assistant asks how long Alice's hair is."
                ),
                variables={
                    "gift_item": "gift_items"
                },
                belief_answer_key="long",  # Jennifer thinks Alice still has long hair
                world_answer_key="short",  # Alice actually has short hair
                description="Hair styling false belief scenario"
            ),
            
            "library_book": ScenarioTemplate(
                name="library_book",
                false_belief_template=(
                    "Sarah checks out a <loc>{book_type}</loc> book from the <loc>{section_1}</loc> section of the library. "
                    "She sits down and reads the book in the desk area in <loc>{section_1}</loc>. Later, she finishes the book and leaves to get coffee from the café downstairs before continuing her work. "
                    "While she's away, a librarian sees the book and, thinking it was left behind accidentally, "
                    "returns it to the <loc>{section_2}</loc> section where it actually belongs. "
                    "The librarian reshelves it properly and it stays there."
                ),
                true_belief_template=(
                    "Sarah checks out a <loc>{book_type}</loc> book from the <loc>{section_1}</loc> section of the library. "
                    "She sits down and reads the book in the desk area in <loc>{section_1}</loc>. Sarah sees a librarian approaching and explains "
                    "she's just getting coffee. The librarian mentions the book should be in <loc>{section_2}</loc> section "
                    "and offers to move it there. Sarah agrees and watches the librarian take the book to "
                    "<loc>{section_2}</loc> section. Sarah then leaves to get coffee."
                ),
                variables={
                    "book_type": "book_types",
                    "section_1": "library_sections",
                    "section_2": "library_sections"
                },
                belief_answer_key="section_1",
                world_answer_key="section_2",
                description="Library book relocation scenario"
            ),
            
            "restaurant_reservation": ScenarioTemplate(
                name="restaurant_reservation",
                false_belief_template=(
                    "Sarah makes a reservation at <loc>{restaurant}</loc> for <loc>{time_1}</loc>. "
                    "She tells Tom about the dinner plans. Later, the restaurant calls "
                    "and changes the reservation to <loc>{time_2}</loc> due to <loc>{reason}</loc>. "
                    "Sarah is in a meeting and misses the call. The restaurant leaves a voicemail "
                    "but Sarah doesn't check it before meeting Tom."
                ),
                true_belief_template=(
                    "Sarah makes a reservation at <loc>{restaurant}</loc> for <loc>{time_1}</loc>. "
                    "She tells Tom about the dinner plans. Later, the restaurant calls "
                    "and changes the reservation to <loc>{time_2}</loc> due to <loc>{reason}</loc>. "
                    "Sarah answers the call and immediately texts Tom about the change."
                ),
                variables={
                    "restaurant": "restaurants",
                    "time_1": "times",
                    "time_2": "times", 
                    "reason": "reasons"
                },
                belief_answer_key="time_1",
                world_answer_key="time_2",
                description="Restaurant reservation time change scenario"
            )
        }
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())
    
    def get_template(self, template_name: str) -> ScenarioTemplate:
        """Get template by name."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {self.list_templates()}")
        return self.templates[template_name]
    
    def generate_scenario(
        self, 
        template_name: str, 
        vignette_type: str
    ) -> Tuple[str, str, str]:
        """Generate scenario with random variable substitution."""
        
        template = self.get_template(template_name)
        
        # Sample variables ensuring different values where needed
        substitutions = {}
        for var_name, vocab_name in template.variables.items():
            if var_name.endswith("_1") and var_name.replace("_1", "_2") in template.variables:
                # Handle paired variables (location_1, location_2) to ensure they're different
                base_name = var_name.replace("_1", "")
                if base_name + "_1" not in substitutions:
                    val_1, val_2 = self.vocab_loader.sample_different(vocab_name, 2)
                    substitutions[base_name + "_1"] = val_1
                    substitutions[base_name + "_2"] = val_2
            elif not var_name.endswith("_2"):
                # Single variable or non-paired variable
                vocab = self.vocab_loader.get_vocab(vocab_name)
                substitutions[var_name] = random.choice(vocab)
        
        # Generate scenario text
        if vignette_type == "false_belief":
            scenario_text = template.false_belief_template.format(**substitutions)
        else:  # true_belief
            scenario_text = template.true_belief_template.format(**substitutions)
        
        # Get expected answers
        if template.belief_answer_key in substitutions:
            belief_answer = substitutions[template.belief_answer_key]
        else:
            # Handle literal values like "long", "short"
            belief_answer = template.belief_answer_key
        
        if template.world_answer_key in substitutions:
            world_answer = substitutions[template.world_answer_key]  
        else:
            world_answer = template.world_answer_key
        
        # For true belief, both answers should match the world state
        if vignette_type == "true_belief":
            belief_answer = world_answer
        
        return scenario_text, belief_answer, world_answer
    
    def generate_cma_pair(self, template_name: str) -> Tuple[Tuple[str, str, str], Tuple[str, str, str]]:
        """Generate contrastive pair for causal mediation analysis.
        
        Returns:
            ((false_belief_scenario, belief_ans, world_ans), (true_belief_scenario, belief_ans, world_ans))
        """
        template = self.get_template(template_name)
        
        # Sample variables once for both scenarios
        substitutions = {}
        for var_name, vocab_name in template.variables.items():
            if var_name.endswith("_1") and var_name.replace("_1", "_2") in template.variables:
                # Handle paired variables (location_1, location_2) to ensure they're different
                base_name = var_name.replace("_1", "")
                if base_name + "_1" not in substitutions:
                    val_1, val_2 = self.vocab_loader.sample_different(vocab_name, 2)
                    substitutions[base_name + "_1"] = val_1
                    substitutions[base_name + "_2"] = val_2
            elif not var_name.endswith("_2"):
                # Single variable or non-paired variable
                vocab = self.vocab_loader.get_vocab(vocab_name)
                substitutions[var_name] = random.choice(vocab)
        
        # Generate both scenarios with same variables
        false_belief_scenario = template.false_belief_template.format(**substitutions)
        true_belief_scenario = template.true_belief_template.format(**substitutions)
        
        # Add <loc> tags around location/spatial variables for CMA compatibility
        for var_name, value in substitutions.items():
            if any(spatial_keyword in var_name.lower() for spatial_keyword in 
                   ["location", "street", "section", "place"]):
                false_belief_scenario = false_belief_scenario.replace(value, f"<loc>{value}</loc>")
                true_belief_scenario = true_belief_scenario.replace(value, f"<loc>{value}</loc>")
        
        # Get answers
        if template.belief_answer_key in substitutions:
            belief_ans = substitutions[template.belief_answer_key]
        else:
            # Handle literal values like "long", "short"
            belief_ans = template.belief_answer_key
        
        if template.world_answer_key in substitutions:
            world_ans = substitutions[template.world_answer_key]
        else:
            world_ans = template.world_answer_key
        
        # Return: (false_belief_tuple, true_belief_tuple)
        return ((false_belief_scenario, belief_ans, world_ans), 
                (true_belief_scenario, world_ans, world_ans))


class UnifiedQuestionFormatter:
    """Unified question formatter for all template types."""
    
    @staticmethod
    def get_questions(template_name: str, variant: str = "standard") -> Dict[str, str]:
        """Get appropriate questions for any template."""
        
        # Template-specific questions
        specific_questions = {
            "basic_object_move": {
                "standard": {
                    "belief": "agent returns and looks where?",
                    "world": "where is object actually located?",
                    "dual": "where does agent look? and where is object actually?"
                },
                "detailed": {
                    "belief": "When Alice returns, where would she look for the object? Please be concise.",
                    "world": "Where is the object actually located? Please be concise.",
                    "dual": "When Alice returns, where would she look? And where is the object actually? Please be concise."
                }
            },
            
            "basic_object_move_detailed": {
                "standard": {
                    "belief": "When Alice returns, where would she look for the object?",
                    "world": "Where is the object actually located?", 
                    "dual": "When Alice returns, where would she look? And where is the object actually?"
                },
                "detailed": {
                    "belief": "When Alice returns, where would she look for the object? Please be concise.",
                    "world": "Where is the object actually located? Please be concise.",
                    "dual": "When Alice returns, where would she look? And where is the object actually? Please be concise."
                }
            },
            
            "food_truck": {
                "standard": {
                    "belief": "Alice looks for the truck where?",
                    "world": "truck is actually where?",
                    "dual": "Alice looks where? truck actually where?"
                },
                "detailed": {
                    "belief": "When Alice returns to buy food, where would she look for the food truck? Please be concise.",
                    "world": "Where is the food truck actually located? Please be concise.",
                    "dual": "When Alice returns to buy food, where would she look? And where is it actually? Please be concise."
                }
            },
            
            "hair_styling": {
                "standard": {
                    "belief": "Jennifer says Alice's hair is?",
                    "world": "Alice's hair is actually?",
                    "dual": "Jennifer says hair is? actually is?"
                },
                "detailed": {
                    "belief": "How would Jennifer respond about Alice's hair length?",
                    "world": "What is Alice's actual hair length?",
                    "dual": "How would Jennifer respond about Alice's hair length? And what is Alice's actual hair length?"
                }
            },
            
            "library_book": {
                "standard": {
                    "belief": "Sarah looks for book in which section?",
                    "world": "book actually in which section?",
                    "dual": "Sarah looks where? book actually in which section?"
                },
                "detailed": {
                    "belief": "When Sarah returns, in which section would she look for her book? Please be concise.",
                    "world": "Which section is the book actually located? Please be concise.",
                    "dual": "When Sarah returns, which section would she look to find the book? Where is the book actually? Please be concise."
                }
            },
            
            "restaurant_reservation": {
                "standard": {
                    "belief": "Tom expects reservation at?",
                    "world": "reservation actually at?",
                    "dual": "Tom expects? actually?"
                },
                "detailed": {
                    "belief": "When Tom arrives at the restaurant, what time does he expect the reservation? Please be concise.",
                    "world": "What time is the reservation actually scheduled for? Please be concise.",
                    "dual": "What time does Tom expect? What time is it actually scheduled? Please be concise."
                }
            }
        }
        
        # Return template-specific questions or generic fallback
        if template_name in specific_questions and variant in specific_questions[template_name]:
            return specific_questions[template_name][variant]
        
        # Generic fallback
        return {
            "belief": "what does the agent think?",
            "world": "what is reality?", 
            "dual": "what does agent think? what is reality?"
        }
    
    @staticmethod
    def get_cma_questions(template_name: str, question_style: str = "completion") -> str:
        """Get CMA-compatible questions that prompt tagged location answers.
        
        Args:
            template_name: Name of the scenario template
            question_style: "completion" or "instruction" prompting approach
        
        Used for causal mediation analysis where precise tokenization is needed.
        Forces model to output locations in <loc></loc> format for token extraction.
        """
        
        if question_style == "completion":
            return UnifiedQuestionFormatter._get_completion_questions(template_name)
        else:  # instruction (default)
            return UnifiedQuestionFormatter._get_instruction_questions(template_name)
    
    @staticmethod  # for the completion case, temporarily removing the location tags.
    def _get_completion_questions(template_name: str) -> str:
        """Completion-based questions that force model to output just the location."""
        completion_questions = {
            "basic_object_move": "agent returns and looks where? She would look",
            "basic_object_move_detailed": "When Alice returns, where would she look for the object? She would look",
            "food_truck": "When Alice wants to buy food, where would she go to find the truck? She would go to",
            "hair_styling": "How would Jennifer respond about Alice's hair length? She would say it is",
            "library_book": "When Sarah returns, which section would she look for her book? She would look in the",
            "restaurant_reservation": "When Tom arrives, what time does he expect the reservation? He expects it at"
        }
        
        return completion_questions.get(template_name, "Answer: <loc>")
        
    @staticmethod  
    def _get_instruction_questions(template_name: str) -> str:
        """Instruction-based CMA questions with explicit formatting requirements."""
        instruction_questions = {
            "basic_object_move": "agent returns and looks where? Put the location in <loc></loc> tags.",
            "basic_object_move_detailed": "When Alice returns, where would she look for the object? Put the location in <loc></loc> tags.",
            "food_truck": "Alice looks for the truck where? Put the location in <loc></loc> tags.", 
            "hair_styling": "Jennifer says Alice's hair is? Answer format: long or short",
            "library_book": "Sarah looks for book in which section? Put the location in <loc></loc> tags.",
            "restaurant_reservation": "Tom expects reservation at what time? Put the time in <loc></loc> tags."
        }
        
        return instruction_questions.get(template_name, "Answer with the location only: <loc></loc>")