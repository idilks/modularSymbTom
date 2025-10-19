"""
Unified prompt builder using the new template system.
Replaces prompt_builder.py with a cleaner, single-system approach.
"""

import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from unified_templates import UnifiedTemplateSystem, UnifiedQuestionFormatter


@dataclass 
class Prompt:
    """Single prompt for evaluation."""
    text: str
    expected_belief: str
    expected_world: str
    metadata: Dict[str, str]


class UnifiedPromptBuilder:
    """Unified prompt builder using template system for all scenarios."""
    
    def __init__(self, vocab_dir: str = "tom_datasets"):
        self.template_system = UnifiedTemplateSystem(vocab_dir)
        self.question_formatter = UnifiedQuestionFormatter()
    
    def generate_prompt(
        self,
        template_name: str,
        vignette_type: str,
        question_type: str = "dual",
        format_type: str = "direct",
        variant: str = "standard"
    ) -> Prompt:
        """Generate single prompt using template system."""
        
        # Generate scenario with random variables
        scenario_text, belief_ans, world_ans = self.template_system.generate_scenario(
            template_name, vignette_type
        )
        
        # Get appropriate questions for this template
        questions = self.question_formatter.get_questions(template_name, variant)
        
        # Format question based on type
        print(f"GENERATE DEBUG: question_type = '{question_type}' for template '{template_name}'")
        
        if question_type == "belief":
            # For single questions, use CMA-style structured prompts when available
            try:
                from unified_templates import UnifiedQuestionFormatter
                cma_question = UnifiedQuestionFormatter.get_cma_questions(template_name, "completion")  # force completion
                print(f"DEBUG: cma_question = {cma_question}")
                if cma_question and cma_question != "Answer with the location only: <loc></loc>":
                    prompt_text = f"{scenario_text}\n{cma_question}"  # NO extra </loc> at end
                else:
                    question = questions["belief"]
                    prompt_text = f"{scenario_text}\n{question}\n<loc></loc>"
            except Exception as e:
                print(f"DEBUG: exception = {e}")
                question = questions["belief"]
                prompt_text = f"{scenario_text}\n{question}\n<loc></loc>"
            print(f"GENERATE DEBUG: Using belief question: {repr(prompt_text.split(scenario_text)[-1])}")
        elif question_type == "world":
            question = questions["world"]
            # For world questions, keep simple format since CMA questions are belief-focused
            prompt_text = f"{scenario_text}\n{question}\nAnswer with using <loc></loc>."
            print(f"GENERATE DEBUG: Using world question: {repr(question)}")
        else:  # dual
            question = questions["dual"]
            print(f"GENERATE DEBUG: Using dual question: {repr(question)}")
            if format_type == "direct":
                prompt_text = (
                    f"{scenario_text}\n{question}\n"
                    f"Answer in this exact schema:\n"
                    f"where the agent thinks: <loc></loc>\n"
                    f"actual location: <loc></loc>\n"
                )
            else:  # multiple_choice - simplified for now
                prompt_text = f"{scenario_text}\n{question}\nA) {belief_ans}\nB) {world_ans}\nAnswer: "
        
        return Prompt(
            text=prompt_text,
            expected_belief=belief_ans,
            expected_world=world_ans,
            metadata={
                "vignette_type": vignette_type,
                "question_type": question_type,
                "format_type": format_type,
                "variant": variant,
                "template_name": template_name
            }
        )
    
    def batch_generate(
        self,
        num_prompts: int,
        template_names: List[str] = None,
        vignette_types: List[str] = None,
        question_types: List[str] = None,
        format_types: List[str] = None,
        variants: List[str] = None
    ) -> List[Prompt]:
        """Generate batch of prompts with specified distributions."""
        
        # Defaults
        template_names = template_names or ["basic_object_move"]
        vignette_types = vignette_types or ["false_belief", "true_belief"]
        question_types = question_types or ["dual"]
        format_types = format_types or ["direct"]
        variants = variants or ["standard"]
        
        prompts = []
        
        # Special handling for paired questions (belief/world with same scenario)
        print(f"UNIFIED DEBUG: question_types = {question_types}")
        print(f"UNIFIED DEBUG: 'belief' in question_types = {'belief' in question_types}")
        print(f"UNIFIED DEBUG: 'world' in question_types = {'world' in question_types}")
        
        if "belief" in question_types and "world" in question_types:
            print(f"UNIFIED DEBUG: Taking paired questions path")
            return self._generate_paired_questions(
                num_prompts, template_names, vignette_types, format_types, variants
            )
        
        # Regular generation
        print(f"UNIFIED DEBUG: Taking regular generation path")
        combinations = [
            (template, vignette, question, format_type, variant)
            for template in template_names
            for vignette in vignette_types
            for question in question_types
            for format_type in format_types
            for variant in variants
        ]
        print(f"UNIFIED DEBUG: Created {len(combinations)} combinations")
        
        per_combo = max(1, num_prompts // len(combinations)) if combinations else 1
        
        for template, vignette, question, format_type, variant in combinations:
            for _ in range(per_combo):
                prompt = self.generate_prompt(template, vignette, question, format_type, variant)
                prompts.append(prompt)
        
        # Fill remainder if needed
        while len(prompts) < num_prompts:
            combo = random.choice(combinations)
            prompt = self.generate_prompt(*combo)
            prompts.append(prompt)
        
        random.shuffle(prompts)
        return prompts[:num_prompts]
    
    def _generate_paired_questions(
        self,
        num_prompts: int,
        template_names: List[str],
        vignette_types: List[str],
        format_types: List[str],
        variants: List[str]
    ) -> List[Prompt]:
        """Generate pairs of belief/world questions with identical scenarios."""
        
        prompts = []
        pairs_needed = num_prompts // 2  # Each pair generates 2 prompts
        
        combinations = [
            (template, vignette, format_type, variant)
            for template in template_names
            for vignette in vignette_types
            for format_type in format_types
            for variant in variants
        ]
        
        pairs_per_combo = max(1, pairs_needed // len(combinations)) if combinations else 1
        pair_counter = 0
        
        for template, vignette, format_type, variant in combinations:
            for _ in range(pairs_per_combo):
                if len(prompts) >= num_prompts:
                    break
                
                # Generate base scenario once per pair
                scenario_text, belief_ans, world_ans = self.template_system.generate_scenario(
                    template, vignette
                )
                
                # Get questions for this template
                questions = self.question_formatter.get_questions(template, variant)
                
                # Create belief question using CMA completion format
                try:
                    cma_question = self.question_formatter.get_cma_questions(template, "completion")
                    if cma_question and cma_question != "Answer with the location only: <loc></loc>":
                        belief_text = f"{scenario_text}\n{cma_question}"
                    else:
                        belief_text = f"{scenario_text}\n{questions['belief']}\n<loc></loc>"
                except:
                    belief_text = f"{scenario_text}\n{questions['belief']}\n<loc></loc>"
                
                belief_prompt = Prompt(
                    text=belief_text,
                    expected_belief=belief_ans,
                    expected_world=world_ans,
                    metadata={
                        "vignette_type": vignette,
                        "question_type": "belief",
                        "format_type": format_type,
                        "variant": variant,
                        "template_name": template,
                        "pair_id": pair_counter
                    }
                )
                prompts.append(belief_prompt)
                
                # Create world question with same scenario
                world_prompt = Prompt(
                    text=f"{scenario_text}\n{questions['world']}\n<loc></loc>",
                    expected_belief=belief_ans,
                    expected_world=world_ans,
                    metadata={
                        "vignette_type": vignette,
                        "question_type": "world",
                        "format_type": format_type,
                        "variant": variant,
                        "template_name": template,
                        "pair_id": pair_counter
                    }
                )
                prompts.append(world_prompt)
                
                pair_counter += 1
        
        # Fill to exact number if needed
        while len(prompts) < num_prompts:
            if prompts:
                prompts.append(random.choice(prompts))
        
        return prompts[:num_prompts]
    
    def list_available_templates(self) -> List[str]:
        """List all available template names."""
        return self.template_system.list_templates()


def create_unified_prompt_builder(config=None) -> UnifiedPromptBuilder:
    """Factory function for backward compatibility."""
    vocab_dir = "tom_datasets"
    if config and hasattr(config, 'tom_locations_file'):
        # Extract directory from locations file path
        import os
        vocab_dir = os.path.dirname(config.tom_locations_file) or "tom_datasets"
    
    return UnifiedPromptBuilder(vocab_dir)