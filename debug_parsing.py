import sys
sys.path.append('codebase/tasks/identity_rules')
from prompt_generators.base import wrap_dual_schema, extract_dual_answers, LOC_RE
import re

# Test the current parsing
test_output = '''what the agent thinks: <loc>suitcase</loc>
the real location of the object: <loc>carpet</loc>'''

print('Test parsing (good case):')
belief, world, malformed = extract_dual_answers(test_output)
print(f'belief: "{belief}"')
print(f'world: "{world}"')
print(f'malformed: {malformed}')
print()

# Test with problem case from your example  
problem_case = 'what the agent thinks:'
print('Problem case parsing (no loc tags):')
matches = list(LOC_RE.finditer(problem_case))
print(f'matches: {matches}')
belief2, world2, malformed2 = extract_dual_answers(problem_case)
print(f'belief: "{belief2}"')  
print(f'world: "{world2}"')
print(f'malformed: {malformed2}')
print()

# Test with incomplete response
incomplete_case = '''what the agent thinks: <loc>suitcase</loc>
the real location'''
print('Incomplete case:')
belief3, world3, malformed3 = extract_dual_answers(incomplete_case)
print(f'belief: "{belief3}"')
print(f'world: "{world3}"')
print(f'malformed: {malformed3}')