"""Test the minimal string-prefix fix for parallel agent visibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'tests' / 'unittests'))

import testing_utils
from google.adk.agents.llm_agent import Agent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent


def test_string_prefix_fix():
  """Test: Sequential[Parallel[A, B, C], Reducer]"""
  
  # Group 1
  A = Agent(
      name='Alice',
      description='An obedient agent.',
      instruction='Say your name.',
      model=testing_utils.MockModel.create(responses=['I am Alice']),
  )
  B = Agent(
      name='Bob',
      description='An obedient agent.',
      instruction='Say your name.',
      model=testing_utils.MockModel.create(responses=['I am Bob']),
  )
  C = Agent(
      name='Charlie',
      description='An obedient agent.',
      instruction='Say your name.',
      model=testing_utils.MockModel.create(responses=['I am Charlie']),
  )
  
  # Parallel ABC
  P1 = ParallelAgent(
      name='ABC',
      description='Parallel group ABC',
      sub_agents=[A, B, C],
  )
  
  # Reducer
  R1 = Agent(
      name='reducer1',
      description='Reducer for ABC',
      instruction='Summarize.',
      model=testing_utils.MockModel.create(responses=['Summary of Alice, Bob, Charlie']),
  )
  
  S1 = SequentialAgent(
      name='Group1_Sequential',
      description='Sequential group for ABC',
      sub_agents=[P1, R1],
  )
  
  # Run
  runner = testing_utils.InMemoryRunner(S1)
  runner.run('Test')
  
  # Check LLM requests
  print('\n' + '*' * 80)
  print('STRING-PREFIX FIX TEST')
  print('*' * 80)
  
  # Check reducer's LLM request
  if R1.model and hasattr(R1.model, 'requests'):
    print('\nReducer1 LLM Request:')
    for i, req in enumerate(R1.model.requests):
      contents = testing_utils.simplify_contents(req.contents)
      for role, text in contents:
        print(f'  {role}: {text}')
  
  # Check branches
  print('\nBranch contexts:')
  for event in runner.session.events:
    if hasattr(event, 'author') and event.author:
      print(f'{event.author}: branch="{event.branch}"')
  
  # Verify reducer can see Alice, Bob, Charlie
  reducer_saw_alice = False
  reducer_saw_bob = False
  reducer_saw_charlie = False
  
  if R1.model and hasattr(R1.model, 'requests') and R1.model.requests:
    req_text = str(R1.model.requests[0].contents)
    reducer_saw_alice = 'Alice' in req_text
    reducer_saw_bob = 'Bob' in req_text
    reducer_saw_charlie = 'Charlie' in req_text
  
  print(f'\n✅ Reducer saw Alice: {reducer_saw_alice}')
  print(f'✅ Reducer saw Bob: {reducer_saw_bob}')
  print(f'✅ Reducer saw Charlie: {reducer_saw_charlie}')
  
  if reducer_saw_alice and reducer_saw_bob and reducer_saw_charlie:
    print('\n🎉 SUCCESS! String-prefix fix works!')
  else:
    print('\n❌ FAILED! Reducer cannot see all parallel agents.')
    sys.exit(1)
  
  print('*' * 80)


if __name__ == '__main__':
  test_string_prefix_fix()
