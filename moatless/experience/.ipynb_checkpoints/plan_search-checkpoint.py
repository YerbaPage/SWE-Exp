def chat(model, user_prompt):
    messages = []
    messages.append({"role": "user", "content": user_prompt})
    output = model._litellm_base_completion(
                    messages=messages
                )
    return output.choices[0].message.content


plan_search_prompt_1 = '''You are a software engineering expert. You will be given an issue (problem statement). You will return several useful, non-obvious, and correct observations
about the problem, like hints to solve the issue. You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct. 
The observations should have the following format:
Observations: [
1. ...
2. ...
...
]

Issue: {}
Observations: 
'''

plan_search_prompt_2 = '''You are a software engineering expert. You will be given an issue (problem statement) and several observations about the problem.\n
You will brainstorm several new, useful, and correct observations about the problem, derived from the given observations. You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.
The new observations should have the following format:
New Observations: [
1. ...
2. ...
...
]

Issue: {}
Observations: {}
New Observations:
'''

combine_obs_prompt = '''Here is the software engineering issue:
{}

Here are the intelligent observations to help analyze the issue:
Observations: {}

Use these observations above to brainstorm a step-by-step plan to solve this issue.
Note that your plan may lead you astray, so come up with simple, creative ideas that
go beyond what you would usually come up with and exceed your narrow intuition.
Quote relevant parts of the observations EXACTLY before each step of the plan. QUOTING
IS CRUCIAL.
'''