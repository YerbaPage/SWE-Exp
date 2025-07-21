import json

# for experience

success_example = '''

'''
failed_example = '''

'''



reflection_system_prompt_1 = f'''
As a software engineering expert, next you will be given an issue and summarize multiple experiences from the failed trajectory. 

Experience usually refers to the reasons for failure and success of tool calls when trying to resolve this issue and some reasons why this trajectory failed. When summarizing your experience, carefully compare the failed trajectory with the issue. Be sure to remember that the end result of this trajectory is failure! Don't think of the final wrong result as the right experience!

Your output should strictly follow JSON format with the following structure:

{{
  "for_all_instances": [
    "1. ...",
    ...
  ],
  "for_this_instance": [
    "1. ...",
    ...
  ]
}}

for_all_instances: These should be higher-level, thought-driven and more versatile insights which are relevant to all instances and not just limited to this instance. Experiences should not be specific to this instance's type of problem and must not mention the specific object name about this instance.
for_this_instance: These should be more fine-grained, capturing observations specific to this particular failed attempt. This may include notable discoveries of the tools calls and suggestions from the previous trial that directly relate to the resolution of this instance.
There should be no overlap between the two sections—generalizable insights belong in for_all_instances, while details unique to this case go in for_this_instance. You should choose the most important insights as experiences.
'''


reflection_system_prompt_2 = f'''
As a software engineering expert, next you will be given an issue and summarize multiple experiences from the resolved trajectory. 

Experience usually refers to the reasons for failure and success of tool calls in the repository and some factors that contributed to the successful trajectory. When summarizing your experience, carefully compare the successful trajectory with the issue. Be sure to focus on the positive outcomes of each action and the decisions that led to success!


Your output should strictly follow JSON format with the following structure:

{{
  "for_all_instances": [
    "1. ...",
    ...
  ],
  "for_this_instance": [
    "1. ...",
    ...
  ]
}}

for_all_instances: These should be higher-level, thought-driven and more versatile insights which are relevant to all instances and not just limited to this instance. Experiences should not be specific to this instance's type of problem and must not mention the specific object name about this instance.
for_this_instance: These should be more fine-grained, capturing observations specific to this particular failed attempt. This may include notable discoveries of the tools calls and suggestions from the previous trial that directly relate to the resolution of this instance.
There should be no overlap between the two sections—generalizable insights belong in for_all_instances, while details unique to this case go in for_this_instance. You should choose the most important insights as experiences.
'''


user_prompt = '''
Issue:
{issue}\n
Trajectory:
{trajectory}\n
Output:
'''

# for the use of experience
experience_prompt = '''
Here are some experiences summarized from the attempt to solve this instance last time:
{}
Please refer to these experiences during fixing the issue.
'''



# for the experience manager
manager_example_1 = '''
'''

manager_example_2 = '''
'''

manager_system_prompt = '''
You are an advanced reasoning agent capable of modifying your existing experiences (represented as rules) by adding, editing, removing, or merging rules based on new rules provided. Your task is to update the ’Existing rules’ according to the ’New rules’.

You may perform the following operations:
APPROVE: if the new rules are present in the existing rules and you think the existing rule should be remained.
REMOVE: if the new rule contradicts existing rules or if there’s redundancy among the existing rules.
ADD: to introduce new rules that substantially differ from existing ones and are applicable to relevant tasks.
EDIT: if an existing rule lacks clarity or generality. Revise it for improvement or to address past issues.
MERGE: to consolidate two similar existing rules into a single, cohesive rule. Each operation must closely follow the specified format:
<OPERATION> <RULE NUMBER>: <RULE>
The format for each operation is as follows:
APPROVE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>
MERGE <EXISTING RULE NUMBER1> <EXISTING RULE NUMBER2>: <NEW RULE>
Please follow the output format:
{{
"Update":
    {{
      "for_all_instances": [
        "1. ADD or EDIT or REMOVE or APPROVE or MERGE ...",
        ...
      ],
      "for_this_instance": [
        "1. ADD or EDIT or REMOVE or APPROVE or MERGE ...",
        ...
      ]
    }},
"Result":
    {{
      "for_all_instances": [
        "1. ...",
        ...
      ],
      "for_this_instance": [
        "1. ...",
        ...
      ]
    }}
}}

Please ensure:
1. There are no repetitions between the "for_all_instances" and "for_this_instance" sections.
2. If the length of EXISTING RULES is greater than 20, you must use remove or merge at least once.
'''

# manager_examples = '''
# Here is an example given below:
# ** Example 1 **
# New rules:
# For all repositories:
# 1. A systematic approach to debugging involves examining the specific functions and classes that are central to the issue, as was correctly done by inspecting the separability_matrix and _separable functions.
# 2. Thoroughly exploring the codebase to understand how nested Models are handled within the relevant functions is crucial for diagnosing and resolving complex bugs.

# For astropy:
# 1. The identification of the problem within the separability_matrix function as a bug in handling nested CompoundModels is a correct interpretation of the observed behavior, demonstrating the importance of code inspection in debugging.
# 2. The approach to analyze the _separable function further after confirming the issue with separability_matrix illustrates the value of a methodical debugging process, particularly when dealing with nested structures that may have unique handling requirements.


# Existing rules:
# For all repositories:
# 1. Precise matching of code snippets is crucial for actions like StringReplace; any discrepancies such as extra whitespace or incorrect line breaks can lead to failure in modifying the code.
# 2. When addressing complex issues like nested Models, it's important to fully understand the behavior of related functions before attempting to modify the code; without this understanding, changes may not resolve the problem or could introduce new issues.

# For astropy:
# 1. The failed attempt to modify the separability_matrix function highlights the importance of accurately targeting the specific code segments that need changes. Misidentification can lead to ineffective patches.
# 2. The need to address nested CompoundModels indicates potential gaps in the handling of complex model structures in astropy's modeling functions, suggesting that a comprehensive review of the _separable function and its logic is necessary.

# Please provide your determined operations and result below based on the above list of EXISTING RULES: 

# {
# "Update":
#     {"for_all_repositories": [
#         "ADD 3: A systematic approach to debugging involves examining the specific functions and classes that are central to the issue. ",
#         "ADD 4: Thoroughly exploring the codebase to understand how nested Models are handled within the relevant functions is crucial for diagnosing and resolving complex bugs.",
#         "MERGE 2 4: When addressing complex issues like nested Models, it's important to fully understand the behavior of related functions and thoroughly explore the codebase before attempting to modify the code; without this understanding, changes may not resolve the problem or could introduce new issues."
#       ],
#       "for_astropy": [
#         "REMOVE 2: The need to address nested Models indicates potential gaps in the handling of complex model structures in astropy's modeling functions, suggesting that a comprehensive review of the _separable function and its logic is necessary.",
#         "ADD 3: The identification of the problem within the separability_matrix function as a bug in handling nested CompoundModels is a correct interpretation of the observed behavior, demonstrating the importance of code inspection in debugging.",
#         "ADD 4: The approach to analyze the _separable function further after confirming the issue with separability_matrix illustrates the value of a methodical debugging process, particularly when dealing with nested structures that may have unique handling requirements."
#       ]
#     },
# "Result":
#     {"for_all_repositories": [
#         "1. Precise matching of code snippets is crucial for actions like StringReplace; any discrepancies such as extra whitespace or incorrect line breaks can lead to failure in modifying the code.",
#         "2. When addressing complex issues like nested Models, it's important to fully understand the behavior of related functions and thoroughly explore the codebase before attempting to modify the code; without this understanding, changes may not resolve the problem or could introduce new issues.",
#         "3. A systematic approach to debugging involves examining the specific functions and classes that are central to the issue.",
#         "4. Thoroughly exploring the codebase to understand how nested Models are handled within the relevant functions is crucial for diagnosing and resolving complex bugs."
#       ],
#       "for_astropy": [
#         "1. The failed attempt to modify the separability_matrix function highlights the importance of accurately targeting the specific code segments that need changes. Misidentification can lead to ineffective patches.",
#         "2. The identification of the problem within the separability_matrix function as a bug in handling nested CompoundModels is a correct interpretation of the observed behavior, demonstrating the importance of code inspection in debugging.",
#         "3. The approach to analyze the _separable function further after confirming the issue with separability_matrix illustrates the value of a methodical debugging process, particularly when dealing with nested structures that may have unique handling requirements."
#       ]
#     }
# }

# '''

user_merge_prompt = '''
New rules:
{new_rules}\n
Existing rules:
{existing_rules}\n
Please provide your determined operations and result below based on the above list of EXISTING RULES:
'''