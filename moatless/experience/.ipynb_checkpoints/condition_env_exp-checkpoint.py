env_exp_prompt = '''
You are a software engineering expert. Given an exploration trajectory, you need to extract insights about the repository’s structure and the purpose of its folders and files.

Your output must strictly follow this JSON format:  

{{ 
  "for <repo>": [
      "for <folder>": [
        "1. <file> ...",
        ...  
      ],
      ...
  ]
}}  

Ensure that your summary accurately reflects the repository’s organization, the role of different files and directories, and any key observations from your exploration. 
- **Your observations must be strictly based on the code found in the trajectory's files and not on assumptions or external knowledge.**  
- **When summarizing a file, explicitly mention the classes and methods it contains.**
'''


cond_exp_prompt_1 = '''
As a software engineering expert, next you will be given an issue and summarize multiple experiences from the failed trajectory. 

Experience usually refers to the reasons for failure and success of tool calls when trying to resolve this issue and some reasons why this trajectory failed. When summarizing your experience, carefully compare the failed trajectory with the issue. Be sure to remember that the end result of this trajectory is failure! Don't think of the final wrong result as the right experience!

Your output should strictly follow JSON format with the following structure:

{{
  "for_all_instances": [
    "1. If <condition>, then <insight>.",
    ...
  ],
  "for_this_instance": [
    "1. If <condition>, then <insight>.",
    ...
  ]
}}

for_all_instances: These should be higher-level, thought-driven and more versatile insights which are relevant to all instances and not just limited to this instance. Experiences should not be specific to this instance's type of problem and must not mention the specific object name about this instance.
for_this_instance: These should be more fine-grained, capturing observations specific to this particular failed attempt. This may include notable discoveries of the tools calls and suggestions from the previous trial that directly relate to the resolution of this instance.
There should be no overlap between the two sections—generalizable insights belong in for_all_instances, while details unique to this case go in for_this_instance. You should choose the most important insights as experiences.

**Important Rule:**  
Each experience should start with a condition clause specifying when the agent should apply this experience. The format should be:  
"If <condition>, then <insight>."
'''


cond_exp_prompt_2 = '''
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

**Important Rule:**  
Each experience should start with a condition clause specifying when the agent should apply this experience. The format should be:  
"If <condition>, then <insight>."
'''