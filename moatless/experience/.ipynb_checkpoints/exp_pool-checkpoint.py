exp_pool_prompt_1 = '''
As a software engineering expert, next you will be given an issue and summarize experiences from the failed trajectory. 

Experience usually refers to the reasons for failure and success of tool calls in the repository and some reasons why this trajectory failed. When summarizing your experience, carefully compare the failed trajectory with the issue.

Your output should strictly follow JSON format with the following structure:

{
  "for_all_repositories": [
    "1. ...",
    ...
  ],
  "for_<repo>": [
    "1. ...",
    ...
  ]
}


If you believe this experience is relevant to all repositories and not just limited
to <repo>, please write them after ’For all repositories:’. If this experience is
only applicable to the <repo> repository, write it after ’For <repo>:’. Note that the
content in the two parts should NOT have any repetitions.
'''


exp_pool_prompt_2 = '''
As a software engineering expert, next you will be given an issue and summarize experiences from the resolved trajectory.

Experience usually refers to the reasons for failure and success of tool calls in the repository and some insights from the resolved trajectory. When summarizing your experience, carefully compare the resolved trajectory with the issue.

Your output should strictly follow JSON format with the following structure:

{
  "for_all_repositories": [
    "1. ...",
    ...
  ],
  "for_<repo>": [
    "1. ...",
    ...
  ]
}

If you believe this experience is relevant to all repositories and not just limited
to <repo>, please write them after ’For all repositories:’. If this experience is
only applicable to the <repo> repository, write it after ’For <repo>:’. Note that the
content in the two parts should NOT have any repetitions.
'''

user_prompt = '''
Repo:
{repo}\n
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
Your output should strictly follow JSON format with the following structure:
{
"Update":
    {
      "for_all_repositories": [
        "1. ADD or EDIT or REMOVE or APPROVE or MERGE ...",
        ...
      ],
      "for_<repo>": [
        "1. ADD or EDIT or REMOVE or APPROVE or MERGE ...",
        ...
      ]
    },
"Result":
    {
      "for_all_repositories": [
        "1. ...",
        ...
      ],
      "for_<repo>": [
        "1. ...",
        ...
      ]
    }
}

Please ensure:
1. There are no repetitions between the "for_all_repositories" and "for_<repo>" sections.
2. If the length of EXISTING RULES is greater than 20, you must use remove or merge at least once.
'''

user_merge_prompt = '''
New rules:
{new_rules}\n
Existing rules:
{existing_rules}\n
Please provide your determined operations and result below based on the above list of EXISTING RULES:
'''