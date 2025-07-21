import os
import json


awm_exp = '''Problem Statement: The `__deepcopy__` method for Django's form fields does not perform a deep copy of the `error_messages` dictionary. As a result, all copies of the same field share the same dictionary, causing modifications to the error messages in one form field instance to be reflected in all other instances. This issue affects Forms and ModelForms that dynamically modify error messages, leading to incorrectly raised errors. The problem has been confirmed in Django versions dating back to 1.11.

Reason: 1. **Identify the Issue**: To address the issue with the `__deepcopy__` method in Django form fields not copying error messages deeply, the first step is to locate the `__deepcopy__` method in the Django forms fields module. This involves searching for the method in the `fields.py` file using the `FindCodeSnippet` action.
2. **Analyze the Code**: After locating the `__deepcopy__` method in the `Field` class, it becomes clear that the method does not perform a deep copy of the `error_messages` dictionary. This omission results in shared state between form field instances, which is the root cause of the issue.
3. **Modify the Code**: To resolve the issue, the `__deepcopy__` method needs to be modified to include a deep copy of the `error_messages` dictionary. This ensures that each form field instance has its own independent copy of the error messages, preventing shared state.
4. **Confirm the Fix**: The issue has been successfully resolved by updating the `__deepcopy__` method. The modification ensures that the `error_messages` dictionary is deep copied, maintaining independent error message state for each form field instance. This concludes the task.

Actions: 1. FindCodeSnippet(file_pattern="django/forms/fields.py", code_snippet="def __deepcopy__")
2. StringReplace(path="django/forms/fields.py", old_str="    def __deepcopy__(self, memo):\n        result = copy.copy(self)\n        memo[id(self)] = result\n        result.widget = copy.deepcopy(self.widget, memo)\n        result.validators = self.validators[:]\n        return result", new_str="    def __deepcopy__(self, memo):\n        result = copy.copy(self)\n        memo[id(self)] = result\n        result.widget = copy.deepcopy(self.widget, memo)\n        result.validators = self.validators[:]\n        result.error_messages = copy.deepcopy(self.error_messages, memo)\n        return result")
'''


issue_exp = '''
Form Field’s __deepcopy__ does not (deep)copy the error messages.\nDescription\n\nThe __deepcopy__ method defined for the formfields (​https://github.com/django/django/blob/146086f219d01dbb1cd8c089b5a5667e396e1cc4/django/forms/fields.py#L200) performs a shallow copy of self and does not include additional treatment for the error_messages dictionary. As a result, all copies of the same field share the same dictionary and any modification of either the dictionary or the error message itself for one formfield is immediately reflected on all other formfiels.
This is relevant for Forms and ModelForms that modify the error messages of their fields dynamically: while each instance of the specific form (e.g., ProfileForm) is expected to have a set of fields “sealed” away from other instances of the same ProfileForm (​https://github.com/django/django/blob/146086f219d01dbb1cd8c089b5a5667e396e1cc4/django/forms/forms.py#L95), in fact all these instances share the same error messages, resulting in incorrectly raised errors.
Confirmed for versions of Django going back to 1.11.
'''

awm_prompt = '''
As a software engineering expert, next you will be given an issue and summarize a workflow. When you organized the actions of the workflow, you could only use the actions below:
{}

Ensure that the generated workflow could summarize the issue's context and follows the structure and style of the given examples:

Issue:\n {}\n
Workflow:\n{}\n

Issue:\n {}\n
Workflow:\n
'''

def get_thoughts(search_tree):
    # 总结一棵树所有的thoughts到一个list
    traj = []
    i = 0
    def dfs(node, traj):
        if not node:
            return
        if node.completions:
            nonlocal i
            i += 1
            thought = node.completions['build_action'].response['choices'][0]['message']['content']
            # feedback = node.completions['value_function'].response['choices'][0]['message']['content']
            # print(f'{node.completions['build_action'].response['choices'][0]['message']['content']}')
            # print(f'{node.completions['value_function'].response['choices'][0]['message']['content']}')
            traj.append(f'In the Step {i},\n' + 'reasoning:' + thought + '\n\n')
        
        for c in node.children:
            dfs(c, traj)
    dfs(search_tree.root, traj)
    traj = ''.join(traj)
    return traj


def get_summary(statement, patch, model):
    # 根据生成patch和problem_statement进行对比生成一个summary，目的是减少problem statement中的噪声
    messages = []
    user_prompt = f'''
            Given an issue description and a corresponding patch, generate a concise summary that describes the issue itself without referencing or mentioning the patch in any way. 
            Please refine the issue description by removing unnecessary statements that are no longer relevant based on the applied patch. Additionally, the summary should focus solely on the problem stated in the issue, ensuring no details about the solution or modifications are included.
            Issue: {statement}\n
            Patch: {patch}\n
            Summary:
            '''
    # user_prompt = f'''
    #     Given an issue description, generate a concise summary that describes the issue itself. 
    #     The summary should focus solely on the problem stated in the issue.
    #     Issue: {statement}\n
    #     Summary:
    #     '''
    messages.append({"role": "user", "content": user_prompt})
    output = model._litellm_base_completion(
                    messages=messages
                )
    return output
    

def get_reasoning(search_tree, model):
    # 根据模型的thoughts生成推理链
    traj = get_thoughts(search_tree)
    user_prompt = f'''
        Given a trajectory, extract and organize the reasoning chain from the thought of each step, following the format below:\n
        1. ...
        2. ...
        3. ...
        ...\n
        Trajectory: {traj}
        Reasoning chain:
    '''
    messages = []
    messages.append({"role": "user", "content": user_prompt})
    output = model._litellm_base_completion(
                    messages=messages
                )
    return output

def get_action(search_tree, model):
    # 根据模型的对话历史生成action链
    traj = get_thoughts(search_tree)
    user_prompt = f'''
        Given a trajectory, extract and organize the chain of called actions from the thoughts of each step. List only the action names and their arguments in the following format:
        1. ActionName(argument1=..., argument2=..., ...)
        2. ActionName(argument1=..., argument2=..., ...)
        3. ActionName(argument1=..., argument2=..., ...)
        ...\n
        Trajectory: {traj}
        Action chain:
    '''
    messages = []
    messages.append({"role": "user", "content": user_prompt})
    output = model._litellm_base_completion(
                    messages=messages
                )
    return output

def get_awm_exp(search_tree, model, patch):
    # 生成AWM形式的experience
    statement = search_tree.root.user_message
    summary = get_summary(statement, patch, reflection_model).choices[0].message.content
    reasoning = get_reasoning(search_tree, model).choices[0].message.content
    actions = get_action(search_tree, model).choices[0].message.content
    experience = f'''
Problem Statement: {summary}\n
Reason: {reasoning}\n
Actions: {actions}
    '''
    return experience

def awm_learning(actions_prompt, issue_exp, awm_exp, statement, model):
    # 根据模型的对话历史生成action链
    messages = []
    messages.append({"role": "user", "content": awm_prompt.format(actions_prompt, issue_exp, awm_exp, statement)})
    output = model._litellm_base_completion(
                    messages=messages
                )
    return output.choices[0].message.content

