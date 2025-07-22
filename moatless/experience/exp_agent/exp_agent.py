import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from dotenv import load_dotenv
load_dotenv(".env")

import re
import logging
import json
import json_repair
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.model import Completion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
)
from moatless.benchmark.utils import get_moatless_instance
from moatless.experience.get_save_json import get_json, save2json
from moatless.experience.prompts.exp_prompts import (
    encode_success_perspective_system_prompt,
    encode_failed_perspective_system_prompt,
    encode_success_modify_system_prompt,
)
from moatless.search_tree import SearchTree
from moatless.node import Node, generate_ascii_tree
from moatless.benchmark.swebench import create_repository

logger = logging.getLogger(__name__)


import os
import glob
from datetime import datetime

def find_latest_trajectory(base_dir, instance_id):
    try:
        instance_dir = os.path.join(base_dir, instance_id)
        
        if not os.path.exists(instance_dir):
            print(f"Wrong: instance directory not found: {instance_dir}")
            return None
            
        trajectory_files = glob.glob(os.path.join(instance_dir, "*_trajectory.json"))
        
        if not trajectory_files:
            print(f"Error: No trajectory file found in directory {instance_dir}")
            return None
            
        latest_file = max(trajectory_files, key=os.path.getctime)
        
        print(f"Found the latest trajectory file: {latest_file}")
        print(f"File creation time: {datetime.fromtimestamp(os.path.getctime(latest_file))}")
        
        return latest_file
        
    except Exception as e:
        print(f"Error happened: {str(e)}")
        return None

def find_trajectory(instance_id):
    base_dir = "SWE-Exp/tmp/trajectory"
    return find_latest_trajectory(base_dir, instance_id)



def find_instance_by_id(file_path, target_instance_id):
    try:
        l = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if target_instance_id.strip() == data.get('instance_id').strip():
                        l.append(data)
        return l
    except FileNotFoundError:
        print(f"Error {file_path}")
        return None
    except json.JSONDecodeError:
        print("Error: JSON parsing failed")
        return None
    except Exception as e:
        print(f"Error happened: {str(e)}")
        return None

def search_instance(file_path, instance_id):
    result = find_instance_by_id(file_path, instance_id)
    if result:
        print(f"Find Matched Result: ")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    else:
        print(f"No matched result for instance_id: {instance_id}")
        trajectory = find_trajectory(instance_id)
        print(f'Found trajectory path: {trajectory}')
        return [{'trajectory_path': trajectory}]


class ExpAgent(BaseModel):
    success_per_system_prompt: str = Field(
        ..., description="System prompt to be used to extract perspective exp for successful trajectories"
    )
    failed_per_system_prompt: str = Field(
        ..., description="System prompt to be used to extract perspective exp for failed trajectories"
    )

    success_mod_system_prompt: str = Field(
        ..., description="System prompt to be used to extract modification exp for successful trajectories"
    )

    issue_type_path: str = Field(
        ..., description="The path of Issue Type"
    )
    _completion: CompletionModel = PrivateAttr()
    def __init__(
            self,
            completion: CompletionModel,
            success_per_system_prompt: str,
            failed_per_system_prompt: str,
            success_mod_system_prompt: str,
            issue_type_path: str,
            **data,
    ):
        super().__init__(
            success_per_system_prompt=success_per_system_prompt,
            failed_per_system_prompt=failed_per_system_prompt,
            success_mod_system_prompt=success_mod_system_prompt,
            issue_type_path=issue_type_path,
            **data,
        )
        self._completion = completion

    def simplify(self, content):
        match = re.search(r"```(?:\w+)?\n?(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return content

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type((KeyError, json.decoder.JSONDecodeError, json.JSONDecodeError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _chat_with_retry(self, messages: List[dict]):
        try:
            response = self._completion._litellm_base_completion(
                messages=messages
            )
            response = response.choices[0].message.content
            response = self.simplify(response)

            response = json_repair.loads(response)

        except Exception as e:
            logger.error(f"Error in _chat_with_retry: {str(e)}")
            logger.error(f"Response: {response}")
            raise

        return response


    def encode_perspective(self, instance_id, rollout, patch, flag):
        issue_type = get_json(self.issue_type_path)
        issue = issue_type[instance_id]['issue']
        issue_type, description = issue_type[instance_id]['issue_type'], issue_type[instance_id]['description']
        user_prompt = f'''
<issue>
{issue}
</issue>

<golden_patch>
{patch}
</golden_patch>

<trajectory>
{rollout}
</trajectory>
'''
        if flag == 'success':
            system_prompt = self.success_per_system_prompt
        else:
            system_prompt = self.failed_per_system_prompt
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        return self._chat_with_retry(messages)


    def encode_modify(self, instance_id, rollout, patch):
            issue_type = get_json(self.issue_type_path)
            issue = issue_type[instance_id]['issue']
            issue_type, description = issue_type[instance_id]['issue_type'], issue_type[instance_id]['description']
            user_prompt = f'''
<issue>
{issue}
</issue>

<trajectory>
{rollout}
</trajectory>

<generated_patch>
{patch}
</generated_patch>
'''
            messages = [{"role": "system", "content": self.success_mod_system_prompt},
                        {"role": "user", "content": user_prompt}]
            return self._chat_with_retry(messages)

    @property
    def completion(self) -> CompletionModel:
        return self._completion

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        return dump


def simplify(content):
    match = re.search(r"```(?:\w+)?\n?(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return content



def get_success_rollout_with_patch(tree, evaluation, with_code=False) -> Tuple[List[List[dict]], str]:
    def dfs(node, traj):
        if not node:
            return
        if node.completions:
            try:
                thought = node.completions['build_action'].response['choices'][0]['message']['content']
                thought = simplify(thought)
                feedback = node.completions['value_function'].response['choices'][0]['message']['content']
                feedback = simplify(feedback)
                feedback = json.loads(feedback)
                if node.observation:
                    observation = node.observation.message
                    if observation == 'No search results found':
                        pass
                    else:
                        node_data = {"node_id": node.node_id, "action": thought, "result": feedback['explanation']}
                        if with_code:
                            node_data["observation"] = observation
                        if hasattr(node, 'instruct_message') and node.instruct_message:
                            if 'reasoning' in node.instruct_message:
                                node_data["reasoning"] = node.instruct_message['reasoning']
                            if 'instruction' in node.instruct_message:
                                node_data["instruction"] = node.instruct_message['instruction']
                        traj.append(node_data)

            except Exception as e:
                print(e)
            dfs(node.parent, traj)

    rollouts = []
    leaves = []
    for leaf in evaluation:
        if isinstance(leaf, dict):
            if leaf['resolved']:
                node = tree.get_node_by_id(leaf['leaf'])
                print(leaf['leaf'])
                print(node.file_context.generate_git_patch())
                print([i.node_id for i in tree.get_finished_nodes()])
                if leaf['leaf'] not in [i.node_id for i in tree.get_finished_nodes()]:
                    raise ValueError(f'leaf {leaf["leaf"]} is not in the finished nodes')
                if node and node.is_terminal():
                    rollout = []
                    dfs(node, rollout)
                    rollouts.append(rollout[::-1])
                    leaves.append(leaf['leaf'])
    shortest = [min(rollouts, key=len)]
    index = rollouts.index(shortest[0])
    node = tree.get_node_by_id(leaves[index])
    return shortest, node.file_context.generate_git_patch()  # the shortest trajectory is the most dense and the best


def get_trajectory(rollouts) -> List[str]:
    trajectories = []
    for rollout in rollouts:
        trajectory = ''
        for i in range(len(rollout)):
            node = rollout[i]
            trajectory += f'In the Stage {i+1},\n'
            
            if node.get('reasoning'):
                trajectory += 'Reasoning: ' + node['reasoning'] + '\n'
            
            if node.get('instruction'):
                trajectory += 'Instruction: ' + node['instruction'] + '\n'
            
            trajectory += 'Action: ' + node['action'] + '\n'
            
            if node.get('observation'):
                trajectory += 'Observation: ' + node['observation'] + '\n'
            
            trajectory += 'Result: ' + node['result'] + '\n\n'
        
        trajectories.append(trajectory)

    return trajectories


def get_failed_rollout(tree, with_code=False):
    def dfs(node, traj):
        if not node:
            return
        if node.completions:
            try:
                thought = node.completions['build_action'].response['choices'][0]['message']['content']
                thought = simplify(thought)
                feedback = node.completions['value_function'].response['choices'][0]['message']['content']
                feedback = simplify(feedback)
                feedback = json.loads(feedback)
                if node.observation:
                    observation = node.observation.message
                    if observation == 'No search results found':
                        pass
                    else:
                        stage_data = {"stage": node.node_id, "action": thought, "result": feedback['explanation']}
                        if with_code:
                            stage_data["observation"] = observation
                        if hasattr(node, 'instruct_message') and node.instruct_message:
                            if 'reasoning' in node.instruct_message:
                                stage_data["reasoning"] = node.instruct_message['reasoning']
                            if 'instruction' in node.instruct_message:
                                stage_data["instruction"] = node.instruct_message['instruction']
                        traj.append(stage_data)

            except Exception as e:
                print(e)
            dfs(node.parent, traj)


    if tree.get_finished_nodes():   # find the longest failed trajectory that generated patch
        ns = tree.get_finished_nodes()
        trajs = []
        for n in ns:
            traj = []
            dfs(n, traj)
            traj = traj[::-1]
            trajs.append(traj)
        return [max(trajs, key=len)]
    else:  # otherwise the longest trajectory (contains the most information)
        ns = tree.get_leaf_nodes()
        trajs = []
        for n in ns:
            traj = []
            dfs(n, traj)
            traj = traj[::-1]
            trajs.append(traj)
        return [max(trajs, key=len)]


if __name__ == '__main__':
    completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
    perspective_agent = ExpAgent(success_per_system_prompt=encode_success_perspective_system_prompt,
                                 failed_per_system_prompt=encode_failed_perspective_system_prompt,
                                 success_mod_system_prompt=encode_success_modify_system_prompt,
                                 issue_type_path='SWE-Exp/tmp/verified_issue_types_final.json',
                                 completion=completion_model)

    with open("SWE-Exp/verified_dataset_ids.txt", "r", encoding="utf-8") as f:
        ids = f.readlines()
    
    eval_path = 'SWE-Exp/tmp/merged_leaf_analysis_with_trajectories.jsonl'

    exp_tree = {}
    for instance_id in tqdm(ids):
        instance_id = instance_id.strip()
        eval = search_instance(eval_path, instance_id)
        tree_path = eval[0]['trajectory_path']
        tree = SearchTree.from_file(tree_path)
        flag = False
        exp_tree[instance_id] = []
        instance = get_moatless_instance(split='verified', instance_id=instance_id)
        golden_patch = instance['golden_patch']
        issue = instance['problem_statement']
        print(generate_ascii_tree(tree.root, tree.get_node_by_id(19)))
        for i in eval:
            if i.get('resolved') == True:
                rollout, patch = get_success_rollout_with_patch(tree, eval, False)
                trajectory = get_trajectory(rollout)
                if not trajectory:
                    raise ValueError(f'trajectory is empty for instance_id: {instance_id}')
                answer = perspective_agent.encode_perspective(instance_id, rollout=trajectory[0], patch=patch, flag='success')
                
                rollout, patch = get_success_rollout_with_patch(tree, eval, True)
                trajectory = get_trajectory(rollout)
                if not trajectory:
                    raise ValueError(f'trajectory is empty for instance_id: {instance_id}')
                answer2 = perspective_agent.encode_modify(instance_id, rollout=trajectory[0], patch=patch)
                answer.update(answer2)
                exp_tree[instance_id].append(answer)
                exp_tree[instance_id][0]['flag'] = 'success'
                flag = True
                break
        if flag == False:
            rollout = get_failed_rollout(tree, True)
            trajectory = get_trajectory(rollout)
            if not trajectory:
                raise ValueError(f'trajectory is empty for instance_id: {instance_id}')
            answer = perspective_agent.encode_perspective(instance_id, rollout=trajectory[0], patch=golden_patch, flag='failed')
            exp_tree[instance_id].append(answer)
            exp_tree[instance_id][0]['flag'] = 'failed'
        
        print(json.dumps(exp_tree[instance_id][0], indent=4))
        exp_tree[instance_id][0]['issue'] = issue
        print('-' * 100)
        save2json(exp_tree, 'SWE-Exp/tmp/verified_experience_tree.json')