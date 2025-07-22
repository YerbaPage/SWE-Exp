import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(current_dir, "../../..", ".env"))
load_dotenv(dotenv_path=env_path)

import re
import logging
import json
import random
import torch
import json_repair
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

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
    select_exp_system_prompt,
    select_exp_user_prompt,
)
from moatless.search_tree import SearchTree
from moatless.node import Node, generate_ascii_tree
from moatless.runtime.testbed import TestbedEnvironment
from moatless.benchmark.swebench import create_repository

logger = logging.getLogger(__name__)


class SelectAgent(BaseModel):
    instance_id: str = Field(
        ..., description="the instance id of the current issue"
    )
    select_system_prompt: str = Field(
        ..., description="System prompt to select experience"
    )
    user_prompt: str = Field(
        ..., description="Input prompt to be used for generating completions"
    )
    exp_path: str = Field(
        ..., description="The path of Hierarchical Experience Tree"
    )
    train_issue_type_path: str = Field(
        ..., description="The path of Issue Type for train"
    )
    test_issue_type_path: str = Field(
        ..., description="The path of Issue Type for test"
    )
    persist_dir: Optional[str] = Field(
        default=None, description="The path to save the experience tree"
    )
    _completion: CompletionModel = PrivateAttr()

    def __init__(
            self,
            completion: CompletionModel,
            instance_id: str,
            select_system_prompt: str,
            user_prompt: str,
            exp_path: str,
            train_issue_type_path: str,
            test_issue_type_path: str,
            persist_dir:str,
            **data,
    ):
        super().__init__(
            instance_id=instance_id,
            select_system_prompt=select_system_prompt,
            user_prompt=user_prompt,
            exp_path=exp_path,
            train_issue_type_path=train_issue_type_path,
            test_issue_type_path=test_issue_type_path,
            persist_dir=persist_dir,
            **data,
        )
        self._completion = completion

    def simplify(self, content):
        match = re.search(r"```(?:\w+)?\n?(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return content

    def persist(self, experiences: Dict):
        directory = os.path.dirname(self.persist_dir)

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.persist_dir, "w", encoding="utf-8") as file:
            json.dump(experiences, file, ensure_ascii=False, indent=4)

    def get_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"Wrong: file {path} not found")
        except json.JSONDecodeError:
            print(f"Wrong: file {path} is not a valid JSON format")
        except Exception as e:
            print(f"Error: {e}")

    def screening(self, pre_issues: Dict, cur_issue: Dict, k = 10):
        import torch
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct",
                                                  cache_dir='tmp/models')
        
        def try_load_model_on_gpus():
            model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct",
                                              cache_dir='tmp/models')
            
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model = model.to(device)
                logger.info("Model loaded to GPU")
            else:
                device = torch.device('cpu')
                model = model.to(device)
                logger.info("CUDA is not available, model loaded to CPU")
            
            return model, device
        
        try:
            model, device = try_load_model_on_gpus()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise e

        def average_pool(last_hidden_states: Tensor,
                         attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        def formalize(issues):
            tmp = []
            ids = []
            for id, data in issues.items():
                ids.append(id)
                tmp.append(f"issue_type: {data['issue_type']} \ndescription: {data['description']}")
            return tmp, ids

        task = "Given the prior issues, your task is to analyze a current issue's problem statement and select the most relevant prior issue that could help resolve it."
        cur_query, _ = formalize(cur_issue)
        queries = [
            get_detailed_instruct(task, cur_query[0])
        ]
        documents, ids = formalize(issues=pre_issues)
        input_texts = queries + documents

        batch_dict = tokenizer(input_texts, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        
        # Move input tensor to GPU
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():  # No gradient during inference, save memory
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:1] @ embeddings[1:].T) * 100
            
        id2score = {}
        for i in range(len(ids)):
            id2score[ids[i]] = scores.cpu().tolist()[0][i]  # Move back to CPU to get results
        id2score = sorted(id2score.items(), key=lambda x: x[1], reverse=True)
        topkids = [k for k, v in id2score[:k]]
        return id2score, topkids


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
            logger.error(f"Error in _generalize_with_retry: {str(e)}")
            logger.error(f"Response: {response}")
            raise

        return response

    def select_perspective(self, pre_issues: Dict, cur_issue, k):
        pre_issues_str = ""
        for id, data in pre_issues.items():
            tmp = f"***issue_id***: {id}\n***issue_description***: {data['issue']}\n***Experiences***:\n"
            if data.get("perspective"):
                if isinstance(data["perspective"], str):
                    tmp += f"   - Perspective: {data['perspective']}\n"
                elif isinstance(data["perspective"], list):
                    for i in range(len(data["perspective"])):
                        tmp += f"   - Reflection {str(i)}: {data['perspective'][i]}\n"
            tmp += '\n\n'
            pre_issues_str += tmp
        user_prompt = self.user_prompt.format(pre_issues_str, cur_issue)
        messages = [{"role": "system", "content": self.select_system_prompt.format(k=k)},
                    {"role": "user", "content": user_prompt}]
        return self._chat_with_retry(messages)

    def generalize_perspective(self, pre_issue: Dict, cur_issue: str, last_exp):
        generalize_exp_system_prompt = f'''
You are a knowledgeable issue resolution assistant. Your task is to analyze a current issue and generalize the received experiences into a new insight that is applicable to this issue.

You will be given:
- A `problem_statement` describing the current issue
- A past trajectory with:
  - `issue_description`: The description of the past issue
  - `experience`: Either a `perspective` (how this successful trajectory understood this issue) with `entry_point` (the first specific object extracted from the issue and successfully found in the code environment) or `reflections` (insights gained from an unsuccessful trajectory)

Your job is to:
1. Compare the current `problem_statement` with each past trajectory’s `issue_description` and `experience`.
2. Adapt the old experience to the current issue and produce a new applicable experience.
3. Identify the most likely entry point in the codebase - based on the problem statement - that is critical to resolving the current issue.

You must output a JSON object with exactly four fields for each selection:
- `new_experience`: A new experience statement tailored to the current issue, based on the old experience. **The more detailed the better**

Your output must strictly follow the JSON format below:
{{
    "new_experience": "<the new experience (1-2 sentences)>",
}}
'''
        pre_issues_str = f"issue_description: {pre_issue['issue']}\n"
        if pre_issue['flag'] == 'success':
            pre_issues_str += f"perspective: {pre_issue['perspective']}\n"
        elif pre_issue['flag'] == 'failed':
            for i in range(len(pre_issue['perspective'])):
                pre_issues_str += f"reflection {str(i+1)}: {pre_issue['perspective'][i]}"
        pre_issues_str += '\n\n'
        user_prompt = self.user_prompt.format(pre_issues_str, cur_issue)
        messages = [{"role": "system", "content": generalize_exp_system_prompt},
                    {"role": "user", "content": user_prompt}]
        return self._chat_with_retry(messages)


    def load_experience(self, exp: Dict, type) -> str:
        if type == 'perspective':
            if isinstance(exp[type], str):
                return f'***Experience***: {exp[type]}\n'
            elif isinstance(exp[type], list):
                tmp = ''
                for i in range(len(exp[type])):
                    tmp += f'***Experience {str(i + 1)}***: {exp[type][i]}\n'
                return tmp
        elif type == 'positioning':
            if isinstance(exp[type], dict):
                tmp = ''
                for k, v in exp[type].items():
                    tmp += f'***{k}***: {v}\n'
                return tmp
            elif isinstance(exp[type], list):
                tmp = ''
                for i in range(len(exp[type])):
                    tmp += f'***Experience {str(i + 1)}***: {exp[type][i]}\n'
                return tmp
        elif type == 'modification':
            if isinstance(exp[type], dict):
                tmp = ''
                for i in range(len(exp[type]['experience'])):
                    tmp += f'***Experience {str(i + 1)}***: {exp[type]['experience'][i]}\n'
                return tmp
            elif isinstance(exp[type], list):
                tmp = ''
                for i in range(len(exp[type])):
                    tmp += f'***Experience {str(i + 1)}***: {exp[type][i]}\n'
                return tmp
            else:
                return f'***Experience 1***: {str(exp[type])}\n'


    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(60),
        retry=retry_if_exception_type((torch.cuda.OutOfMemoryError, RuntimeError, KeyError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def select_workflow(self, n=1) -> Dict:
        if n > 10:
            logger.error(f"the number of return experiences shouldn't be greater than 10")
        exp_tree = self.get_json(self.exp_path)
        train_issue_type = self.get_json(self.train_issue_type_path)
        test_issue_type = self.get_json(self.test_issue_type_path)

        instance = get_moatless_instance(instance_id=self.instance_id)
        cur_issue = instance['problem_statement']
        # Filter out instances in the same repository
        issue_type_tmp = {}
        repo = self.instance_id.split('__')[0]
        for k, v in train_issue_type.items():
            if repo not in k:
                issue_type_tmp[k] = v
        id2score, topkids = self.screening(pre_issues=issue_type_tmp, cur_issue={self.instance_id: test_issue_type[self.instance_id]})
        select_issues = {}
        for i in topkids:
            select_issues[i] = exp_tree[i][0]
        answer = self.select_perspective(pre_issues=select_issues, cur_issue=cur_issue, k=n)
        logger.info(f'answer:\n{json.dumps(answer, indent=4)}')
        return {k: exp_tree[k][0] for k, v in answer.items()}



    def polish_workflow(self, old_experiences: Dict, type, history, instruction):
        issue_type = self.get_json(self.test_issue_type_path)
        cur_issue = issue_type[self.instance_id]['issue']
        enhance_system_prompt = f'''
You are a strategic assistant helping an agent improve its next-step instruction in a debugging task.

You are given:
- A `problem_statement`: a natural language description of the current software problem
- A `current_code_exploration_history`: The recent exploration steps taken to understand or debug the current codebase. This may include what has been examined, eliminated, or hypothesized so far.
- An `instruction`: the next step the agent is expected to take
- A list of `experiences`: each offering past insights about how to better approach the corresponding issue.

Your task is to:
1. Analyze how the current `instruction` relates to the given `issue` and `current_code_exploration_history`
2. Identify useful, transferable, generalized insights from the past experiences of type **{type}**
3. Based on those insights, rewrite the instruction to make it more robust, strategically informed, and better suited to succeed in this situation

### Important Notes
    - Focus only on experience of type **{type}**, and ensure the improved instruction aligns with the original goal but incorporates better reasoning or coverage.
    - NEVER add the content that are not related to solving the current problem.
    
Output only the following JSON structure:
{{
  "enhanced_instruction": "<A single improved and robust instruction, rewritten based on relevant experience of type {type}>"
}}
'''
        enhanced_instruction = instruction
        for id, exp in old_experiences.items():
            experiences = f'***Past Issue***: {issue_type[id]['issue']}\n'
            if type == 'perspective':
                experiences += self.load_experience(exp, type)
                experiences += '\n'
            elif type == 'positioning':
                experiences += self.load_experience(exp, type)
                experiences += '\n'
            elif type == 'modification':
                experiences += self.load_experience(exp, type)
                experiences += '\n'
            else:
                logger.error(f"the type of experiences should be perspective, positioning or modification")

        user_prompt = f'''
<problem_statement>
{cur_issue}
</problem_statement>

<current_code_exploration>
{history}
</current_code_exploration>

<instruction>
{enhanced_instruction}
</instruction>

<experiences>
{experiences}
</experiences>
'''
        messages = [{"role": "system", "content": enhance_system_prompt},
                    {"role": "user", "content": user_prompt}]

        enhanced_instruction = self._chat_with_retry(messages)['enhanced_instruction']

        return enhanced_instruction

    def generalize_workflow(self, old_experiences: Dict, type: str, history, cur_code, instruction):
        # if n > 10:
        #     logger.error(f"the number of return experiences shouldn't be greater than 10")
    
        new_experiences = []
        issue_type = self.get_json(self.test_issue_type_path)
        cur_issue = issue_type[self.instance_id]['issue']
        exp_items = [v for k, v in old_experiences.items()]
    
        def generate_single(exp):
            if type == 'perspective':
                return self.generalize_perspective(pre_issue=exp,
                                                   cur_issue=cur_issue, last_exp='')
            elif type == 'modification':
                return self.generalize_modify(pre_issue=exp,
                                              cur_issue=cur_issue, cur_code=cur_code,
                                              modify_instruction=instruction, last_exp='')
            else:
                raise ValueError("type must be 'perspective' or 'modification'")
    
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_single, exp) for exp in exp_items]
    
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Error processing experience {i}: {e}")
                    result = {"new_experience": "Error", "entry_point": ""}
                new_experiences.append(result)
    
        def format_block():
            content = ''
            for i, exp in enumerate(new_experiences):
                content += f"***Experience {i + 1}***: {exp['new_experience']}\n"
    
            return content

        if type == 'perspective':
            return format_block()
        elif type == 'modification':
            return format_block()
        else:
            logger.error("The type of experiences should be 'perspective', 'positioning', or 'modification'")
            return ""



if __name__ == '__main__':
    completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
    with open("/SWE-Exp/moatless/tmp/verified_dataset_ids.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    issue_type = get_json('/SWE-Exp/moatless/tmp/het/split_train_issue_type.json')
    counter = 0 
    for line in lines:
        counter += 1
        if counter % 30 == 0:
            print(f'counter: {counter}')
            instance_id = line.strip()
            select_agent = SelectAgent(completion=completion_model, instance_id=instance_id,
                                    select_system_prompt=select_exp_system_prompt,
                                    user_prompt=select_exp_user_prompt, 
                                    exp_path='SWE-Exp/moatless/tmp/het/verified_experience_tree.json', 
                                    train_issue_type_path='SWE-Exp/moatless/tmp/het/verified_issue_types_final.json', 
                                    test_issue_type_path='SWE-Exp/moatless/tmp/het/verified_issue_types_final.json', 
                                    persist_dir='')
            old_experiences = select_agent.select_workflow(n=5)

            new_experiences = 'Here are some experiences you can refer to:'
            new_experiences = select_agent.generalize_workflow(old_experiences, type='perspective',
                                                                history=None, cur_code=None,
                                                                instruction=None)
            instance = get_moatless_instance(instance_id=instance_id)
            print(f'problem_statement:\n{instance["problem_statement"]}')
            print(f'The coressponding experiences that may be helpful:\n{new_experiences}')
            print('=' * 100)
