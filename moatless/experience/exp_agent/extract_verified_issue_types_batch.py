import json
import time
import re
import json_repair
import logging
import os
import argparse
from typing import Dict, Any
from moatless.completion.completion import CompletionModel
from moatless.benchmark.utils import get_moatless_instance
from moatless.experience.prompts.exp_prompts import (
    issue_type_system_prompt,
    issue_type_user_prompt,
)
from pydantic import BaseModel, Field, PrivateAttr
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

class IssueAgent(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt to be used for generating completions"
    )
    user_prompt: str = Field(
        ..., description="Input prompt to be used for generating completions"
    )
    _completion: CompletionModel = PrivateAttr()

    def __init__(
            self,
            completion: CompletionModel,
            system_prompt: str,
            user_prompt: str,
            **data,
    ):
        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
    def _analyze_with_retry(self, issue):
        user_prompt = self.user_prompt.format(issue)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}]
        try:
            response = self._completion._litellm_base_completion(
                messages=messages
            )
            response = response.choices[0].message.content
            response = self.simplify(response)
            response = json_repair.loads(response)

            issue_type, description = response.get("issue_type"), response.get('description')

        except Exception as e:
            logger.error(f"Error in _analyze_with_retry: {str(e)}")
            logger.error(f"Response: {response}")
            raise

        return {"issue_type": issue_type, "description": description}
    def analyze(self, issue):
        return self._analyze_with_retry(issue)

    @property
    def completion(self) -> CompletionModel:
        return self._completion

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["completion"] = self._completion.model_dump(**kwargs)
        return dump


def extract_issue_types_batch(start_idx=0, end_idx=None, resume_from_file=None):
    
    with open("SWE-Exp/verified_dataset_ids.txt", "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]

    if end_idx is None:
        end_idx = len(ids)
    
    ids_to_process = ids[start_idx:end_idx]
    
    completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
    issue_agent = IssueAgent(system_prompt=issue_type_system_prompt, user_prompt=issue_type_user_prompt, completion=completion_model)

    issue_type_verified = {}
    if resume_from_file and os.path.exists(resume_from_file):
        with open(resume_from_file, 'r', encoding='utf-8') as f:
            issue_type_verified = json.load(f)
        print(f"Loaded {len(issue_type_verified)} processed instances from file")
    
    print(f"Start processing instances from {start_idx} to {end_idx-1}, total {len(ids_to_process)} instances")
    
    for idx, i in enumerate(ids_to_process):
        actual_idx = start_idx + idx
        
        if i in issue_type_verified:
            print(f"Skip processed instance: {i}")
            continue
            
        try:
            print(f"Processing instance {actual_idx + 1}/{len(ids)}: {i}")
            
            instance = get_moatless_instance(split='verified', instance_id=i)
            issue = instance['problem_statement']
            
            answer = issue_agent.analyze(issue)
            
            print(f"Instance ID: {i}")
            print(f"Issue: {issue[:200]}...")
            print(f"Analysis result: {json.dumps(answer, indent=4)}")
            print('-' * 100)
            
            answer['issue'] = issue
            issue_type_verified[i] = answer
            
            if (idx + 1) % 5 == 0:
                save_batch_results(issue_type_verified, start_idx, actual_idx + 1)
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Error processing instance {i}: {str(e)}")
            issue_type_verified[i] = {"error": str(e), "issue_type": None, "description": None}
            continue
    
    save_batch_results(issue_type_verified, start_idx, end_idx)
    return issue_type_verified

def save_batch_results(results, start_idx, end_idx):
    filename = f"SWE-Exp/tmp/verified_issue_types_batch_{start_idx}_{end_idx}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved batch results to: {filename}")

def merge_batch_files():
    import glob
    import os
    
    pattern = "SWE-Exp/tmp/verified_issue_types_batch_*.json"
    batch_files = glob.glob(pattern)
    
    merged_results = {}
    for batch_file in sorted(batch_files):
        print(f"Merging file: {batch_file}")
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
            merged_results.update(batch_data)
    
    merged_filename = "SWE-Exp/tmp/verified_issue_types_merged.json"
    with open(merged_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=4)
    
    print(f"Merged {len(merged_results)} instances, saved to: {merged_filename}")    
    return merged_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract issue types from verified dataset')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, help='End index')
    parser.add_argument('--resume', type=str, help='Resume file path')
    parser.add_argument('--merge', action='store_true', help='Merge all batch files')
    
    args = parser.parse_args()
    
    if args.merge:
        merge_batch_files()
    else:
        extract_issue_types_batch(args.start, args.end, args.resume) 