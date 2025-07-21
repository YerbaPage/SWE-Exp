from dotenv import load_dotenv
load_dotenv(".env")
import os
import json
import random
import argparse
import logging
from datetime import datetime
from typing import Dict

from moatless.benchmark.utils import get_moatless_instance
from moatless.benchmark.swebench import create_repository
from moatless.index.code_index import CodeIndex
from moatless.index.settings import IndexSettings

from moatless.file_context import FileContext
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.value_function.base import ValueFunction

from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, RunTests, StringReplace, CreateFile
from moatless.agent.agent import ActionAgent
from moatless.search_tree import SearchTree
from moatless.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)

from moatless.experience.get_save_json import get_json, save2json
from moatless.discriminator import AgentDiscriminator

from moatless.experience.prompts.agent_prompts import (
    INSTRUCTOR_ROLE, ASSISTANT_ROLE, INSTRUCTION_GUIDELINES, ASSISTANT_GUIDELINES, INSTRUCTOR_FORMAT,
)
from moatless.experience.instructor import Instructor
from moatless.experience.exp_agent.select_agent import SelectAgent
from moatless.experience.prompts.exp_prompts import select_exp_system_prompt, select_exp_user_prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
            else:
                logger.warning(f"Empty line found in {file_path}, \nline{line}")
    return data_list

def save_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def add_data_to_jsonl(file_path, new_data):
    if file_path and os.path.exists(file_path):
        data_list = load_jsonl(file_path)
    else:
        data_list = []

    if isinstance(new_data, list):
        data_list.extend(new_data)
    else:
        data_list.append(new_data)

    save_to_jsonl(data_list, file_path)


def main(instance_id, max_iterations, max_finish_nodes, max_expansions):
    completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
    discriminator_model = CompletionModel(model="deepseek/deepseek-chat", temperature=1)
    value_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.2)

    completion_model.response_format = LLMResponseFormat.REACT
    value_model.response_format = LLMResponseFormat.REACT
    discriminator_model.response_format = LLMResponseFormat.REACT

    instance = get_moatless_instance(instance_id=instance_id)  
    repository = create_repository(instance)

    code_index = CodeIndex.from_index_name(
        instance["instance_id"], file_repo=repository
    )
    file_context = FileContext(repo=repository)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    instance_path = f'tmp/trajectory/{instance_id}/'
    persist_path = f'tmp/trajectory/{instance_id}/{current_date}_trajectory.json'
    experience_path = f"tmp/experience/{instance_id}/{current_date}_experience.json"

    value_function = ValueFunction(completion_model=value_model)
    actions = [
        FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
        FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
        SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
        ViewCode(completion_model=completion_model, repository=repository),
        StringReplace(repository=repository, code_index=code_index),
        Finish(),
    ]


    agent = ActionAgent(system_prompt=ASSISTANT_ROLE + ASSISTANT_GUIDELINES, 
                        actions=actions, completion=completion_model)

    discriminator = AgentDiscriminator(
        completion=discriminator_model,
        n_agents=5,
        n_rounds=3,
    )

    feedback_generator = FeedbackAgent(
        completion_model=completion_model, instance_dir=instance_path
    )

    max_depth = min(max_iterations, 20)
    instructor = Instructor(completion=completion_model, system_prompt=INSTRUCTOR_ROLE + INSTRUCTION_GUIDELINES, output_format=INSTRUCTOR_FORMAT, 
                            task=instance['problem_statement'], taken_actions=max_depth)

    search_tree = SearchTree.create(
        message=instance["problem_statement"],
        assistant=agent,
        instructor=instructor,
        file_context=file_context,
        value_function=value_function,
        discriminator=discriminator,
        feedback_generator=feedback_generator,
        max_finished_nodes=max_finish_nodes,
        max_iterations=max_iterations,
        max_expansions=max_expansions,
        max_depth=max_depth,
        persist_path=persist_path,
    )

    select_agent = SelectAgent(completion=completion_model, instance_id=instance_id,
                            select_system_prompt=select_exp_system_prompt,
                            user_prompt=select_exp_user_prompt, 
                            exp_path='tmp/het/verified_experience_tree.json', 
                            train_issue_type_path='tmp/het/verified_issue_types_final.json', 
                            test_issue_type_path='tmp/het/verified_issue_types_final.json', 
                            persist_dir=experience_path)
    old_experiences = select_agent.select_workflow(n=1)
    logger.info(f"old_experiences:\n{json.dumps(old_experiences, indent=4)}")
    select_agent.persist({"old_experiences": old_experiences, "HET": {}, "trajectory": persist_path})


    finished_node = search_tree.run_search(select_agent, old_experiences)
    search_tree.persist(persist_path)

    if finished_node:
        if finished_node.file_context.generate_git_patch():
            logger.info(f"{instance} patch: {finished_node.file_context.generate_git_patch()}")
            tmp = {
                "model_name_or_path": "DeepSeek_IA",
                "instance_id": instance_id,
                "model_patch": finished_node.file_context.generate_git_patch(),
                "leaf_id": finished_node.node_id,
                'source_tree_path': persist_path
            }   
            add_data_to_jsonl('prediction_verified.jsonl', tmp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument("--instance_ids", type=str, required=True,
                        help="The file path to instance ID(s)")

    parser.add_argument("--max_iterations", type=int, default=20, help="Max iteration for tree search")

    parser.add_argument("--max_finished_nodes", type=int, default=2, help="Max finished nodes for tree search")

    parser.add_argument("--max_expansions", type=int, default=3, help="Max expansions for tree search")

    args = parser.parse_args()

    with open(args.instance_ids, "r", encoding='utf-8') as f:
        instance_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"{len(instance_ids)} instances found")

    if isinstance(instance_ids, list):
        for instance_id in instance_ids:
            instance_id = instance_id.strip()
            main(instance_id, args.max_iterations, args.max_finished_nodes, args.max_expansions)
            import time
            time.sleep(3)
            logger.info("wait for half minute and then run the next instance")
    elif isinstance(instance_ids, str):
        main(instance_ids, args.max_iterations, args.max_finished_nodes, args.max_expansions)
    

    logger.info('All Finished')