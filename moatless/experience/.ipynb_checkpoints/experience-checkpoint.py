import os
import json
from .silin_prompt import *

def get_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)  # 解析 JSON 文件内容为 Python 对象
            # print("JSON 文件内容：")
            # print(json.dumps(data, indent=4, ensure_ascii=False))  # 格式化输出
            return data
    except FileNotFoundError:
        print(f"错误：文件 {path} 未找到。")
    except json.JSONDecodeError:
        print(f"错误：文件 {path} 不是有效的 JSON 格式。")
    except Exception as e:
        print(f"发生错误：{e}")


def extract_experience(repo, issue, traj, ref, failed=True):
    messages = []
    if failed:
        messages.insert(0, {"role": "system", "content": reflection_system_prompt_1})
    else:
        messages.insert(0, {"role": "system", "content": reflection_system_prompt_2})
    user_prompt = '''
Repository: 
{repo}\n
Issue:
{issue}\n
Trajectory:
{trajectory}\n
Output:
            '''
    messages.append({"role": "user", "content": user_prompt.format(repo=repo, issue=issue, trajectory=traj)})
    output = ref._litellm_base_completion(
                    messages=messages, response_format={"type": "json_object"}
                )
    return output.choices[0].message.content


def get_trajectory(search_tree):
    # 总结一棵树所有的经验到一个list
    traj = []
    i = 0
    def dfs(node, traj):
        if not node:
            return
        if node.completions:
            nonlocal i
            i += 1
            thought = node.completions['build_action'].response['choices'][0]['message']['content']
            feedback = node.completions['value_function'].response['choices'][0]['message']['content']
            # print(f'{node.completions['build_action'].response['choices'][0]['message']['content']}')
            # print(f'{node.completions['value_function'].response['choices'][0]['message']['content']}')
            traj.append(f'In the Stage {i},\n' + 'Action:' + thought + '\n' + 'Result: ' + feedback + '\n\n')
        
        for c in node.children:
            dfs(c, traj)
    dfs(search_tree.root, traj)
    traj = ''.join(traj)
    return traj


def get_trajectory_2(search_tree):
    # 总结多次rollout分别到不同的list
    trajs = []
    def dfs(node, traj, step):
        if not node:
            return

        if not node.children:
            step += 1
            thought = node.completions['build_action'].response['choices'][0]['message']['content']
            feedback = node.completions['value_function'].response['choices'][0]['message']['content']
            traj.append(f'In the Stage {step},\n' + 'Action:' + thought + '\n' + 'Result: ' + feedback + '\n\n')
            trajs.append(traj)
            return
        
        for child in node.children:
            new_traj = traj.copy()
            if node.completions:
                thought = node.completions['build_action'].response['choices'][0]['message']['content']
                feedback = node.completions['value_function'].response['choices'][0]['message']['content']
                new_traj.append(f'In the Stage {step},\n' + 'Action:' + thought + '\n' + 'Result: ' + feedback + '\n\n')
            dfs(child, new_traj, step+1)
    dfs(search_tree.root, [], 0)
    trajs = [''.join(i) for i in trajs]
    return trajs
    
def get_save_experience(name, search_tree, reflection_model, experience_path, failed=True):
    traj = get_trajectory(search_tree)

    out = extract_experience(repo=name, issue=search_tree.root.user_message, traj=traj, ref=reflection_model, failed=failed)
    exp = out.strip()
    exp = json.loads(exp)

    # os.makedirs(os.path.dirname(experience_path), exist_ok=True)
    # with open(experience_path, "w", encoding="utf-8") as file:
    #     json.dump(exp, file, ensure_ascii=False, indent=4)
        
    os.makedirs(os.path.dirname(experience_path), exist_ok=True)

    if os.path.exists(experience_path):
        with open(experience_path, "r", encoding="utf-8") as file:
            existing_exp = json.load(file)
        
        for key in exp:
            if key in existing_exp:
                existing_exp[key].extend(exp[key])
            else:
                existing_exp[key] = exp[key]
        final_exp = existing_exp
    else:
        final_exp = exp
        
    with open(experience_path, "w", encoding="utf-8") as file:
        json.dump(final_exp, file, ensure_ascii=False, indent=4)


def save2json(data, path):
    directory = os.path.dirname(path)
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def format_experience(exp):
    experience = ''
    for k, l in exp.items():
        experience += k + ':\n'
        for i in l:
            experience += i + '\n'
        experience += '\n'
    return experience