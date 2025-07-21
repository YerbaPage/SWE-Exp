import os
import json

def get_json(path):
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


def save2json(data, path):
    directory = os.path.dirname(path)
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)