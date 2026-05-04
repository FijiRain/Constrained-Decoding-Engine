from llm_sdk.llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel

# transformer = Small_LLM_Model()

class Function(BaseModel):
    name: str



f_call_path = "data/input/function_calling_tests.json"
try:
    with open(f_call_path, 'r') as f:
        dict_prompts = json.load(f)
        prompts_list = []
        for k in dict_prompts:
            prompts_list.append(k["prompt"])

        print(prompts_list)

except FileNotFoundError:
    print("[ERROR] - File not found.")


# Pour avoir que la liste d'ID quand on utilise encode() : encodage[0].tolist()
