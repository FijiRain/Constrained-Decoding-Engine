from JSON_parsing import load_prompts, load_functions, Function
from llm_sdk import Small_LLM_Model
from math import inf
import numpy as np
import json


def output(model: Small_LLM_Model, prompt: str,
           fn_list: list[Function]) -> dict[str, object]:
    main_template = "{" + '"prompt": ' + json.dumps(prompt) + ', "name": '
    functions = {fn.name: fn.model_dump() for fn in fn_list}
    pre_prompt = ("You are a function calling agent.\n"
                  "TASK: Analyze the prompt and call the correct function.\n"
                  "CONSTRAINTS:\n"
                  "- Call exactly ONE function\n"
                  "- Use only the available functions below\n"
                  "- Generate the function NAME and its PARAMETERS\n"
                  "- Output format: {\"name\": \"function_name\", "
                  "\"parameters\": {...}}\n\n"
                  f"Available functions:\n{functions}\n\n"
                  f"Prompt: {prompt}\n"
                  f"Answer: ")

    encoded_pre_prompt = model.encode(pre_prompt)[0].tolist()
    encoded_template = model.encode(main_template)[0].tolist()

    encoded_name = enforce_name(model, encoded_pre_prompt,
                                encoded_template, fn_list)
    encoded_template += encoded_name

    params_template = ', "parameters": {'
    encoded_template += model.encode(params_template)[0].tolist()

    right_fn = None
    for fn in fn_list:
        if fn.name == model.decode(encoded_name):
            right_fn = fn
    if right_fn is None:
        raise ValueError("No matching function found for generated name.")

    encoded_params = enforce_params(model, encoded_pre_prompt,
                                    encoded_template, right_fn)

    answer = {
        "prompt": prompt,
        "name": model.decode(encoded_name),
        "parameters": {model.decode(encoded_params)},
    }
    return answer


def enforce_name(model: Small_LLM_Model, pre_prompt: list[int],
                 template: list[int], fn_list: list[Function]) -> list[int]:
    encoded_fn_names = [model.encode(i.name)[0].tolist() for i in fn_list]
    chosen_name_tokens: list[int] = []

    context = pre_prompt + template

    finished = False
    while not finished:
        allowed_tokens = set()
        size = len(chosen_name_tokens)

        # Allowed token selection
        for function in encoded_fn_names:
            if size < len(function) and chosen_name_tokens == function[:size]:
                allowed_tokens.add(function[size])

        # Mask logits using allowed tokens
        logits = model.get_logits_from_input_ids(context + chosen_name_tokens)
        masked = [weight if token in allowed_tokens else -inf
                  for token, weight in enumerate(logits)]
        next = int(np.argmax(masked))

        chosen_name_tokens.append(next)

        # Check if a full function name has been generated
        for function in encoded_fn_names:
            if chosen_name_tokens == function:
                finished = True

        # Print LLM generated token
        print(model.decode(chosen_name_tokens))

    return chosen_name_tokens


def enforce_params(model: Small_LLM_Model, pre_prompt: list[int],
                   template: list[int], fn: Function) -> list[int]:
    context = pre_prompt + template
    len_params = len(fn.parameters)
    res: list[int] = []

    actual_nb_param = 0
    for k, v in fn.parameters.items():
        chosen_tokens: list[int] = []
        actual_nb_param += 1
        encoded_key = model.encode(f'"{k}": ')[0].tolist()
        context += encoded_key
        res += encoded_key

        if v.type == "number":
            allowed_char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                            ".", "-"]
            allowed_token = [model.encode(i)[0].tolist()[0]
                             for i in allowed_char]
            while True:
                logits = model.get_logits_from_input_ids(
                    context + chosen_tokens)
                next_token = int(np.argmax(logits))
                if next_token in allowed_token:
                    chosen_tokens.append(next_token)
                    res.append(next_token)
                else:
                    break
            if actual_nb_param != len_params:
                comma = model.encode(",")[0].tolist()
                chosen_tokens += comma
                res += comma
            else:
                bracket = model.encode("}")[0].tolist()
                chosen_tokens += bracket
                res += bracket
            context += chosen_tokens

        elif v.type == "string":
            ...
        else:
            raise ValueError("Unknown parameter type in functions_definition.")

    print(model.decode(res))

    return res


# a virer et mettre dans le main avec --input etc
f_call_path = "data/input/function_calling_tests.json"
f_def_path = "data/input/functions_definition.json"

prompts_list = load_prompts(f_call_path)
functions_list = load_functions(f_def_path)

model = Small_LLM_Model()
final_JSON = []

# Pour chaque prompt on generera une sortie type JSON
for p in prompts_list:
    res = output(model, p.prompt, functions_list)
    final_JSON.append(res)

with open("data/input/function_calling_results.json", "w") as f:
    json.dump(final_JSON, f, indent=2)

print(final_JSON)



# logits = model.get_logits_from_input_ids(
#     context + chosen_nb_tokens)
# masked = [weight if token in allowed_token else -inf
#           for token, weight in enumerate(logits)]
# next_token = int(np.argmax(masked))
# chosen_nb_tokens.append(next_token)