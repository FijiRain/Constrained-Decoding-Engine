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
                  "If you need to use a regex "
                  "call the correct REGEX expression\n"
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

    answer: dict[str, object] = {
        "prompt": prompt,
        "name": model.decode(encoded_name),
        "parameters": encoded_params,
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
        # print(model.decode(chosen_name_tokens))

    return chosen_name_tokens


def enforce_params(model: Small_LLM_Model, pre_prompt: list[int],
                   template: list[int], fn: Function) -> dict[str, object]:
    context = pre_prompt + template
    len_params = len(fn.parameters)
    res: dict[str, object] = {}

    actual_nb_param = 0
    for k, v in fn.parameters.items():
        chosen_tokens: list[int] = []
        actual_nb_param += 1
        encoded_key = model.encode(f'"{k}": ')[0].tolist()
        context += encoded_key

        if v.type == "number":
            allowed_char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                            ".", "-"]
            allowed_token = [model.encode(i)[0].tolist()[0]
                             for i in allowed_char]
            numeric_tokens: list[int] = []
            while True:
                logits = model.get_logits_from_input_ids(
                    context + chosen_tokens)
                next_token = int(np.argmax(logits))
                if next_token in allowed_token:
                    chosen_tokens.append(next_token)
                    numeric_tokens.append(next_token)
                else:
                    break
            res[k] = float(model.decode(numeric_tokens))
            context += chosen_tokens
            add_separator(model, actual_nb_param, len_params, chosen_tokens)

        elif v.type == "string":
            str_tokens: list[int] = []
            context += model.encode('"')[0].tolist()
            chosen_tokens += model.encode('"')[0].tolist()

            while True:
                logits = model.get_logits_from_input_ids(
                    context + chosen_tokens)
                next_token = int(np.argmax(logits))
                if '"' in model.decode([next_token]):
                    chosen_tokens.append(next_token)
                    # str_tokens.append(next_token)
                    break
                else:
                    chosen_tokens.append(next_token)
                    str_tokens.append(next_token)
            res[k] = model.decode(str_tokens)
            context += chosen_tokens
            add_separator(model, actual_nb_param, len_params, chosen_tokens)

        else:
            raise ValueError("Unknown parameter type in functions_definition.")
    print(res)
    return res


def add_separator(model: Small_LLM_Model, actual_param: int,
                  total_param: int, token_list: list[int]) -> None:
    if actual_param != total_param:
        comma = model.encode(",")[0].tolist()
        token_list += comma
    else:
        bracket = model.encode("}")[0].tolist()
        token_list += bracket


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
    json.dump(final_JSON, f, indent=4)

# print(final_JSON)
