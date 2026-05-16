from .JSON_parsing import Function, Prompt
from llm_sdk import Small_LLM_Model
from math import inf
import numpy as np
import json
import pathlib


def output(model: Small_LLM_Model, prompt: str,
           fn_list: list[Function]) -> dict[str, object]:
    """Generate the structured answer for one prompt.

    Args:
        model: The language model used for token generation.
        prompt: The prompt to resolve.
        fn_list: The available function definitions.

    Returns:
        dict[str, object]: The generated prompt, name, and parameters.
    """
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
    """Constrain generation to one of the available function names.

    Args:
        model: The language model used for token selection.
        pre_prompt: The encoded prompt context.
        template: The encoded response template.
        fn_list: The available function definitions.

    Returns:
        list[int]: The encoded function name tokens.
    """
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
    """Generate parameters that match the selected function schema.

    Args:
        model: The language model used for token generation.
        pre_prompt: The encoded prompt context.
        template: The encoded response template.
        fn: The selected function definition.

    Returns:
        dict[str, object]: The generated parameter values.
    """
    context = pre_prompt + template
    len_params = len(fn.parameters)
    res: dict[str, object] = {}

    actual_nb_param = 0
    for k, v in fn.parameters.items():
        chosen_tokens: list[int] = []
        actual_nb_param += 1
        encoded_key = model.encode(f'"{k}": ')[0].tolist()
        context += encoded_key

        if v.type == "number" or v.type == "integer":
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
    """Append the next JSON separator token for the current parameter.

    Args:
        model: The language model used to encode separator tokens.
        actual_param: The current parameter index.
        total_param: The total number of parameters.
        token_list: The token buffer to update.

    Returns:
        None: The token list is updated in place.
    """
    if actual_param != total_param:
        comma = model.encode(",")[0].tolist()
        token_list += comma
    else:
        bracket = model.encode("}")[0].tolist()
        token_list += bracket


def run(prompts: list[Prompt], fn: list[Function], output_path: str) -> None:
    """Process all prompts and write the results to disk.

    Args:
        prompts: The prompts to process.
        fn: The available function definitions.
        output_path: The destination file path.

    Returns:
        None: The results are written to disk.
    """
    model = Small_LLM_Model()
    final_JSON = []

    # For each prompt we generate a type JSON output
    for p in prompts:
        res = output(model, p.prompt, fn)
        final_JSON.append(res)

    # Create the output directory
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(final_JSON, f, indent=4)
