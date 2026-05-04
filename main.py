from JSON_parsing import load_prompts, load_functions

f_call_path = "data/input/function_calling_tests.json"
f_def_path = "data/input/functions_definition.json"

prompts_list = load_prompts(f_call_path)
functions_list = load_functions(f_def_path)

print(prompts_list)

# Pour avoir que la liste d'ID quand on utilise encode() : encodage[0].tolist()
