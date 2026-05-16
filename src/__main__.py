import argparse
from .JSON_parsing import load_functions, load_prompts
from .LLM_run import run


def main() -> None:
    """Run the command-line interface.

    Parses the command-line arguments, loads the input data, and writes the
    generated results to the output file.

    Returns:
        None: This function is executed for its side effects.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")

    args = parser.parse_args()

    f_call_path = args.input
    f_def_path = args.functions_definition
    f_answer = args.output

    prompts_list = load_prompts(f_call_path)
    functions_list = load_functions(f_def_path)

    run(prompts_list, functions_list, f_answer)


if __name__ == "__main__":
    main()
