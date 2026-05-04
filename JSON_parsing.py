from pydantic import BaseModel
import json
import sys


class Parameters(BaseModel):
    type: str


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameters]
    returns: Parameters


class Prompt(BaseModel):
    prompt: str


def load_JSON(path: str) -> list:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("[ERROR] - File not found.")
        sys.exit(1)


def load_functions(path: str) -> list[Function]:
    try:
        res = [Function(**item) for item in load_JSON(path)]
    except Exception as e:
        print(f"[ERROR] - Incorrect function definition, detail:\n{e}.")
        sys.exit(1)
    return res


def load_prompts(path: str) -> list[Prompt]:
    try:
        res = [Prompt(**item) for item in load_JSON(path)]
    except Exception as e:
        print(f"[ERROR] - Incorrect prompt detected, detail:\n{e}.")
        sys.exit(1)
    return res
