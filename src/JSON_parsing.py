from pydantic import BaseModel
import json
import sys
from typing import Any


class Parameters(BaseModel):
    """Describe the schema of a function parameter.

    Attributes:
        type: The expected JSON type for the parameter.
    """
    type: str


class Function(BaseModel):
    """Represent a callable function definition.

    Attributes:
        name: The function name.
        description: A short description of the function.
        parameters: The parameter schema keyed by parameter name.
        returns: The return value schema.
    """
    name: str
    description: str
    parameters: dict[str, Parameters]
    returns: Parameters


class Prompt(BaseModel):
    """Represent a prompt entry loaded from JSON.

    Attributes:
        prompt: The user prompt to process.
    """
    prompt: str


def load_JSON(path: str) -> list[dict[str, Any]]:
    """Load a JSON file and return its list of objects.

    Args:
        path: Path to the JSON file to read.

    Returns:
        list[dict[str, Any]]: The validated list of JSON objects.
    """
    try:
        with open(path, 'r') as f:
            data: object = json.load(f)
    except FileNotFoundError:
        print("[ERROR] - File not found.")
        sys.exit(1)

    if not isinstance(data, list):
        raise ValueError("[ERROR] - JSON root must be a list.")

    items: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("[ERROR] - JSON items must be objects.")
        items.append(item)

    return items


def load_functions(path: str) -> list[Function]:
    """Load function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        list[Function]: The parsed function definitions.
    """
    try:
        res = [Function(**item) for item in load_JSON(path)]
    except Exception as e:
        print(f"[ERROR] - Incorrect function definition, detail:\n{e}.")
        sys.exit(1)
    return res


def load_prompts(path: str) -> list[Prompt]:
    """Load prompt entries from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        list[Prompt]: The parsed prompt entries.
    """
    try:
        res = [Prompt(**item) for item in load_JSON(path)]
    except Exception as e:
        print(f"[ERROR] - Incorrect prompt detected, detail:\n{e}.")
        sys.exit(1)
    return res
