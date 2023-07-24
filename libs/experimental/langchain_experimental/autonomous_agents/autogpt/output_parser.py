import json
import re
from abc import abstractmethod
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser


class AutoGPTAction(NamedTuple):
    name: str
    args: Dict

class AutoGPTThoughts(NamedTuple):
    name: str
    args: Dict


class BaseAutoGPTOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> AutoGPTAction:
        """Return AutoGPTAction"""


def preprocess_json_input(input_str: str) -> str:
    """Preprocesses a string to be parsed as json.

    Replace single backslashes with double backslashes,
    while leaving already escaped ones intact.

    Args:
        input_str: String to be preprocessed

    Returns:
        Preprocessed string
    """
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    # removing AI comment,if it is contained response.
    start = 0
    end = len(corrected_str)
    for str_i in range(len(corrected_str)):
        if corrected_str[str_i]=='{':
            start = str_i
            break
    for str_i in range(len(corrected_str))[::-1]:
        if corrected_str[str_i]=='}':
            end = str_i
            break
    return corrected_str[start:end+1]

class AutoGPTOutputThoughtParser(BaseAutoGPTOutputParser):
    def parse(self, text: str) -> AutoGPTAction:
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(text)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
            except Exception:
                return AutoGPTAction(
                    name="ERROR",
                    args={"content": f"Could not parse invalid json: {text}"},
                )
        try:
            return AutoGPTThoughts(
                name = 'thoughts',
                args = {'content':parsed["thoughts"]}
            )
        except (KeyError, TypeError):
            # If the command is null or incomplete, return an erroneous tool
            return AutoGPTThoughts(
                name="ERROR", args={"content": f"Incomplete thoughts: {parsed}"}
            )



class AutoGPTOutputParser(BaseAutoGPTOutputParser):
    def parse(self, text: str) -> AutoGPTAction:
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(text)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
            except Exception:
                return AutoGPTAction(
                    name="ERROR",
                    args={"error": f"Could not parse invalid json: {text}"},
                )
        try:
            return AutoGPTAction(
                name=parsed["command"]["name"],
                args=parsed["command"]["args"],
            )
        except (KeyError, TypeError):
            # If the command is null or incomplete, return an erroneous tool
            return AutoGPTAction(
                name="ERROR", args={"error": f"Incomplete command args: {parsed}"}
            )
