"""MLC LLM benchmark dataset classes"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
)
from transformers import AutoTokenizer  # pylint: disable=import-error

from request_record import Metrics, RequestRecord


class Dataset:  # pylint: disable=too-few-public-methods
    """The dataset base class."""

    # We set a truncation limit of 100k.
    truncate_length = int(1e5)
    # For some that datasets (e.g., dataset that has shared common prefix),
    # we need fake warmup requests to avoid prefilling common prefixes to the engine.
    require_fake_warmup: bool = False
    # Whether the dataset contains timestamps already.
    # If the dataset comes with timestamps, the benchmark can just replay
    # the requests according to their timestamps.
    timestamp_available: bool = False

    def generate_request_records(
            self,
            input_len: Optional[int],
            output_len: Optional[int],
            input_len_std: float = 0.0,
            output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        """Get the raw unprocessed request records of the dataset."""
        raise NotImplementedError()


GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

from enum import IntEnum


class Err_type(IntEnum):
    FORMAT_ERROR = 0
    CALL_NUMBER_ERROR = 1
    FUNC_NAME_ERROR = 2
    PARA_KEY_ERROR = 3
    TYPE_ERROR = 4
    ENUM_ERROR = 5
    PARA_VALUE_ERROR = 6
    NONE = 7


class Error:
    def __init__(self, message: str = "", err_type: Err_type = Err_type.NONE):
        self.message = message
        self.error_type = err_type


class GorillaDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for Gorilla dataset.
    Reference: https://github.com/ShishirPatil/gorilla
    """

    def __init__(self, dataset: str, dataset_path: str, tokenizer: AutoTokenizer, use_stag: bool) -> None:
        self.tokenizer = tokenizer
        self.require_fake_warmup = True
        self.gorilla_data = []
        base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data"
        id = 0
        dataset_file = f"{dataset_path}/{dataset}.json"
        if os.path.exists(dataset_file):
            with open(dataset_file, mode="r", encoding="utf-8") as file:
                self.gorilla_data = json.load(file)
        else:
            function_url = f"{base_url}/{dataset}.json"
            answer_url = f"{base_url}/possible_answer/{dataset}.json"
            print(f"Downloading {dataset}.json from GitHub...")
            functions_data = []
            answers_data = []
            try:
                function_response = requests.get(function_url)
                function_response.raise_for_status()
                function_text = function_response.text
                for line in function_text.strip().split("\n"):
                    if line.strip():
                        try:
                            functions_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing function line in {dataset}.json: {e}")
                answer_response = requests.get(answer_url)
                answer_response.raise_for_status()
                answer_text = answer_response.text
                for line in answer_text.strip().split("\n"):
                    if line.strip():
                        try:
                            answers_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing answer line in {dataset}.json: {e}")
                print(
                    f"Successfully downloaded {dataset}.json: {len(functions_data)} functions, {len(answers_data)} answers"
                )
            except requests.RequestException as e:
                print(f"Error downloading {dataset}.json: {e}")
                functions_data = []
                answers_data = []
            if not functions_data or not answers_data:
                print(f"Skipping {dataset}.json - failed to download data")
                return
            print(f"Processing {dataset}.json...")
            answers_by_id = {item["id"]: item for item in answers_data}
            for item in functions_data:
                item_id = item["id"]
                question = item["question"][0]
                if item_id not in answers_by_id:
                    print(f"Warning: No answer found for item {item_id}")
                    continue
                if "function" not in item or not item["function"]:
                    print(f"Warning: No function definition for item {item_id}")
                    continue
                tool = [{"type": "function", "function": func} for func in item["function"]]
                self.map_type_values(tool)
                answer = answers_by_id[item_id]
                if "ground_truth" not in answer or not answer["ground_truth"]:
                    print(f"Warning: No ground truth for item {item_id}")
                    continue
                ideal_call = []
                for ground_truth in answer["ground_truth"]:
                    function_name = list(ground_truth.keys())[0]
                    params = ground_truth[function_name]
                    ideal_call.append({"name": function_name, "arguments": params})
                self.gorilla_data.append(
                    {
                        "id": id,
                        "question": question,
                        "tool": tool,
                        "ideal_call": ideal_call,
                        "source": f"{dataset}.json",
                    }
                )
                id += 1
            with open(dataset_file, mode="w", encoding="utf-8") as file:
                json.dump(self.gorilla_data, file, ensure_ascii=False, indent=4)
        if self.tokenizer is not None:
            for item in self.gorilla_data:
                num_tokens = 0
                for message in item["question"]:
                    num_tokens += len(
                        tokenizer.encode(message["content"], add_special_tokens=False)
                    )
                item["num_tokens"] = num_tokens
        if not use_stag:
            for item in self.gorilla_data:
                for tool in item["tool"]:
                    tool["function"]["strict"] = False

    def generate_request_records(
            self,
            input_len: Optional[int],
            output_len: Optional[int],
            input_len_std: float = 0.0,
            output_len_std: float = 0.0,
    ) -> List[RequestRecord]:

        request_records = []
        for entry in self.gorilla_data:
            # If the request does not have enough length, discard it.
            # if input_len is not None and entry["num_tokens"] < input_len + 4 * input_len_std:
            #     continue

            if output_len is not None:
                output_length = max(
                    round(np.random.normal(loc=output_len, scale=output_len_std)), 1
                )
            else:
                output_length = 256
            request_records.append(
                RequestRecord(
                    request_id=entry["id"],
                    chat_cmpl=ChatCompletionRequest(
                        messages=[
                            ChatCompletionMessage(content=message["content"], role=message["role"])
                            for message in entry["question"]
                        ],
                        model="",
                        max_tokens=output_length,
                        tools=entry["tool"],
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=entry["num_tokens"],
                    ),
                )
            )
        return request_records

    # Modified by https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/eval_checker/ast_eval/ast_checker.py
    def check_simple(self, tool_call: Dict[str, Any],
                     tool: Dict[str, Any], ideal: Dict[str, Any]) -> Tuple[bool, bool, Error]:
        # check func name
        if ideal["name"] != tool_call["function"]["name"]:
            return True, False, Error("wrong function name.", Err_type.FUNC_NAME_ERROR)
        func = tool["function"]
        # check func args
        for arg in func["parameters"]["required"]:
            if arg not in tool_call["function"]["arguments"]:
                return True, False, Error(f"missing arg: {arg}", Err_type.PARA_KEY_ERROR)
        for arg in tool_call["function"]["arguments"].keys():
            ideal_arg: List = ideal["arguments"][arg] if arg in ideal["arguments"] else None
            real_arg = tool_call["function"]["arguments"][arg]
            if arg not in func["parameters"]["properties"]:
                return True, False, Error(f"unknown arg: {arg}", Err_type.PARA_KEY_ERROR)
            info_arg = func["parameters"]["properties"][arg]
            if info_arg["type"] == "integer":
                acc, err = self.check_integer(real_arg, ideal_arg)
                if not acc:
                    return True, False, err
            elif info_arg["type"] == "number":
                acc, err = self.check_number(real_arg, ideal_arg)
                if not acc:
                    return True, False, err
            elif info_arg["type"] == "boolean":
                acc, err = self.check_boolean(real_arg, ideal_arg)
                if not acc:
                    return True, False, err
            elif info_arg["type"] == "string":
                enum = info_arg["enum"] if "enum" in info_arg else None
                acc, err = self.check_string(real_arg, ideal_arg, enum)
                if not acc:
                    return True, False, err
            elif info_arg["type"] == "array":
                acc, err = self.check_list(real_arg, ideal_arg, info_arg["items"])
                if not acc:
                    return True, False, err
            elif info_arg["type"] == "dict":
                acc, err = self.check_dict(real_arg, ideal_arg, info_arg["properties"])
                if not acc:
                    return True, False, err
        return True, True, Error()

    def check_integer(self, real_arg: Any, ideal_arg: Optional[List[Any]]) -> Tuple[bool, Error]:
        if type(real_arg) != int:
            return False, Error(f"wrong type {real_arg}: not int", Err_type.TYPE_ERROR)
        if ideal_arg is None:
            return True, Error()
        match = False
        err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
        for ideal in ideal_arg:
            if real_arg == ideal:
                match = True
                err = Error()
                break
        return match, err

    def check_number(self, real_arg: Any, ideal_arg: Optional[List[Any]]) -> Tuple[bool, Error]:
        if type(real_arg) != float and type(real_arg) != int:
            return False, Error(f"wrong type {real_arg}: not number", Err_type.TYPE_ERROR)
        if ideal_arg is None:
            return True, Error()
        match = False
        err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
        for ideal in ideal_arg:
            if real_arg == ideal:
                match = True
                err = Error()
                break
        return match, err

    def check_string(self, real_arg: Any, ideal_arg: Optional[List[Any]], enum: Optional[List[str]]) -> Tuple[bool, Error]:

        def standardize_string(string: Any) -> str:
            if not isinstance(string, str):
                return "-----Error------"
            regex_string = r"[ \,\.\/\-\_\*\^]"
            return re.sub(regex_string, "", string).lower().replace("'", '"')

        if type(real_arg) != str:
            return False, Error(f"wrong type {real_arg}: not string", Err_type.TYPE_ERROR)
        match = False
        err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
        real_arg = standardize_string(real_arg)
        if ideal_arg is None:
            if enum is None:
                return True, Error()
            else:
                err.error_type = Err_type.ENUM_ERROR
                for ideal in enum:
                    if real_arg == standardize_string(ideal):
                        match = True
                        err = Error()
                        break
        else:
            for ideal in ideal_arg:
                if real_arg == standardize_string(ideal):
                    match = True
                    err = Error()
                    break
        return match, err

    def check_boolean(self, real_arg: bool, ideal_arg: Optional[List[bool]]) -> Tuple[bool, Error]:
        if type(real_arg) != bool:
            return False, Error(f"wrong type {real_arg}: not bool", Err_type.TYPE_ERROR)
        if ideal_arg is None:
            return True, Error()
        match = False
        err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
        for ideal in ideal_arg:
            if real_arg == ideal:
                match = True
                err = Error()
                break
        return match, err

    def check_list(self, real_arg: List, ideal_arg: Optional[List[List]], item: Dict[str, Any]) -> Tuple[bool, Error]:
        if type(real_arg) != list:
            return False, Error(f"wrong type of {real_arg}: not list.", Err_type.TYPE_ERROR)
        item_type = item["type"]
        if ideal_arg is None:
            if item_type == "integer":
                for i, integer in enumerate(real_arg):
                    acc, err = self.check_integer(integer, None)
                    if not acc:
                        return False, err
            elif item_type == "number":
                for i, integer in enumerate(real_arg):
                    acc, err = self.check_number(integer, None)
                    if not acc:
                        return False, err
            elif item_type == "boolean":
                for i, boolean in enumerate(real_arg):
                    acc, err = self.check_boolean(boolean, None)
                    if not acc:
                        return False, err
            elif item_type == "string":
                for i, string in enumerate(real_arg):
                    enum = item["enum"] if "enum" in item else None
                    acc, err = self.check_string(string, None, enum)
                    if not acc:
                        return False, err
            elif item_type == "array":
                for i, array in enumerate(real_arg):
                    acc, err = self.check_list(array, None, item["items"])
                    if not acc:
                        return False, err
            elif item_type == "dict":
                for i, dictionary in enumerate(real_arg):
                    acc, err = self.check_dict(dictionary, None, item["properties"])
                    if not acc:
                        return False, err
            return True, Error()
        else:
            final_err = ""
            err_type = Err_type.NONE
            for j, ideal in enumerate(ideal_arg):
                if len(ideal) != len(real_arg):
                    final_err += f"[ideal {j}] wrong length of {real_arg}."
                    err_type = min(err_type, Err_type.PARA_VALUE_ERROR)
                    continue
                match = True
                if item_type == "integer":
                    for i, integer in enumerate(real_arg):
                        acc, err = self.check_integer(integer, [ideal[i]])
                        if not acc:
                            match = False
                            final_err += f"[ideal {j}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                elif item_type == "number":
                    for i, integer in enumerate(real_arg):
                        acc, err = self.check_number(integer, [ideal[i]])
                        if not acc:
                            match = False
                            final_err += f"[ideal {j}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                elif item_type == "boolean":
                    for i, boolean in enumerate(real_arg):
                        acc, err = self.check_boolean(boolean, [ideal[i]])
                        if not acc:
                            match = False
                            final_err += f"[ideal {j}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                elif item_type == "string":
                    for i, string in enumerate(real_arg):
                        enum = item["enum"] if "enum" in item else None
                        acc, err = self.check_string(string, [ideal[i]], enum)
                        if not acc:
                            match = False
                            final_err += f"[ideal {j}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                elif item_type == "array":
                    for i, array in enumerate(real_arg):
                        acc, err = self.check_list(array, [ideal[i]], item["items"])
                        if not acc:
                            match = False
                            final_err += f"[ideal {j}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                elif item_type == "dict":
                    for i, dictionary in enumerate(real_arg):
                        acc, err = self.check_dict(dictionary, [ideal[i]], item["properties"])
                        if not acc:
                            match = False
                            final_err += f"[ideal {j}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                if match:
                    return True, Error()
            return False, Error(final_err, err_type)

    def check_dict(self, real_arg: Dict[str, Any], ideal_arg: Optional[Dict[str, Any]], properties: Dict[str, Any]) -> Tuple[bool, Error]:
        if type(real_arg) != dict:
            return False, Error(f"wrong type of {real_arg}: not dict.", Err_type.TYPE_ERROR)
        if ideal_arg is None:
            for key in properties.keys():
                if key not in real_arg:
                    return False, Error(f"missing key: {key}.", Err_type.PARA_KEY_ERROR)
                item_type = properties[key]["type"]
                if item_type == "integer":
                    acc, err = self.check_integer(real_arg[key], None)
                    if not acc:
                        return False, err
                elif item_type == "number":
                    acc, err = self.check_number(real_arg[key], None)
                    if not acc:
                        return False, err
                elif item_type == "boolean":
                    acc, err = self.check_boolean(real_arg[key], None)
                    if not acc:
                        return False, err
                elif item_type == "string":
                    enum = properties[key]["enum"] if "enum" in properties[key] else None
                    acc, err = self.check_string(real_arg[key], None, enum)
                    if not acc:
                        return False, err
                elif item_type == "array":
                    acc, err = self.check_list(real_arg[key], None, properties[key]["items"])
                    if not acc:
                        return False, err
                elif item_type == "dict":
                    acc, err = self.check_dict(real_arg[key], None, properties[key]["properties"])
                    if not acc:
                        return False, err
            return True, Error()
        else:
            final_err = ""
            err_type = Err_type.NONE
            for i, ideal in enumerate(ideal_arg):
                match = True
                for key in properties.keys():
                    if key not in real_arg:
                        match = False
                        final_err += f"[ideal {i}] missing key: {key}."
                        err_type = min(err_type, Err_type.PARA_KEY_ERROR)
                        break
                    item_type = properties[key]["type"]
                    if item_type == "integer":
                        acc, err = self.check_integer(real_arg[key], [ideal[key]])
                        if not acc:
                            match = False
                            final_err += f"[ideal {i}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                    elif item_type == "number":
                        acc, err = self.check_number(real_arg[key], [ideal[key]])
                        if not acc:
                            match = False
                            final_err += f"[ideal {i}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                    elif item_type == "boolean":
                        acc, err = self.check_boolean(real_arg[key], [ideal[key]])
                        if not acc:
                            match = False
                            final_err += f"[ideal {i}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                    elif item_type == "string":
                        enum = properties[key]["enum"] if "enum" in properties[key] else None
                        acc, err = self.check_string(real_arg[key], [ideal[key]], enum)
                        if not acc:
                            match = False
                            final_err += f"[ideal {i}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                    elif item_type == "array":
                        acc, err = self.check_list(real_arg[key], [ideal[key]], properties[key]["items"])
                        if not acc:
                            match = False
                            final_err += f"[ideal {i}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                    elif item_type == "dict":
                        acc, err = self.check_dict(real_arg[key], [ideal[key]], properties[key]["properties"])
                        if not acc:
                            match = False
                            final_err += f"[ideal {i}] {err}"
                            err_type = min(err_type, err.error_type)
                            break
                if match:
                    return True, Error()
            return False, Error(final_err, err_type)

    def map_type_values(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self.map_type_values(value)
                elif key == "type" and value in GORILLA_TO_OPENAPI:
                    data[key] = GORILLA_TO_OPENAPI[value]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self.map_type_values(item)


SUPPORTED_DATASET = [
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
]


def create_dataset(  # pylint: disable=too-many-return-statements,too-many-branches
        args: argparse.Namespace, tokenizer: AutoTokenizer
) -> Dataset:
    """Create a dataset instance with regard to the specified dataset kind and file path."""
    if args.dataset_path is not None and not isinstance(args.dataset_path, str):
        raise TypeError(f"Invalid dataset path {args.dataset_path}. Please use a string.")
    if args.dataset in SUPPORTED_DATASET:
        if args.dataset_path is None:
            raise ValueError(
                "Gorilla dataset requires dataset path. "
                'Please specify it with "--dataset-path".'
            )
        assert (
                args.apply_chat_template is False
        ), "Gorilla dataset does not support applying chat template"
        return GorillaDataset(args.dataset, args.dataset_path, tokenizer, args.use_stag)
    raise ValueError(f"Unrecognized dataset {args.dataset}")
