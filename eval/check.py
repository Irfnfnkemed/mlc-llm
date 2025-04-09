"""MLC LLM benchmark main entrance"""

import argparse
import json
import re
from typing import Dict, Any, Tuple, List, Optional

SUPPORTED_DATASET = [
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
    "BFCL_v3_live_simple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_live_parallel",
    "ALL",
]

SUPPORTED_MODEL = [
    "Llama-3.2-1B-Instruct-q0f32-MLC",
    "Llama-3.2-3B-Instruct-q0f32-MLC",
    "Llama-3-8B-Instruct-q0f16-MLC",
    "Qwen2.5-0.5B-Instruct-q0f32-MLC",
    "Qwen2.5-3B-Instruct-q0f16-MLC",
    "Hermes-3-Llama-3.2-3B-q0f16-MLC",
    "ALL",
]

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


# Modified by https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/eval_checker/ast_eval/ast_checker.py
def check_simple(
        gorilla, tool_call: Dict[str, Any], tool: Dict[str, Any], ideal: Dict[str, Any]
) -> Tuple[bool, bool, Error]:
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
            acc, err = check_integer(gorilla, real_arg, ideal_arg)
            if not acc:
                return True, False, err
        elif info_arg["type"] == "number":
            acc, err = check_number(gorilla, real_arg, ideal_arg)
            if not acc:
                return True, False, err
        elif info_arg["type"] == "boolean":
            acc, err = check_boolean(gorilla, real_arg, ideal_arg)
            if not acc:
                return True, False, err
        elif info_arg["type"] == "string":
            enum = info_arg["enum"] if "enum" in info_arg else None
            acc, err = check_string(gorilla, real_arg, ideal_arg, enum)
            if not acc:
                return True, False, err
        elif info_arg["type"] == "array":
            acc, err = check_list(gorilla, real_arg, ideal_arg, info_arg["items"])
            if not acc:
                return True, False, err
        elif info_arg["type"] == "dict":
            acc, err = check_dict(real_arg, ideal_arg, info_arg["properties"])
            if not acc:
                return True, False, err
    return True, True, Error()


def check_integer(gorilla, real_arg: Any, ideal_arg: Optional[List[Any]]) -> Tuple[bool, Error]:
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


def check_number(gorilla, real_arg: Any, ideal_arg: Optional[List[Any]]) -> Tuple[bool, Error]:
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


def check_string(
        gorilla, real_arg: Any, ideal_arg: Optional[List[Any]], enum: Optional[List[str]]
) -> Tuple[bool, Error]:
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


def check_boolean(gorilla, real_arg: bool, ideal_arg: Optional[List[bool]]) -> Tuple[bool, Error]:
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


def check_list(
        gorilla, real_arg: List, ideal_arg: Optional[List[List]], item: Dict[str, Any]
) -> Tuple[bool, Error]:
    if type(real_arg) != list:
        return False, Error(f"wrong type of {real_arg}: not list.", Err_type.TYPE_ERROR)
    item_type = item["type"]
    if ideal_arg is None:
        if item_type == "integer":
            for i, integer in enumerate(real_arg):
                acc, err = check_integer(gorilla, integer, None)
                if not acc:
                    return False, err
        elif item_type == "number":
            for i, integer in enumerate(real_arg):
                acc, err = check_number(gorilla, integer, None)
                if not acc:
                    return False, err
        elif item_type == "boolean":
            for i, boolean in enumerate(real_arg):
                acc, err = check_boolean(gorilla, boolean, None)
                if not acc:
                    return False, err
        elif item_type == "string":
            for i, string in enumerate(real_arg):
                enum = item["enum"] if "enum" in item else None
                acc, err = check_string(gorilla, string, None, enum)
                if not acc:
                    return False, err
        elif item_type == "array":
            for i, array in enumerate(real_arg):
                acc, err = check_list(gorilla, array, None, item["items"])
                if not acc:
                    return False, err
        elif item_type == "dict":
            for i, dictionary in enumerate(real_arg):
                acc, err = check_dict(dictionary, None, item["properties"])
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
                    acc, err = check_integer(gorilla, integer, [ideal[i]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "number":
                for i, integer in enumerate(real_arg):
                    acc, err = check_number(gorilla, integer, [ideal[i]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "boolean":
                for i, boolean in enumerate(real_arg):
                    acc, err = check_boolean(gorilla, boolean, [ideal[i]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "string":
                for i, string in enumerate(real_arg):
                    enum = item["enum"] if "enum" in item else None
                    acc, err = check_string(gorilla, string, [ideal[i]], enum)
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "array":
                for i, array in enumerate(real_arg):
                    acc, err = check_list(gorilla, array, [ideal[i]], item["items"])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "dict":
                for i, dictionary in enumerate(real_arg):
                    acc, err = check_dict(dictionary, [ideal[i]], item["properties"])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            if match:
                return True, Error()
        return False, Error(final_err, err_type)


def check_dict(
        gorilla,
        real_arg: Dict[str, Any],
        ideal_arg: Optional[Dict[str, Any]],
        properties: Dict[str, Any],
) -> Tuple[bool, Error]:
    if type(real_arg) != dict:
        return False, Error(f"wrong type of {real_arg}: not dict.", Err_type.TYPE_ERROR)
    if ideal_arg is None:
        for key in properties.keys():
            if key not in real_arg:
                return False, Error(f"missing key: {key}.", Err_type.PARA_KEY_ERROR)
            item_type = properties[key]["type"]
            if item_type == "integer":
                acc, err = check_integer(gorilla, real_arg[key], None)
                if not acc:
                    return False, err
            elif item_type == "number":
                acc, err = check_number(gorilla, real_arg[key], None)
                if not acc:
                    return False, err
            elif item_type == "boolean":
                acc, err = check_boolean(gorilla, real_arg[key], None)
                if not acc:
                    return False, err
            elif item_type == "string":
                enum = properties[key]["enum"] if "enum" in properties[key] else None
                acc, err = check_string(gorilla, real_arg[key], None, enum)
                if not acc:
                    return False, err
            elif item_type == "array":
                acc, err = check_list(gorilla, real_arg[key], None, properties[key]["items"])
                if not acc:
                    return False, err
            elif item_type == "dict":
                acc, err = check_dict(real_arg[key], None, properties[key]["properties"])
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
                    acc, err = check_integer(gorilla, real_arg[key], [ideal[key]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "number":
                    acc, err = check_number(gorilla, real_arg[key], [ideal[key]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "boolean":
                    acc, err = check_boolean(gorilla, real_arg[key], [ideal[key]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "string":
                    enum = properties[key]["enum"] if "enum" in properties[key] else None
                    acc, err = check_string(gorilla, real_arg[key], [ideal[key]], enum)
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "array":
                    acc, err = check_list(
                        gorilla, real_arg[key], [ideal[key]], properties[key]["items"]
                    )
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "dict":
                    acc, err = check_dict(
                        real_arg[key], [ideal[key]], properties[key]["properties"]
                    )
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            if match:
                return True, Error()
        return False, Error(final_err, err_type)


def check_acc(dataset: str, gorilla: Dict, output_dir: str):
    """Check the accuracy of the generated requests."""
    request_records = []
    final_output = {"fail_format": [], "fail_call": [], "fail_reason": {}}
    with open(f"{output_dir}/result.json", "r") as f:
        request_records = json.load(f)
    count = 0
    err_types = [0] * (len(Err_type) - 1)
    if dataset == "BFCL_v3_simple" or dataset == "BFCL_v3_live_simple":
        for request in request_records:
            info = gorilla[request["id"]]
            count += 1
            if "call" not in request:
                final_output["fail_format"].append(request["id"])
                final_output["fail_call"].append(request["id"])
                final_output["fail_reason"][request["id"]] = {
                    "type": "FORMAT_ERROR",
                    "message": "wrong format.",
                }
                err_types[Err_type.FORMAT_ERROR] += 1
                continue
            if len(request["call"]) != 1:
                format, call, err = (
                    True,
                    False,
                    Error("wrong calling numbers.", Err_type.CALL_NUMBER_ERROR),
                )
            else:
                format, call, err = check_simple(
                    gorilla, request["call"][0], info["tool"][0], info["ideal_call"][0]
                )
            if not format:
                final_output["fail_format"].append(request["id"])
            if not call:
                final_output["fail_call"].append(request["id"])
            if err.error_type != Err_type.NONE:
                final_output["fail_reason"][request["id"]] = {
                    "type": Err_type(err.error_type).name,
                    "message": err.message,
                }
                err_types[err.error_type] += 1
    elif dataset == "BFCL_v3_multiple" or dataset == "BFCL_v3_live_multiple":
        for request in request_records:
            info = gorilla[request["id"]]
            count += 1
            if "call" not in request:
                final_output["fail_format"].append(request["id"])
                final_output["fail_call"].append(request["id"])
                final_output["fail_reason"][request["id"]] = {
                    "type": "FORMAT_ERROR",
                    "message": "wrong format.",
                }
                err_types[Err_type.FORMAT_ERROR] += 1
                continue
            if len(request["call"]) != 1:
                format, call, err = (
                    True,
                    False,
                    Error("wrong calling numbers.", Err_type.CALL_NUMBER_ERROR),
                )
            else:
                expected_tool = None
                for tool in info["tool"]:
                    if tool["function"]["name"] == info["ideal_call"][0]["name"]:
                        expected_tool = tool
                        break
                format, call, err = check_simple(
                    gorilla, request["call"][0], expected_tool, info["ideal_call"][0]
                )
            if not format:
                final_output["fail_format"].append(request["id"])
            if not call:
                final_output["fail_call"].append(request["id"])
            if err.error_type != Err_type.NONE:
                final_output["fail_reason"][request["id"]] = {
                    "type": Err_type(err.error_type).name,
                    "message": err.message,
                }
                err_types[err.error_type] += 1
    elif dataset == "BFCL_v3_parallel" or dataset == "BFCL_v3_live_parallel":
        for request in request_records:
            info = gorilla[request["id"]]
            count += 1
            if "call" not in request:
                final_output["fail_format"].append(request["id"])
                final_output["fail_call"].append(request["id"])
                final_output["fail_reason"][request["id"]] = {
                    "type": "FORMAT_ERROR",
                    "message": "wrong format.",
                }
                err_types[Err_type.FORMAT_ERROR] += 1
                continue
            if len(request["call"]) != len(info["ideal_call"][0]):
                format, call, err = (
                    True,
                    False,
                    Error("wrong calling numbers.", Err_type.CALL_NUMBER_ERROR),
                )
            else:
                for ideal in info["ideal_call"]:
                    expected_tool = None
                    for tool in info["tool"]:
                        if tool["function"]["name"] == ideal["name"]:
                            expected_tool = tool
                            break
                    expected_request = None
                    for single_request in request["call"]:
                        if single_request["function"]["name"] == ideal["name"]:
                            expected_request = single_request
                            break
                    if expected_request is None:
                        format, call, err = (
                            True,
                            False,
                            Error("not calling expected function", Err_type.FUNC_NAME_ERROR),
                        )
                        break
                    else:
                        format, call, err = check_simple(
                            gorilla, expected_request, expected_tool, ideal
                        )
                        if not format or not call:
                            break
            if not format:
                final_output["fail_format"].append(request["id"])
            if not call:
                final_output["fail_call"].append(request["id"])
            if err.error_type != Err_type.NONE:
                final_output["fail_reason"][request["id"]] = {
                    "type": Err_type(err.error_type).name,
                    "message": err.message,
                }
                err_types[err.error_type] += 1

    correct_format = count - len(final_output["fail_format"])
    correct_call = count - len(final_output["fail_call"])
    final_output["FORMAT_ACCURACY"] = correct_format / count
    final_output["CALL_ACCURACY"] = correct_call / count
    for i in range(len(Err_type) - 1):
        final_output[Err_type(i).name] = err_types[i] / count
    print(f"correct_format: {correct_format}/{count}, correct_call: {correct_call}/{count}")
    with open(f"{output_dir}/final.json", "w", encoding="utf-8") as file:
        json.dump(final_output, file, indent=4)


def main(args: argparse.Namespace):
    """Main benchmark entrance."""
    models = []
    datasets = []
    if args.dataset == "ALL":
        datasets = SUPPORTED_DATASET
        datasets.pop(-1)
    else:
        datasets.append(args.dataset)
    if args.model == "ALL":
        models = SUPPORTED_MODEL
        models.pop(-1)
    else:
        models.append(args.model)
    print(models, datasets)
    for model in models:
        for dataset in datasets:
            output_dir = f"{args.output_root}/{model}/{dataset}/{'use_stag' if args.use_stag else 'no_stag'}/"
            dataset_file = f"{args.output_root}/dataset/{dataset}.json"
            with open(dataset_file, mode="r", encoding="utf-8") as file:
                gorilla = json.load(file)
            check_acc(dataset, gorilla, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLC LLM benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASET,
        help=f"The benchmark dataset kind. Supporting {SUPPORTED_DATASET}",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODEL,
        help=f"The benchmark model kind. Supporting {SUPPORTED_MODEL}",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The dataset file path.",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=str,
        required=True,
        help="The root of the output file.",
    )
    parser.add_argument(
        "--use-stag",
        action="store_true",
        help="Whether to set stag.",
    )
    args = parser.parse_args()
    main(args)
