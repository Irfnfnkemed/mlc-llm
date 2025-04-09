import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

models = ['Llama-3.2-1B-Instruct-q0f32-MLC', 'Llama-3.2-3B-Instruct-q0f32-MLC', 'Llama-3-8B-Instruct-q0f16-MLC',
          'Qwen2.5-0.5B-Instruct-q0f32-MLC', 'Qwen2.5-3B-Instruct-q0f16-MLC', 'Hermes-3-Llama-3.2-3B-q0f16-MLC']
datasets = ["BFCL_v3_simple", "BFCL_v3_multiple", "BFCL_v3_parallel", "BFCL_v3_live_simple", "BFCL_v3_live_multiple", "BFCL_v3_live_parallel"]


def summary(args: argparse.ArgumentParser) -> Dict:
    summary = {}
    file_path = args.data_root
    for model in models:
        summary[model] = {}
        for dataset in datasets:
            with open(f"{file_path}/{model}/{dataset}/use_stag/final.json", 'r', encoding='utf-8') as file:
                stag = json.load(file)
            with open(f"{file_path}/{model}/{dataset}/no_stag/final.json", 'r', encoding='utf-8') as file:
                no_stag = json.load(file)
            with open(f"{file_path}/{model}/{dataset}/use_stag/result.json", 'r', encoding='utf-8') as file:
                stag_result = json.load(file)
            with open(f"{file_path}/{model}/{dataset}/no_stag/result.json", 'r', encoding='utf-8') as file:
                no_stag_result = json.load(file)
            with open(f"{file_path}/dataset/{dataset}.json", mode="r", encoding="utf-8") as file:
                gorilla_data = json.load(file)
            summary[model][dataset] = {"use_stag": {}, "no_stag": {}}
            summary[model][dataset]["correct_schema/output_trigger"] = {}
            summary[model][dataset]["use_stag"]["format_error"] = stag["FORMAT_ERROR"]
            summary[model][dataset]["use_stag"]["call_num_error"] = stag["CALL_NUMBER_ERROR"]
            summary[model][dataset]["use_stag"]["func_name_error"] = stag["FUNC_NAME_ERROR"]
            summary[model][dataset]["use_stag"]["para_key_error"] = stag["PARA_KEY_ERROR"]
            summary[model][dataset]["use_stag"]["type_error"] = stag["TYPE_ERROR"]
            summary[model][dataset]["use_stag"]["enum_error"] = stag["ENUM_ERROR"]
            summary[model][dataset]["use_stag"]["para_value_error"] = stag["PARA_VALUE_ERROR"]
            summary[model][dataset]["use_stag"]["correct_call"] = stag["CALL_ACCURACY"]
            output_trigger = 0
            correct_schema = 0
            ideal_func_name = {}
            for entry in gorilla_data:
                ideal_func_name[str(entry["id"])] = set()
                for tool in entry["tool"]:
                    ideal_func_name[str(entry["id"])].add(tool["function"]["name"])
            for entry in stag_result:
                if "output" in entry and "{\"name\":" in entry["output"]:
                    output_trigger += 1
                    t = correct_schema
                    if str(entry["id"]) in stag["fail_reason"]:
                        if "type" in stag["fail_reason"][str(entry["id"])]:
                            err_type = stag["fail_reason"][str(entry["id"])]["type"]
                            if err_type == "PARA_VALUE_ERROR":
                                correct_schema += 1
                            elif err_type == "FUNC_NAME_ERROR" or err_type == "CALL_NUMBER_ERROR":
                                flag = True
                                for call in entry["call"]:
                                    if call["function"]["name"] not in ideal_func_name[str(entry["id"])]:
                                        flag = False
                                if flag:
                                    correct_schema += 1
                    else:
                        correct_schema += 1
                    if t == correct_schema:
                        print(model, dataset, entry["id"], flush=True)
            summary[model][dataset]["correct_schema/output_trigger"]["use_stag"] = correct_schema / output_trigger

            summary[model][dataset]["no_stag"]["format_error"] = no_stag["FORMAT_ERROR"]
            summary[model][dataset]["no_stag"]["call_num_error"] = no_stag["CALL_NUMBER_ERROR"]
            summary[model][dataset]["no_stag"]["func_name_error"] = no_stag["FUNC_NAME_ERROR"]
            summary[model][dataset]["no_stag"]["para_key_error"] = no_stag["PARA_KEY_ERROR"]
            summary[model][dataset]["no_stag"]["type_error"] = no_stag["TYPE_ERROR"]
            summary[model][dataset]["no_stag"]["enum_error"] = no_stag["ENUM_ERROR"]
            summary[model][dataset]["no_stag"]["para_value_error"] = no_stag["PARA_VALUE_ERROR"]
            summary[model][dataset]["no_stag"]["correct_call"] = no_stag["CALL_ACCURACY"]
            output_trigger = 0
            correct_schema = 0
            ideal_func_name = {}
            for entry in gorilla_data:
                ideal_func_name[str(entry["id"])] = set()
                for tool in entry["tool"]:
                    ideal_func_name[str(entry["id"])].add(tool["function"]["name"])

            for entry in no_stag_result:

                if "output" in entry and "{\"name\":" in entry["output"]:
                    output_trigger += 1
                    if str(entry["id"]) in no_stag["fail_reason"]:
                        if "type" in no_stag["fail_reason"][str(entry["id"])]:
                            err_type = no_stag["fail_reason"][str(entry["id"])]["type"]
                            if err_type == "PARA_VALUE_ERROR":
                                correct_schema += 1
                            elif err_type == "FUNC_NAME_ERROR" or err_type == "CALL_NUMBER_ERROR":
                                flag = True
                                for call in entry["call"]:
                                    if call["function"]["name"] not in ideal_func_name[str(entry["id"])]:
                                        flag = False
                                if flag:
                                    correct_schema += 1
                    else:
                        correct_schema += 1

            summary[model][dataset]["correct_schema/output_trigger"]["no_stag"] = correct_schema / output_trigger

    with open(f"{file_path}/summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    return summary


def draw(args: argparse.ArgumentParser, dataset: str, summary: Dict):
    if dataset not in datasets:
        return
    bars = ['use_stag', 'no_stag']
    draw_info = {}
    for model in summary:
        draw_info[model] = summary[model][dataset]
    subcategories = list(next(iter(draw_info.values()))['use_stag'].keys())
    x = np.arange(len(models))
    width = 0.35
    gap = 0.05
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i, bar in enumerate(bars):
        bottom = np.zeros(len(models))
        for j, subcategory in enumerate(subcategories):
            values = [draw_info[model][bar][subcategory] for model in models]
            ax.bar(x + i * (width + gap), values, width, bottom=bottom, color=colors[j], label=subcategory if i == 0 else "")
            bottom += values
        for idx, value in enumerate(bottom):
            text_height = value + max(bottom) * 0.02
            ax.text(x[idx] + i * (width + gap), text_height, bar, ha='center', va='center', fontsize=8, color='black')
    ax.set_title(f'Test on BFCL_v3_{dataset}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=8, ha='center', fontsize=9)
    ax.legend(ncol=1, fontsize=9, bbox_to_anchor=(0.8, 0.9), loc='upper left')
    plt.subplots_adjust(bottom=0.2, top=0.85, right=0.85)
    plt.savefig(f'{args.data_root}/{dataset}_e2e.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, bar in enumerate(bars):
        values = [draw_info[model]['correct_schema/output_trigger'][bar] for model in models]
        ax.bar(x + i * (width + gap), values, width, color=colors[i], label=bar)
    ax.set_title(f'Correct_schema/Output_trigger on BFCL_v3_{dataset}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=8, ha='center', fontsize=9)
    ax.legend(ncol=1, fontsize=9, bbox_to_anchor=(0.8, 0.9), loc='upper left')
    plt.subplots_adjust(bottom=0.2, top=0.85, right=0.85)
    plt.savefig(f'{args.data_root}/{dataset}_rate.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Draw")
    parser.add_argument(
        "--data-root",
        type=str,
        help="The root path of the data. The data should be placed in [data-root]/[model]/[dataset]/[stag]/final.json",
    )
    args = parser.parse_args()
    data = summary(args)
    for dataset in datasets:
        draw(args, dataset, data)
