import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

models = ['Llama-3.2-1B-Instruct-q0f32', 'Llama-3.2-3B-Instruct-q0f32', 'Llama-3-8B-Instruct-q4f16',
          'Qwen2.5-0.5B-Instruct-q0f32', 'Qwen2.5-3B-Instruct-q4f32', 'Hermes-3-Llama-3.2-3B-q4f16']
datasets = ["simple", "multiple", "parallel"]


def summary(args: argparse.ArgumentParser) -> Dict:
    summary = {}
    file_path = args.data_root
    for model in models:
        summary[model] = {}
        for dataset in datasets:
            with open(f"{file_path}/{model}/{dataset}/stag/final.json", 'r', encoding='utf-8') as file:
                stag = json.load(file)
            with open(f"{file_path}/{model}/{dataset}/no_stag/final.json", 'r', encoding='utf-8') as file:
                no_stag = json.load(file)
            summary[model][dataset] = {"stag": {}, "no_stag": {}}
            summary[model][dataset]["stag"]["format_error"] = stag["FORMAT_ERROR"]
            summary[model][dataset]["stag"]["call_num_error"] = stag["CALL_NUMBER_ERROR"]
            summary[model][dataset]["stag"]["func_name_error"] = stag["FUNC_NAME_ERROR"]
            summary[model][dataset]["stag"]["para_key_error"] = stag["PARA_KEY_ERROR"]
            summary[model][dataset]["stag"]["type_error"] = stag["TYPE_ERROR"]
            summary[model][dataset]["stag"]["enum_error"] = stag["ENUM_ERROR"]
            summary[model][dataset]["stag"]["para_value_error"] = stag["PARA_VALUE_ERROR"]
            summary[model][dataset]["stag"]["correct_call"] = stag["CALL_ACCURACY"]

            summary[model][dataset]["no_stag"]["format_error"] = no_stag["FORMAT_ERROR"]
            summary[model][dataset]["no_stag"]["call_num_error"] = no_stag["CALL_NUMBER_ERROR"]
            summary[model][dataset]["no_stag"]["func_name_error"] = no_stag["FUNC_NAME_ERROR"]
            summary[model][dataset]["no_stag"]["para_key_error"] = no_stag["PARA_KEY_ERROR"]
            summary[model][dataset]["no_stag"]["type_error"] = no_stag["TYPE_ERROR"]
            summary[model][dataset]["no_stag"]["enum_error"] = no_stag["ENUM_ERROR"]
            summary[model][dataset]["no_stag"]["para_value_error"] = no_stag["PARA_VALUE_ERROR"]
            summary[model][dataset]["no_stag"]["correct_call"] = no_stag["CALL_ACCURACY"]

    with open(f"{file_path}/summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    return summary


def draw(args: argparse.ArgumentParser, dataset: str, summary: Dict):
    if dataset not in datasets:
        return
    bars = ['stag', 'no_stag']
    draw_info = {}
    for model in summary:
        draw_info[model] = summary[model][dataset]
    subcategories = list(next(iter(draw_info.values()))['stag'].keys())
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
    plt.savefig(f'{args.data_root}/{dataset}.png', dpi=300, bbox_inches='tight')
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
