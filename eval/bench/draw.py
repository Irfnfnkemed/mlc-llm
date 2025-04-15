import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

models = ['Llama-3.2-1B-Instruct-q0f32-MLC', 'Llama-3.2-3B-Instruct-q0f32-MLC', 'Llama-3-8B-Instruct-q0f16-MLC',
          'Qwen2.5-0.5B-Instruct-q0f32-MLC', 'Qwen2.5-3B-Instruct-q0f16-MLC', 'Hermes-3-Llama-3.2-3B-q0f16-MLC']
datasets = ["BFCL_v3_simple", "BFCL_v3_multiple", "BFCL_v3_parallel", "BFCL_v3_live_simple", "BFCL_v3_live_multiple", "BFCL_v3_live_parallel"]



def draw(args: argparse.ArgumentParser, summary: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    bars = ['use_stag', 'no_stag']

    for ax_idx, ax in enumerate(axes.flat):
        dataset = datasets[ax_idx]
        draw_info = {}
        for model in summary:
            draw_info[model] = summary[model][dataset]
        subcategories = list(next(iter(draw_info.values()))['use_stag'].keys())
        x = np.arange(len(models))
        width = 0.35
        gap = 0.05
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for i, bar in enumerate(bars):
            bottom = np.zeros(len(models))
            for j, subcategory in enumerate(subcategories):
                values = [draw_info[model][bar][subcategory] for model in models]
                ax.bar(x + i * (width + gap), values, width, bottom=bottom, color=colors[j], label=subcategory if i == 0 else "")
                bottom += values
            for idx, value in enumerate(bottom):
                text_height = value + max(bottom) * 0.02
                ax.text(x[idx] + i * (width + gap), text_height, ["w/", "w/o"][i], ha='center', va='center', fontsize=8, color='black')
        ax.set_title(f'Test on {dataset}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=15)
        ax.set_ylabel('Proportion', fontsize=15)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([model.rstrip("-MLC") for model in models], rotation=12, ha='center', fontsize=9)
        if ax_idx == 0:
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(subcategories))]

    fig.legend(handles=legend_handles, labels=subcategories,
               loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=len(subcategories), fontsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f'{args.summary_root}/accuracy.png', dpi=300, bbox_inches='tight')
  #  plt.show()
  
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    bars = ['use_stag', 'no_stag']

    for ax_idx, ax in enumerate(axes.flat):
        dataset = datasets[ax_idx]
        draw_info = {}
        for model in summary:
            draw_info[model] = summary[model][dataset]
        for i, bar in enumerate(bars):
            values = [draw_info[model]["correct_schema_rate"][bar] for model in models]
            ax.bar(x + i * (width + gap), values, width, color=colors[i], label=bar)
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Models', fontsize=15)
        ax.set_ylabel('Correct_schema_rate', fontsize=15)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x + width / 2)
        ax.set_title(f'Test on {dataset}', fontsize=14, fontweight='bold')
        ax.set_xticklabels([model.rstrip("-MLC") for model in models], rotation=12, ha='center', fontsize=9)
        if ax_idx == 0:
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(2)]

    plt.subplots_adjust(bottom=0.2, top=0.85, right=0.85)
    fig.legend(handles=legend_handles, labels=["with structual tag", "without structual tag"],
               loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f'{args.summary_root}/correct_schema_rate.png', dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Draw")
    parser.add_argument(
        "--summary-root",
        type=str,
        help="The summary root path of the result.",
    )
    args = parser.parse_args()
    with open(f"{args.summary_root}/summary.json", mode="r", encoding="utf-8") as file:
        summary = json.load(file)
    draw(args, summary)
