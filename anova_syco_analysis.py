import os
import re
import scipy.stats as stats
from collections import defaultdict

SETTINGS = [
    'debate_setting',
    'ethical-setting',
    'false-presuppositions-setting'
]

PROMPT_FILES = [f'prompt{i}_results.txt' for i in range(5)]

def extract_tof_scores(filepath):
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'ToF =' in line:
                match = re.search(r'ToF\s*=\s*(\d+)', line)
                if match:
                    scores.append(int(match.group(1)))
            else:
                stripped = line.strip()
                if stripped and stripped[-1].isdigit():
                    scores.append(int(stripped[-1]))
    return scores

def main():
    output_lines = []

    for setting in SETTINGS:
        output_lines.append(f"============================================================\n")
        output_lines.append(f"=== SETTING: {setting.upper()} ===\n")
        output_lines.append(f"============================================================\n\n")
        
        base_dir = os.path.join(setting, 'results')
        if not os.path.exists(base_dir):
            continue
            
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        # 1. Cross-Model Comparison (Independent Two-Sample Welch's T-Test / ANOVA)
        output_lines.append("--- CROSS-MODEL COMPARISON (Instruct vs. Uncensored) ---\n")
        # Find common baseline prompt file across models
        for prompt_idx in range(5):
            prompt_scores = {}
            for model in sorted(model_dirs):
                path = os.path.join(base_dir, model, f'prompt{prompt_idx}_results.txt')
                if os.path.exists(path):
                    scores = extract_tof_scores(path)
                    if len(scores) > 0:
                        prompt_scores[model] = scores
            
            if len(prompt_scores) >= 2:
                models_keys = list(prompt_scores.keys())
                s1 = prompt_scores[models_keys[0]]
                s2 = prompt_scores[models_keys[1]]
                stat, pval = stats.ttest_ind(s1, s2, equal_var=False)
                output_lines.append(f"Prompt {prompt_idx} Welch's t-test ({models_keys[0]} vs {models_keys[1]}):\n")
                output_lines.append(f"  t-statistic = {stat:.4f}, p-value = {pval:.4e}\n\n")
            elif len(prompt_scores) == 1:
                model_name = list(prompt_scores.keys())[0]
                mean_s = sum(prompt_scores[model_name])/len(prompt_scores[model_name])
                output_lines.append(f"Prompt {prompt_idx}: Only evaluated on {model_name} (Mean ToF = {mean_s:.2f})\n\n")

        # 2. Within-Model ANOVA across Prompts
        output_lines.append("--- WITHIN-MODEL ANOVA (Effect of System Prompting) ---\n")
        for model in sorted(model_dirs):
            prompt_data = []
            valid_prompts = []
            for i in range(5):
                path = os.path.join(base_dir, model, f'prompt{i}_results.txt')
                if os.path.exists(path):
                    scores = extract_tof_scores(path)
                    if len(scores) > 0:
                        prompt_data.append(scores)
                        valid_prompts.append(i)

            if len(prompt_data) >= 2:
                try:
                    if all(len(p) == len(prompt_data[0]) for p in prompt_data):
                        stat, pval = stats.f_oneway(*prompt_data)
                        output_lines.append(f"Model: {model} across Prompts {valid_prompts}:\n")
                        output_lines.append(f"  One-Way ANOVA F-statistic = {stat:.4f}, p-value = {pval:.4e}\n\n")
                    else:
                        stat, pval = stats.kruskal(*prompt_data)
                        output_lines.append(f"Model: {model} across Prompts {valid_prompts}:\n")
                        output_lines.append(f"  Kruskal-Wallis H-statistic = {stat:.4f}, p-value = {pval:.4e}\n\n")
                except Exception as e:
                    output_lines.append(f"Model: {model} - Statistical test error: {e}\n\n")
            else:
                output_lines.append(f"Model: {model} - Evaluated on single prompt baseline (Prompts {valid_prompts}).\n\n")

    # Save results to a file
    with open("consistency_anova_results.txt", "w", encoding="utf-8") as out_file:
        out_file.writelines(output_lines)

if __name__ == "__main__":
    main()
