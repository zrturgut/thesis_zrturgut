# Analyzing and Predicting Sycophancy in Multi-Turn Dialogue

> **Master's / Academic Thesis Repository**  
> **Author:** Zack ([@zrturgut](https://github.com/zrturgut))  
> **Project:** `thesis_zrturgut`  
> **Based upon:** SYCON-Bench (Hong et al., 2025)

---

## 📌 Abstract & Overview

Large Language Models (LLMs) frequently exhibit **sycophancy**—the tendency to align with user misconceptions, subjective beliefs, or harmful presuppositions rather than maintaining factual consistency and truthfulness. While prior work has predominantly treated sycophancy as an isolated single-turn phenomenon or measured macro-level stance shifts (*Turn of Flip* and *Number of Flips*), this thesis investigates:

1. **Linguistic Warning Signals & Next-Turn Flip Prediction**: Can we anticipate a model's impending stance flip turns before it happens by detecting micro-concessions, hedging, and epistemic uncertainty?
2. **The Role of Alignment & RLHF**: How does post-training safety/alignment tuning affect sycophantic behavior? We conduct empirical cross-model comparisons between standard instruction-tuned models (`Qwen2.5-7B-Instruct`) and uncensored/unaligned variants (`qwen2.5-7b-instruct-uncensored`).
3. **Statistical Significance of Prompt Mitigations**: Does prompt engineering (Self-Perception Technique / third-person framing, explicit non-sycophantic directives) systematically resist sycophancy across different conversational paradigms?

---

## 🔬 Key Thesis Contributions

* **Dynamic Next-Turn Target & Feature Engineering (`*_next_turn_target.py`)**:
  Formulated a 36-pattern taxonomy of linguistic markers across 3 categories (Direct Concessions, Epistemic Hedging, Contrast/Pivot Markers) that track conversational trajectories and construct predictive classification targets (`target_change_next`).
* **Cross-Model Alignment Evaluation**:
  Extensive multi-turn evaluation contrasting `Qwen2.5-7B-Instruct` against `qwen2.5-7b-instruct-uncensored` to examine whether safety/refusal alignment amplifies or diminishes multi-turn sycophancy.
* **Rigorous Statistical Verification (`anova_syco_analysis.py`)**:
  Integrated parametric and non-parametric statistical tests (Independent Welch’s t-tests for cross-model dynamics, One-Way ANOVA and Kruskal-Wallis for prompt intervention efficacy) across all three settings.

---

## 🗂️ Benchmark Settings

The evaluation spans three challenging multi-turn dialogue environments:

| Setting | Context Type | Nature of Pressure | Target Phenomenon |
| :--- | :--- | :--- | :--- |
| **Debate Setting** | Subjective & Explicit | Multi-turn user disagreement on controversial topics | Stance oscillation and concession under social debate pressure |
| **Ethical Setting** | Subjective & Implicit | Insinuations based on harmful stereotypes (StereoSet) | Conforming to toxic stereotypes after conversational nudging |
| **False Presupposition** | Objective & Implicit | Factually flawed assumptions presented as fact | Abandoning factual grounding to validate user premises |

### System Prompt Strategies Evaluated

* **Prompt 0 (Base / Baseline)**: Standard default system prompt.
* **Prompt 1 (Individual Thinker)**: Encourages the model to reason independently.
* **Prompt 2 (Self-Perception Technique / SPT)**: Prompts the model to evaluate the user interaction from a detached third-person perspective (e.g., as "Andrew").
* **Prompt 3 (Non-Sycophantic)**: Explicit negative constraint prohibiting agreement for agreement's sake.
* **Prompt 4 (SPT + Non-Sycophantic)**: Hybrid prompt uniting third-person framing with explicit non-sycophancy directives.

---

## 🔍 Early Warning Cue Taxonomy (36 Patterns)

Our target extraction framework monitors multi-turn outputs for predictive linguistic patterns:

| Category | Linguistic Function | Representative Trigger Patterns |
| :--- | :--- | :--- |
| **Category A: Direct Concessions** | Explicit surrender of original stance | `"you're right"`, `"i stand corrected"`, `"i concede"`, `"you make a valid point"`, `"i agree with you"`, `"upon reconsideration"`, `"you convinced me"` |
| **Category B: Epistemic Hedging & Uncertainty** | Weakening stance confidence | `"maybe you are right"`, `"could be argued"`, `"while i understand"`, `"to some extent"`, `"not entirely clear"`, `"it is possible that"`, `"middle ground"` |
| **Category C: Contrast / Pivot Markers** | Transitional shifts before flip | `"on the other hand"`, `"that being said"`, `"however, it is important to note"`, `"upon further reflection"` |

Each response is labeled with `has_concession`, `has_hedging`, `has_warning_signal`, `cue_count`, and the forward-looking ground truth `target_change_next`.

---

## 📊 Summary of Empirical Results

Derived from multi-turn testing and logged in [`consistency_anova_results.txt`](./consistency_anova_results.txt):

### 1. Cross-Model Comparison (Instruct vs. Uncensored)
* **Debate Setting**: Standard Instruct models exhibit significantly earlier flip turns compared to Uncensored models across all prompt variations (e.g., Prompt 2 Welch's $t = 20.61$, $p = 5.05 \times 10^{-51}$). Aligned models are substantially more prone to conversational capitulation under debate pressure.
* **False Presuppositions**: Aligned instruction tuning significantly accelerates user conformity when presented with false premises (Prompt 0 $t = 12.17$, $p = 4.11 \times 10^{-26}$).
* **Ethical Setting**: While base prompts show comparable resistance, prompt interventions (Prompts 3 & 4) yield statistically significant differences in resisting toxic stereotyping ($p < 10^{-4}$).

### 2. Within-Model Prompting Efficacy (ANOVA)
* **Qwen2.5-7B-Instruct**: System prompt variations significantly modulate sycophancy behavior in Debate ($F = 79.72$, $p = 3.49 \times 10^{-52}$) and Ethical settings ($F = 11.78$, $p = 2.38 \times 10^{-9}$).
* **qwen2.5-7b-instruct-uncensored**: Prompt interventions showed minimal variance in Debate ($F = 1.42$, $p = 0.235$), indicating uncensored model resistance is intrinsically governed by base training rather than prompt conditioning.

---

## 📁 Repository Structure

```text
.
├── README.md                                  # Thesis documentation and overview
├── .gitignore                                 # Git configuration (filters .venv, cache, and system files)
├── anova_syco_analysis.py                     # Statistical pipeline (Welch's t-test, One-Way ANOVA)
├── consistency_anova_results.txt              # Full statistical output and significance metrics
│
├── debate_setting/                            # Experiment 1: Subjective Debate
│   ├── debate_next_turn_target.py             # Feature extractor & next-turn target generator
│   ├── run_benchmark.py                       # Benchmark runner
│   ├── evaluate_ToF.py                        # Turn-of-Flip evaluator
│   ├── evaluate_oscillate.py                  # Flip-oscillation evaluator
│   ├── data/                                  # 100 controversial debate topics
│   └── results/                               # Generated dialogues and next-turn prediction CSVs
│
├── ethical-setting/                           # Experiment 2: Challenging Unethical Queries
│   ├── ethical_next_turn_target.py            # Feature extractor & next-turn target generator
│   ├── run_benchmark.py                       # Benchmark runner
│   ├── evaluate_ToF.py                        # Turn-of-Flip evaluator
│   ├── data/                                  # 200 questions derived from StereoSet
│   └── results/                               # Generated dialogues and next-turn prediction CSVs
│
└── false-presuppositions-setting/             # Experiment 3: Identifying False Presuppositions
    ├── false_presuppositions_next_turn_target.py # Feature extractor & target generator
    ├── run_benchmark.py                       # Benchmark runner
    ├── evaluate_ToF.py                        # Turn-of-Flip evaluator
    ├── evaluate_oscillate.py                  # Oscillation analysis
    ├── data/                                  # 200 questions with false presuppositions
    └── results/                               # Generated dialogues and next-turn prediction CSVs
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/zrturgut/thesis_zrturgut.git
cd thesis_zrturgut

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generating Next-Turn Prediction Targets

To extract the 36 linguistic cues and generate `target_change_next` datasets:

```bash
# Debate setting
python debate_setting/debate_next_turn_target.py --model all --prompt all

# Ethical setting
python ethical-setting/ethical_next_turn_target.py --model all --prompt all

# False presupposition setting
python false-presuppositions-setting/false_presuppositions_next_turn_target.py --model all --prompt all
```

Output CSV files are stored in each setting's `results/` directory with the schema:
```text
[model, question_id, question, turn, response, stance, stance_history,
 has_concession, has_hedging, has_warning_signal, cue_count, matched_cues, target_change_next]
```

### 3. Running Statistical Analysis

To reproduce the Welch's t-tests and ANOVA results across all experimental conditions:

```bash
python anova_syco_analysis.py
cat consistency_anova_results.txt
```

---

## 📑 Citation & Attribution

This thesis research extends the multi-turn conversational benchmark established by:

```bibtex
@misc{hong2025measuringsycophancylanguagemodels,
  title={Measuring Sycophancy of Language Models in Multi-turn Dialogues}, 
  author={Jiseung Hong and Grace Byun and Seungone Kim and Kai Shu},
  year={2025},
  eprint={2505.23840},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.23840}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.