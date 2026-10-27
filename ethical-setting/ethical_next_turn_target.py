# ethical_next_turn_target.py
#
# Creates next-turn prediction targets for SYCON-Bench ethical-setting.
# Detects early warning cues (concessions, hedging, uncertainty) before ethical flips.
# Supports single prompts or --prompt all, and single models or --model all.
#
# Run:
#   python ethical_next_turn_target.py --model all --prompt all
#   or:
#   python ethical_next_turn_target.py --model uncensored --prompt prompt1

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS & LINGUISTIC CUES (36 PATTERNS)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_TURN = 5

LINGUISTIC_CUES = {
    # --- Category A: Direct Concessions & Surrender Phrases ---
    "you_are_right": r"\b(?:yes,?\s*)?you(?:'re| are) right\b",
    "my_mistake_or_bad": r"\bmy (?:mistake|bad|fault|apologies)\b|\bi apologize\b",
    "i_stand_corrected": r"\bi stand corrected\b",
    "i_concede": r"\bi concede\b|\bconceded\b",
    "you_make_a_point": r"\b(?:you|they) make a (?:good|great|valid|fair) point\b",
    "valid_point": r"\bvalid (?:point|argument|critique)\b",
    "fair_point": r"\bfair (?:point|critique|assessment)\b",
    "good_point": r"\bgood (?:point|counterargument)\b",
    "i_agree": r"\bi agree (?:with you|with that|completely|entirely)?\b",
    "i_see_your_point": r"\bi see (?:your point|what you mean)\b",
    "you_have_a_point": r"\byou have a point\b",
    "thats_true": r"\b(?:that is|that's) (?:true|a valid point|correct)\b",
    "valid_perspective": r"\b(?:raise a |offer a )?valid perspective\b",
    "see_merit": r"\b(?:see the |has some |merit in your )merit\b",
    "rightly_pointed_out": r"\b(?:as you )?(?:rightly|correctly) (?:pointed out|noted|stated)\b",
    "to_your_point": r"\bto your point\b",
    "upon_reconsideration": r"\b(?:upon|after) reconsideration\b|\breconsidering my position\b",
    "in_light_of_argument": r"\bin light of (?:your|this) (?:argument|point|evidence)\b",
    "you_convinced_me": r"\byou(?:\'ve| have) convinced me\b",

    # --- Category B: Hedging, Epistemic Uncertainty & Compromise ---
    "i_dont_know": r"\bi (?:don't|do not) know\b",
    "maybe_you_are_right": r"\b(?:maybe|perhaps|you might be) (?:you(?:'re| are) )?right\b",
    "could_be_argued": r"\b(?:it could|it can|one could) be argued\b",
    "while_i_understand": r"\bwhile (?:i|we) (?:understand|acknowledge|recognize)\b|\bi understand your\b",
    "on_second_thought": r"\bon second thought\b",
    "to_some_extent": r"\bto (?:some|a certain) extent\b|\bin some respects\b",
    "not_entirely_clear": r"\b(?:not entirely clear|unclear|remains uncertain)\b",
    "admittedly": r"\badmittedly\b",
    "it_is_possible": r"\b(?:it is|it's) possible that\b|\bpotentially\b",
    "valid_concerns": r"\bvalid (?:concerns|counterarguments|reservations)\b",
    "while_true": r"\bwhile (?:it is|it's|that is) true\b",
    "certainly_while": r"\bcertainly[,!]? while\b",
    "hard_to_say": r"\b(?:hard to say|cannot say for certain|open to debate)\b",
    "concessive_challenges_admission": r"\bwhile (?:the )?(?:[a-z]+\s+){0,6}(?:challenges|concerns|risks|issues|drawbacks).{1,60}(?:are significant|exist|remain|are valid)\b",
    "balanced_view": r"\b(?:balanced (?:view|perspective)|both (?:sides|perspectives) (?:offer|have))\b",
    "middle_ground_or_compromise": r"\bmiddle ground\b|\bcompromise\b",

    # --- Category C: Contrast / Pivot Markers ---
    "on_the_other_hand": r"\bon the other hand\b",
    "that_being_said": r"\b(?:that|having) (?:been|being) said\b",
    "however_important_to_note": r"\bhowever,? it(?:'s| is) (?:important|crucial) to note\b",
    "upon_further_reflection": r"\b(?:upon further (?:reflection|consideration)|reconsidering|rethinking)\b",
}

CONCESSION_KEYS = {
    "you_are_right", "my_mistake_or_bad", "i_stand_corrected", "i_concede",
    "you_make_a_point", "valid_point", "fair_point", "good_point",
    "i_agree", "i_see_your_point", "you_have_a_point", "thats_true",
    "valid_perspective", "see_merit", "rightly_pointed_out", "to_your_point",
    "upon_reconsideration", "in_light_of_argument", "you_convinced_me"
}

COMPILED_CUES = {k: re.compile(v, re.IGNORECASE) for k, v in LINGUISTIC_CUES.items()}


def extract_linguistic_cues(text):
    text_str = str(text)
    return [k for k, pattern in COMPILED_CUES.items() if pattern.search(text_str)]


# ============================================================
# 2. RESOLVE MODELS & PROMPTS
# ============================================================

def get_available_models(repo_root: Path) -> list:
    output_dir = repo_root / "ethical-setting" / "output"
    return sorted([
        d.name for d in output_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def resolve_model_folder(repo_root: Path, target_name: str) -> str:
    available = get_available_models(repo_root)
    if target_name in available:
        return target_name
    lower_map = {d.lower(): d for d in available}
    if target_name.lower() in lower_map:
        return lower_map[target_name.lower()]
    clean_target = re.sub(r"[^a-zA-Z0-9]", "", target_name).lower()
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", target_name.lower()) if t]
    matches = []
    for d in available:
        clean_d = re.sub(r"[^a-zA-Z0-9]", "", d).lower()
        if clean_target in clean_d or (tokens and all(t in clean_d for t in tokens)):
            matches.append(d)
    if "uncensored" in target_name.lower():
        unc = [m for m in matches if "uncensored" in m.lower()]
        if unc: return unc[0]
    elif matches:
        non_unc = [m for m in matches if "uncensored" not in m.lower()]
        if non_unc: return non_unc[0]
        return matches[0]
    raise FileNotFoundError(f"Model '{target_name}' not found. Available: {available}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create next-turn prediction targets and analyze early warning cues in ethical-setting."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model folder name, nickname (e.g., 'uncensored', 'instruct'), or 'all' for all models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="all",
        help="Prompt name (e.g. prompt1), comma-separated list, or 'all' to process all available prompts.",
    )
    parser.add_argument(
        "--max_turn",
        type=int,
        default=MAX_TURN,
        help=f"Maximum turns to include (default: {MAX_TURN})",
    )
    return parser.parse_args()


# ============================================================
# 3. CORE PROCESSING PER PROMPT
# ============================================================

def process_single_prompt(repo_root: Path, model_folder: str, prompt_name: str, max_turn: int):
    raw_file = repo_root / "ethical-setting" / "output" / model_folder / f"{prompt_name}.csv"
    results_file = repo_root / "ethical-setting" / "results" / model_folder / f"{prompt_name}_result_all.csv"

    if not raw_file.exists():
        print(f"Skipping {prompt_name}: raw file not found ({raw_file})")
        return None
    if not results_file.exists():
        print(f"Skipping {prompt_name}: results file not found ({results_file})")
        return None

    raw = pd.read_csv(raw_file)
    results = pd.read_csv(results_file)

    # Find response columns
    response_columns = {}
    for column in raw.columns:
        match = re.fullmatch(r"(?i)response_?(\d+)", str(column).strip())
        if match:
            turn = int(match.group(1))
            if 1 <= turn <= max_turn:
                response_columns[turn] = column

    available_turns = sorted(response_columns)
    if not available_turns:
        print(f"Skipping {prompt_name}: No response columns found.")
        return None

    question_column = None
    for column in raw.columns:
        if str(column).strip().lower() == "question":
            question_column = column
            break

    # Unpivot to turn level
    rows = []
    num_questions = min(len(raw), len(results))

    for i in range(num_questions):
        q_text = raw.loc[i, question_column] if question_column else None
        for turn in available_turns:
            resp_col = response_columns[turn]
            res_col = f"Response_{turn}" if f"Response_{turn}" in results.columns else resp_col
            stance_val = results.loc[i, res_col]
            if pd.notna(stance_val) and int(stance_val) in (0, 1):
                rows.append({
                    "model": model_folder,
                    "question_id": i + 1,
                    "question": q_text,
                    "turn": turn,
                    "response": raw.loc[i, resp_col],
                    "stance": int(stance_val),
                })

    turn_data = pd.DataFrame(rows).sort_values(["question_id", "turn"]).reset_index(drop=True)

    # Transitions
    turn_data["previous_stance"] = turn_data.groupby("question_id")["stance"].shift(1)
    turn_data["changed_from_previous"] = (turn_data["stance"] != turn_data["previous_stance"]).astype("Int64")
    turn_data.loc[turn_data["previous_stance"].isna(), "changed_from_previous"] = pd.NA

    turn_data["next_stance"] = turn_data.groupby("question_id")["stance"].shift(-1).astype("Int64")
    turn_data["target_change_next"] = (turn_data["stance"] != turn_data["next_stance"]).astype("Int64")
    turn_data.loc[turn_data["next_stance"].isna(), "target_change_next"] = pd.NA

    # Stance history
    turn_data["stance_history"] = turn_data.groupby("question_id", group_keys=False)["stance"].apply(
        lambda s: pd.Series([",".join(map(str, s.iloc[: i + 1])) for i in range(len(s))], index=s.index)
    )

    # Linguistic cues
    turn_data["matched_cues_list"] = turn_data["response"].apply(extract_linguistic_cues)
    turn_data["cue_count"] = turn_data["matched_cues_list"].apply(len)
    turn_data["has_concession"] = turn_data["matched_cues_list"].apply(
        lambda cues: int(any(c in CONCESSION_KEYS for c in cues))
    )
    turn_data["has_hedging"] = turn_data["matched_cues_list"].apply(
        lambda cues: int(any(c not in CONCESSION_KEYS for c in cues))
    )
    turn_data["has_warning_signal"] = (turn_data["cue_count"] > 0).astype(int)
    turn_data["matched_cues"] = turn_data["matched_cues_list"].apply(lambda cues: ";".join(cues))

    prediction_data = turn_data[turn_data["target_change_next"].notna()].copy()
    prediction_data = prediction_data[[
        "model", "question_id", "question", "turn", "response", "stance",
        "stance_history", "has_concession", "has_hedging", "has_warning_signal",
        "cue_count", "matched_cues", "target_change_next"
    ]]

    # Save output
    out_dir = repo_root / "ethical-setting" / "results" / model_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{prompt_name}_next_turn_prediction_target.csv"
    root_out_file = repo_root / "ethical-setting" / f"{model_folder}_{prompt_name}_next_turn_prediction_target.csv"

    prediction_data.to_csv(out_file, index=False)
    prediction_data.to_csv(root_out_file, index=False)

    # Compute stats
    st1 = prediction_data[prediction_data["stance"] == 1]
    total_st1 = len(st1)
    flips = int((st1["target_change_next"] == 1).sum()) if total_st1 > 0 else 0
    flip_rate = (flips / total_st1) * 100 if total_st1 > 0 else 0

    top_cue = "None"
    top_cue_or = 1.0
    total_flips = flips
    total_stable = total_st1 - flips
    for cue_name, pat in COMPILED_CUES.items():
        matched = st1["response"].str.contains(pat)
        if matched.sum() >= 2:
            A = int((matched & (st1["target_change_next"] == 1)).sum())
            B = int((matched & (st1["target_change_next"] == 0)).sum())
            C = total_flips - A
            D = total_stable - B
            odds_ratio = (A * D) / (B * C) if (B * C) > 0 else np.nan
            if not np.isnan(odds_ratio) and odds_ratio > top_cue_or:
                top_cue_or = odds_ratio
                top_cue = f"{cue_name} ({odds_ratio:.2f}x)"

    return {
        "model": model_folder,
        "prompt": prompt_name,
        "rows": len(prediction_data),
        "stance1_turns": total_st1,
        "flips_next": flips,
        "flip_rate": flip_rate,
        "top_cue": top_cue,
        "out_file": str(out_file)
    }


# ============================================================
# 4. MAIN DISPATCHER
# ============================================================

def main():
    args = parse_args()
    
    if args.model.lower() == "all":
        models_to_process = get_available_models(REPO_ROOT)
    else:
        models_to_process = [resolve_model_folder(REPO_ROOT, args.model)]

    all_summaries = []

    for model_folder in models_to_process:
        print("\n" + "=" * 80)
        print(f"PROCESSING ETHICAL MODEL: {model_folder}")
        print("=" * 80)

        output_dir = REPO_ROOT / "ethical-setting" / "output" / model_folder

        if args.prompt.lower() == "all":
            prompts_to_process = sorted([p.stem for p in output_dir.glob("prompt*.csv")])
        else:
            prompts_to_process = [p.strip() for p in args.prompt.split(",") if p.strip()]

        model_summaries = []
        for prompt_name in prompts_to_process:
            print(f"--> Analyzing prompt: {prompt_name}...")
            res = process_single_prompt(REPO_ROOT, model_folder, prompt_name, args.max_turn)
            if res:
                model_summaries.append(res)
                all_summaries.append(res)

        if model_summaries:
            print(f"\n--- Cross-Prompt Mitigation Summary: {model_folder} ---")
            sum_df = pd.DataFrame(model_summaries)[[
                "prompt", "rows", "stance1_turns", "flips_next", "flip_rate", "top_cue"
            ]]
            sum_df["flip_rate"] = sum_df["flip_rate"].map(lambda x: f"{x:.1f}%")
            sum_df.columns = ["Prompt", "Total Rows", "Ethical Turns", "Flips Next", "Flip Rate (%)", "Top Pre-Flip Cue"]
            print(sum_df.to_string(index=False))

    if len(all_summaries) > len(model_summaries):
        print("\n" + "=" * 80)
        print("ETHICAL SETTING: OVERALL CROSS-MODEL SUMMARY TABLE")
        print("=" * 80)
        full_df = pd.DataFrame(all_summaries)[[
            "model", "prompt", "rows", "stance1_turns", "flips_next", "flip_rate", "top_cue"
        ]]
        full_df["flip_rate"] = full_df["flip_rate"].map(lambda x: f"{x:.1f}%")
        full_df.columns = ["Model", "Prompt", "Rows", "Ethical Turns", "Flips Next", "Flip Rate (%)", "Top Pre-Flip Cue"]
        print(full_df.to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    main()
