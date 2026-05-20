import json
import argparse 
import re 
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter 
from glob import glob
from pathlib import Path
import random


from pipeline_steps.district_generator import _generate_districting_plans
from pipeline_steps.settings_writer import _create_settings_files
from pipeline_steps.preference_profile_generator import _generate_preference_profiles
from pipeline_steps.election_simulator import _simulate_elections

#TODO: remove hard coded values from here and modules above 
POP_COL = "TOTPOP"
VAP_COL = "VAP"
BVAP_COL = "BVAP"

# TODO: Move summarize and visualize results into own module and add helper functions 
def _summarize_results(config) -> dict:
    # count number of seats won by slate B 
    run_name = config["run_name"]
    election_results_path = Path(f"outputs/{run_name}/election_results")
    summary_dict = {}
    for generative_mode in ["name-cumulative", "slate-pl"]:
        election_results_files = Path(f"{election_results_path}/{generative_mode}").glob("*.json")
        for filepath in election_results_files:
            fname = filepath.stem
            match = re.search(r'_(\d+)_districts_(\d+)_winners', fname)
            num_districts = int(match.group(1))
            num_winners = int(match.group(2))
            key = f"{num_districts} x {num_winners}"
            if key not in summary_dict:
                summary_dict[f"{num_districts} x {num_winners}"] = {
                    "number_of_districts" : num_districts,
                    "num_seats_per_district" : num_winners,
                    "generative_mode" : [],
                }
            if generative_mode not in summary_dict[key]["generative_mode"]:
                summary_dict[key]["generative_mode"].append(generative_mode)
            if generative_mode not in summary_dict[key]:
                summary_dict[key][generative_mode] = {}

            # open file 
            # count the number of seats won by B candidate per each election 
            with open(filepath) as f: 
                results = json.load(f)
                # create a dictionary of rule dictionaries with list of district results
                grouped = defaultdict(lambda: defaultdict(list))
                summary_grouped = defaultdict(lambda: defaultdict(int))
                for (profile_file, election_result) in zip(results["profile_files"], results["election_results"]):
                    # each election result file represents all the districting plan results for a particular district configuration and generative mode 
                    # get the plan idx, pp iteration, and district for each election result
                    match = re.search(r'sample_(\d+)_district_\d+_rep_v(\d+)', profile_file)
                    plan_index = int(match.group(1))
                    pp_sample = int(match.group(2))
                    idx_key = (plan_index, pp_sample)
                    for rule, winners in election_result.items():
                        if "~" not in winners:
                            grouped[idx_key][rule].extend(winners)
                        if len(winners) != num_winners:
                            print(f"incorrect number of winners for {profile_file}")
                            print(winners)
                for key, rule_winners in grouped.items():
                    expected_winners = num_districts * num_winners
                    for rule, winners in rule_winners.items():
                        if len(winners) != expected_winners:
                            print(f"Winners across districts {len(winners)} does not meet expected {expected_winners}")
                            print(key, rule, winners, profile_file)
                        b_count = sum(1 for w in winners if w.startswith("B"))
                        summary_grouped[key][rule] = b_count
            summary_dict[f"{num_districts} x {num_winners}"][generative_mode] = summary_grouped
    return summary_dict

def _visualize_results(summary_dict : dict, config):
    # subsample the summary plans to ensure each districting config has the same number of results
    def subsample_summary_plans(summary_dict : dict, num : int, seed : int = 42) -> dict:
        random.seed(seed)
        # extract the keys per summary dict (plan_idx, summary_idx)
        keys = list(summary_dict.keys())
        sampled = random.sample(keys, min(num, len(keys)))
        return {k: summary_dict[k] for k in sampled}
    for dc in summary_dict.keys():
        for mode in summary_dict[dc]['generative_mode']:
            summary_dict[dc][mode] = subsample_summary_plans(summary_dict[dc][mode], config["chain_length"])
    # update tuple key (plan_idx, sample_idx) to be a string
    # TODO: think about how nested this is. Is this necessary? 
    bcounts_dict = defaultdict(list)
    for districting_plan in summary_dict.keys():
        for mode in summary_dict[districting_plan]["generative_mode"]:
            for plan in summary_dict[districting_plan][mode]:
                for election in summary_dict[districting_plan][mode][plan]:
                    bcounts_dict[f"{election} ({districting_plan})"].append(summary_dict[districting_plan][mode][plan][election])
    rows = []
    for election_config, counts in bcounts_dict.items():
        freq_counts = Counter(counts)
        rows.append((election_config, freq_counts))
        print(f"{election_config} # results: {freq_counts.total()}")

    # max frequency for scaling the bubble 
    max_freq = max(freq for _,counts in rows for freq in counts.values())
    # plot counts of number of black seats won per election and districting plan configuration
    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.8 + 2))
    colors = [
            "#e41a1c", "#ff7f00", "#4daf4a",
            "#377eb8"
    ]
    # label = election (num_districts x num_winners), counts = frequency count for number of black seats won 
    for y_pos, (label, counts) in enumerate(rows):
    # update colors to match based on the type of election and order the labels in that way too 
        color = colors[y_pos % len(colors)]
        for n_black, freq in counts.items():
            size = (freq / max_freq) * 3000 # max size is 3000, the rest are proptionate to 3000
            ax.scatter(n_black, y_pos, s=size,
                        color=color)
    #TODO: remove this hardcoded value
    max_seats = 6
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=10)
    ax.set_xlabel("Number of seats won by B slate", fontsize=12)
    ax.set_xticks(range(max_seats + 1))
    ax.set_xlim(-0.5, max_seats + 0.5)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig("bubble_chart.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()

def main(config):
    _generate_districting_plans(config)
    _create_settings_files(config)
    _generate_preference_profiles(config)
    _simulate_elections(config)
    summary_dict = _summarize_results(config)
    _visualize_results(summary_dict, config)

def _create_filepaths(config) -> dict:
    run_name = config["run_name"]
    dc_paths = {
        "plans_folder" :  Path(f"outputs/{run_name}/districts"),
        "plan_files" : {},
        "settings_folder" : Path(f"outputs/{run_name}/settings/"),
        "profile_folder" : Path(f"outputs/{run_name}/profiles/")
    }

    for i, dc in enumerate(config["districting_configs"]):
        dc_paths["plan_files"][i] = dc_paths["plans_folder"]/ f"{run_name}_{dc['num_districts']}_districts.jsonl.ben"
    for k, filepath in dc_paths.items():
        if "folder" in k:
            filepath.mkdir(parents=True, exist_ok=True)

    config.update(dc_paths)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config = _create_filepaths(config)
    
    # calculate number of district plans per configuration
    # NUM REPS for equalization of sample sizes per districting configuration is only relevant where we don't use the markov chain (1 district)
    # TODO: Calculate the number of preference profile samples per districting configuration so they're the same
    main(config)