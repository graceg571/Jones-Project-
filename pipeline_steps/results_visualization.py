import re 
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter 
from pathlib import Path
import pandas as pd 
import json

def _summarize_results(config) -> pd.DataFrame:
    # count number of seats won by slate B 
    run_name = config["run_name"]
    election_results_path = Path(f"outputs/{run_name}/election_results")
    records = []
    for generative_mode in ["name-cumulative", "slate-pl"]:
        election_results_files = Path(f"{election_results_path}/{generative_mode}").glob("*.json")
        for filepath in election_results_files:
            fname = filepath.stem
            match = re.search(r'_(\d+)_districts_(\d+)_winners', fname)
            num_districts = int(match.group(1))
            num_winners = int(match.group(2))
            total_seats = num_districts * num_winners

            with open(filepath) as f: 
                results = json.load(f)

            # accumulate winners across districts for each plan 
            grouped = defaultdict(lambda: defaultdict(list))
            for (profile_file, election_result) in zip(results["profile_files"], results["election_results"]):
                match = re.search(r'sample_(\d+)_district_\d+_rep_v(\d+)', profile_file)
                plan_index = int(match.group(1))
                pp_sample = int(match.group(2))
                for rule, winners in election_result.items():
                    if "~" in winners:
                        continue
                    if len(winners) != num_winners:
                        raise ValueError("Incorrect number of winners for {profile_file}: {winners}")
                    grouped[(plan_index, pp_sample)][rule].extend(winners)

            # determine the number of b seats won per plan + rep
            for (plan_index, pp_sample), rule_winners in grouped.items():
                for rule, winners in rule_winners.items():
                    if len(winners) != total_seats:
                        raise ValueError(f"Expected {total_seats} across districts, got {len(winners)} for plan {plan_index}, rep {pp_sample}, rule {rule}")
                    records.append({
                        "num_districts": num_districts,
                        "num_winners_per_district": num_winners,
                        "total_seats": total_seats,
                        "generative_mode": generative_mode,
                        "plan_index": plan_index,
                        "rep": pp_sample,
                        "election_rule": rule,
                        "b_seats_won": sum(1 for w in winners if w.startswith("B")), #TODO: pass along B
                    })
    df = pd.DataFrame(records)
    df.to_parquet(Path(f"outputs/{run_name}/summary.parquet"), index=False)
    return df

RULE_ORDER = ['cumulative', 'stv', 'borda', 'ranked_pairs', 'plurality', 'irv']
def _rows_sort_key(item):
    label = item[0] # election rule (districting config)
    match = re.match(r'(\S+)\s+\((\d+)\s+x\s+(\d+)\)', label)
    rule = match.group(1)
    nd = int(match.group(2))
    # determine the position of the 
    return (RULE_ORDER.index(rule) if rule in RULE_ORDER else len(RULE_ORDER), -nd)

def _visualize_results(df : pd.DataFrame, run_name: str):

    df['label'] = df['election_rule'] + " (" + df['num_districts'].astype(str) + " x " + df['num_winners_per_district'].astype(str) + ")"
    bcounts_dict = df.groupby("label")["b_seats_won"].apply(list).to_dict()

    rows = []
    for election_config, counts in bcounts_dict.items():
        freq_counts = Counter(counts)
        rows.append((election_config, freq_counts))
        print(f"{election_config} # results: {freq_counts.total()}")
    rows.sort(key= _rows_sort_key)
    # max frequency for scaling the bubble 
    max_freq = max(freq for _,counts in rows for freq in counts.values())
    # plot counts of number of black seats won per election and districting plan configuration
    _, ax = plt.subplots(figsize=(12, len(rows) * 0.8 + 2))

    colors = {
        "plurality": "#984ea3",
        "irv": "#a65628",
        "ranked_pairs": "#e41a1c",
        "borda": "#ff7f00",
        "stv": "#4daf4a",
        "cumulative": "#377eb8"
    }
    
    # label = election (num_districts x num_winners), counts = frequency count for number of black seats won 
    for y_pos, (label, counts) in enumerate(rows): 
        rule = label.split(" (")[0]
        color = colors[rule]
        for n_black, freq in counts.items():
            size = (freq / max_freq) * 3000 
            ax.scatter(n_black, y_pos, s=size,
                        color=color)
    #TODO: remove this hardcoded value
    max_seats = 6
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=10)
    ax.set_xlabel("Number of seats won by B slate", fontsize=12)
    ax.set_xticks(range(max_seats + 1))
    ax.set_xlim(-0.5, max_seats + 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_facecolor("white")
    ax.tick_params(left=False)
    plt.tight_layout()
    plt.savefig(f"outputs/{run_name}/bubble_chart.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()

def _visualize_bcounts_results(config):
    df = _summarize_results(config)
    run_name = config["run_name"]
    _visualize_results(df, run_name)