from gerrychain import Graph, Partition, MarkovChain
from pathlib import Path
import geopandas as gpd
import networkx as nx
from gerrychain.updaters import Tally
from functools import partial
from gerrychain.proposals import recom 
from gerrychain.accept import always_accept
from gerrychain.constraints import contiguous
import gzip 
from tqdm import tqdm
import jsonlines as jl
from votekit.ballot_generator import BlocSlateConfig, slate_pl_profile_generator
import pandas as pd
import json
from votekit.ballot_generator import (
    BlocSlateConfig,
    slate_pl_profile_generator,
    name_cumulative_profile_generator)
from votekit import RankProfile, ScoreProfile
from votekit.elections import Plurality, FastSTV as STV, Borda, Cumulative, RankedPairs
import argparse 
from glob import glob
from joblib import Parallel, delayed
import re 
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter 
import random

# TODO: Add these as user inputs/config file
POP_COL = "TOTPOP"
VAP_COL = "VAP"
BVAP_COL = "BVAP"



def generate_districting_plans(config):
    # read the graph and demographic data
    gdf = gpd.read_file(config["shapefile_path"])
    for col in [POP_COL, VAP_COL, BVAP_COL]:
        gdf[col] = gdf[col].fillna(0).astype(int)

    # County-level summary
    total_pop  = gdf[POP_COL].sum()
    total_vap  = gdf[VAP_COL].sum()
    total_bvap = gdf[BVAP_COL].sum()
    zero_pop   = (gdf[POP_COL] == 0).sum()

    print(f"    Total population  : {total_pop}")
    print(f"    Voting age pop    : {total_vap}")
    print(f"    Black VAP (BVAP)  : {total_bvap}  ({100*total_bvap/total_vap:.1f}% of VAP)")
    print(f"    Non-Black VAP     : {total_vap-total_bvap}  ({100*(total_vap-total_bvap)/total_vap:1f}% of VAP)")
    print(f"\n    Zero-population blocks: {zero_pop} of {len(gdf)} ({100*zero_pop/len(gdf):1f}%)")

    # build the dual graph 
    graph = Graph.from_geodataframe(gdf)
    # relabel nodes as 0-indexed integers for list-based assignment serialization 
    graph = Graph.from_networkx(nx.convert_node_labels_to_integers(graph, first_label=0))

    # generate districting plans with MCMC

    run_name = config["run_name"]
    chain_length = config["chain_length"]
    district_nums = [int(d["num_districts"]) for d in config["districting_configs"]]
    output_dir = Path(f"outputs/{run_name}/districts")
    graph_node_order = list(graph.nodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    for num_districts in district_nums:
        if num_districts == 1:
            print("Only one district, skipping Markov chain generation since no districting plan yet")
            assignment = [0] * len(graph_node_order)
            output_path = output_dir / f"{run_name}_{num_districts}_districts.jsonl.gz"
            print(f"Saving districting plans to: {output_path}")
            with gzip.open(output_path, mode="wt", encoding="utf-8") as gz_file:
                writer = jl.Writer(gz_file)
                writer.write({"assignment": assignment, "sample": 1,})
            continue
        # create intital partition of graph (initial state of Markov chain)
        my_updaters = {
            "population": Tally(POP_COL),
            "vap": Tally(VAP_COL),
            "bvap": Tally(BVAP_COL),
        }
        partition = Partition.from_random_assignment(
            graph = graph,
            n_parts = num_districts,
            epsilon = config["population_tolerance"],
            pop_col = POP_COL,
            updaters = my_updaters
        )
        ideal_pop = sum(partition["population"].values()) / num_districts
        # run the markov chain with the Recom proposal (merge 2 adjacent districts and re-split with a spanning tree)
        proposal = partial(
            recom,
            pop_col = POP_COL,
            pop_target = ideal_pop,
            epsilon = config["population_tolerance"],
        )
        chain = MarkovChain(
            proposal=proposal,
            constraints=[],
            accept=always_accept,
            initial_state=partition,
            total_steps=chain_length,
        )
        output_path = output_dir / f"{run_name}_{num_districts}_districts.jsonl.gz"
        print(f"Saving districting plans to: {output_path}")
        with gzip.open(output_path, mode="wt", encoding="utf-8") as gz_file:
            writer = jl.Writer(gz_file)
            for sample_num, step in enumerate(tqdm(chain, total=chain_length, desc=f"Generating {num_districts}-district plans"), start=1):
                assignment = list(step.assignment.to_series().loc[graph_node_order].astype(int))
                writer.write({"assignment": assignment, "sample": sample_num,})
            writer.close()

def create_settings_files(config):

    run_name = config["run_name"]
    # read the graph and demographic data
    gdf = gpd.read_file(config["shapefile_path"])
    demo_df = gdf[[POP_COL, VAP_COL, BVAP_COL]].copy()
    for dc in config["districting_configs"]:
        path_to_districting = f"outputs/{run_name}/districts/{run_name}_{dc['num_districts']}_districts.jsonl.gz"
        all_plans = []
        with gzip.open(path_to_districting, mode="rt", encoding="utf-8") as gz_file:
            reader = jl.Reader(gz_file)
            for plan in reader:
                all_plans.append(plan)
        settings_folder = Path(f"outputs/{run_name}/settings/{dc['num_districts']}_districts")
        settings_folder.mkdir(parents=True, exist_ok=True)
        for plan_idx, plan in tqdm(enumerate(all_plans), total=len(all_plans), desc=f"Creating settings files for each district of {dc['num_districts']}-district plans"):
            assignment = plan["assignment"]
            demo_df["district"] = assignment # district assignment per node 
            district_demos = demo_df.groupby("district")[[POP_COL, VAP_COL, BVAP_COL]].sum()
            for district_label, row in district_demos.iterrows():

                # make settings
                vap = int(row[VAP_COL])
                bvap = int(row[BVAP_COL])
                nbvap = vap - bvap
                bvap_prop = round(bvap / vap, 2) if vap > 0 else 0
                nbvap_prop = 1.0 - bvap_prop
                if (bvap_prop + nbvap_prop) > 1.0 :
                    print(f"District {district_label}: BVAP% = {bvap_prop:.2f}, NBVAP% = {nbvap_prop:.2f}") 
                black_candidates = []
                nonblack_candidates = []
                for cand_num in range(dc["num_candidates_per_slate"]):

                    black_candidates.append("B" + str(cand_num+1))
                    nonblack_candidates.append("NB" + str(cand_num+1))

                settings = {
                    "district" : str(district_label),
                    "sample" : str(plan["sample"]),
                    "bloc_proportions" : {
                        "B": bvap_prop,
                        "NB": nbvap_prop
                    },
                    "slate_to_candidates" : {
                        "B" : black_candidates,
                        "NB" : nonblack_candidates
                    },
                    "candidates" : black_candidates + nonblack_candidates,
                    "cohesion_parameters" : config["cohesion_values"],
                    "bvap" : str(int(row[BVAP_COL])),
                    "vap" : str(int(row[VAP_COL])),
                    "alphas" : config["alpha_values"],
                    "num_voters" : config["total_population"] // dc["num_districts"],
                    "total_seats" : 6, 
                    "chain_length" : config["chain_length"],
                    
                }
                out_path = settings_folder / f"{run_name}_{dc['num_districts']}_districts_settings_sample_{plan['sample']}_district_{district_label+1}.json"
                with open(out_path, "w") as f:
                    json.dump(settings, f, indent=4)

# process settings files into Bloc Slate configurations 
def generate_preference_profiles(config):
    def use_generative_model(generative_mode, config, total_points=0):
        if generative_mode == "slate-pl":
            return slate_pl_profile_generator(config)
        elif generative_mode == "name-cumulative":
            return name_cumulative_profile_generator(config, total_points=total_points)
        else:
            raise ValueError("Non-compatible generative mode for preference profile given")

    run_name = config["run_name"]
    # for each districting plan and each district of that plan, generate NUM_REPS preference profiles and save them to file
    for dc in config["districting_configs"]:
        num_reps = dc["num_reps"]
        for rep in range(num_reps):
            for generative_mode in ['slate-pl', 'name-cumulative']:
                settings_folder = Path(f"outputs/{run_name}/settings/{dc['num_districts']}_districts")
                for settings_path in settings_folder.glob("*.json"): # loop through all plans and all districts 
                    settings_file = open(settings_path, 'r')
                    settings = json.load(settings_file)
                    bloc_config = BlocSlateConfig(
                        n_voters=settings["num_voters"],
                        slate_to_candidates=settings["slate_to_candidates"],
                        bloc_proportions=settings["bloc_proportions"],
                        cohesion_mapping=settings["cohesion_parameters"],

                    )
                    setting_file_stem = Path(settings_path).stem
                    profile_folder = Path(f"outputs/{run_name}/profiles/{dc['num_districts']}_districts/{generative_mode}")
                    profile_folder.mkdir(parents=True, exist_ok=True)
                    profile_output = profile_folder / f"{setting_file_stem.replace('settings', 'profile')}_rep_v{rep+1}.csv"
                    bloc_config.set_dirichlet_alphas(settings["alphas"])
                    n_points = dc['num_winners_per_district'] if generative_mode == "name-cumulative" else 0
                    profile = use_generative_model(generative_mode, bloc_config, total_points=n_points)
                    profile.to_csv(profile_output)
                    settings_file.close()


def simulate_elections(config):
    # for each preference profile, simulate an election and save results to file 
    def get_profile(filepath):
        return ScoreProfile.from_csv(filepath) if "name-cumulative" in str(filepath) else RankProfile.from_csv(filepath)
    def get_winners(elected_candidates):
        winners = []
        return [winner for winners in elected_candidates for winner in winners]
    def process_pp(pf, num_winners):
        pp = get_profile(pf)
        election_results = {}
        if num_winners > 1:
            # stv, borda, cumulative, ranked pairs
            if type(pp) == RankProfile:
                elected_stv = STV(pp, n_seats=num_winners, tiebreak='random').get_elected()
                elected_borda = Borda(pp, n_seats=num_winners, tiebreak='random').get_elected()
                elected_ranked_pairs = RankedPairs(pp, n_seats=num_winners, tiebreak='random').get_elected()
                election_results["stv"] = get_winners(elected_stv)
                election_results["borda"] = get_winners(elected_borda)
                election_results["ranked_pairs"] = get_winners(elected_ranked_pairs)
            if type(pp) == ScoreProfile:
                elected_cumulative = Cumulative(pp, n_seats = num_winners, tiebreak='random').get_elected()
                election_results["cumulative"] = get_winners(elected_cumulative)
        else:
            # plurality, irv
            if type(pp) == RankProfile:
                elected_plurality = Plurality(pp, n_seats=1, tiebreak='random').get_elected()
                elected_irv = STV(pp, n_seats=1, tiebreak='random').get_elected()
                election_results['plurality'] = get_winners(elected_plurality)
                election_results['irv'] = get_winners(elected_irv)
        return (election_results, pf)
    
    run_name = config["run_name"]
    for generative_mode in ['name-cumulative', 'slate-pl']:
        election_results_folder = Path(f"outputs/{run_name}/election_results/{generative_mode}")
        election_results_folder.mkdir(parents=True, exist_ok=True)
        for dc in config["districting_configs"]:
            profile_fp = Path(f"outputs/{run_name}/profiles/{dc['num_districts']}_districts/{generative_mode}")
            profile_files = glob(f"{profile_fp}/*.csv")
            print(f"{generative_mode} - {dc['num_districts']} : {len(profile_files)}")
            all_election_results = Parallel(n_jobs=-1)(delayed(process_pp)(pp, dc["num_winners_per_district"]) for pp in profile_files)
            # save election results to file
            out_path = election_results_folder / f"{run_name}_{dc['num_districts']}_districts_{dc['num_winners_per_district']}_winners_election_results.json"
            election_results, fps = zip(*all_election_results)
            json_info = {
                "run_name" : run_name,
                "num_districts": dc["num_districts"],
                "num_winners_per_district": dc["num_winners_per_district"],
                "election_results": election_results,
                "profile_files" : fps,
                "preference_profile_generative_model" : generative_mode
            }
            with open(out_path, "w") as f:
                json.dump(json_info, f, indent=4)

def summarize_results(config) -> dict:
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
                print(f"number of elections: {len(results['election_results'])}")
                print(f"number of preference profile files {len(results['profile_files'])}")
                # extract information of the districting plan, sample #, and district 
                district_plan = {}
                summary_results = {} 
                # create a dictionary of rule dictionaries with list of district results
                grouped = defaultdict(lambda: defaultdict(list))
                summary_grouped = defaultdict(lambda: defaultdict(int))
                for (profile_file, election_result) in zip(results["profile_files"], results["election_results"]):
                    # each election result file represents all the districting plan results for a particular district configuration and generative mode 
                    # get the plan idx, pp iteration, and district for each election result
                    match = re.search(r'sample_(\d+)_district_(\d+)_rep_v(\d+)', profile_file)
                    plan_index = int(match.group(1))
                    district = int(match.group(2))
                    pp_sample = int(match.group(3))
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

def visualize_results(summary_dict : dict, config):
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
    # TODO: convert the tuple key to a string when summarizing 
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
    generate_districting_plans(config)
    create_settings_files(config)
    generate_preference_profiles(config)
    simulate_elections(config)
    summary_dict = summarize_results(config)
    visualize_results(summary_dict, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)