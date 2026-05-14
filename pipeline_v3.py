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
    # for each settings file, generate a bloc slate profile using slate pl 
    num_reps = config["num_reps"] # number of preference profile samples to generate per district
    run_name = config["run_name"]
    # for each districting plan and each district of that plan, generate NUM_REPS preference profiles and save them to file
    for rep in range(num_reps):
        for dc in config["districting_configs"]:
            for generative_mode in ['slate-pl', 'name-cumulative']:
                settings_folder = Path(f"outputs/{run_name}/settings/{dc['num_districts']}_districts")
                for settings_path in tqdm(settings_folder.glob("*.json"), total=len(list(settings_folder.glob("*.json"))), desc=f"Generating preference profiles using {generative_mode} model for each district plan of {dc['num_districts']}-district plans"): # loop through all plans and all districts 
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
        for candidate in elected_candidates:
            winners.append(str(next(iter(candidate))))
        return winners
    run_name = config["run_name"]
    for generative_mode in ['slate-pl', 'name-cumulative']:
        election_results_folder = Path(f"outputs/attempt_5_split/election_results/{generative_mode}")
        election_results_folder.mkdir(parents=True, exist_ok=True)
        for dc in config["districting_configs"]:
            profile_files = Path(f"outputs/{run_name}/profiles/{dc['num_districts']}_districts/{generative_mode}")
            all_election_results = []
            profile_files_paths = []
            profile_files_v1_50 = [f for f in profile_files.glob("*_v*.csv") if any(f.name.endswith(f"_v{i}.csv" ) for i in range (1,51))]
            for profile_file in tqdm(profile_files_v1_50, total = len(list(profile_files_v1_50)), desc=f"Simulating election results for profiles generated with {generative_mode} model for each district plan of {dc['num_districts']}-district plans"):
                pp = get_profile(profile_file)
                profile_files_paths.append(str(profile_file))
                if dc["num_winners_per_district"] > 1:
                    election_results = {}
                    # stv, borda, cumulative, ranked pairs
                    if type(pp) == RankProfile:
                        elected_stv = STV(pp, n_seats=dc["num_winners_per_district"], tiebreak='random').get_elected()
                        elected_borda = Borda(pp, n_seats=dc["num_winners_per_district"], tiebreak='random').get_elected()
                        elected_ranked_pairs = RankedPairs(pp, n_seats=dc["num_winners_per_district"], tiebreak='random').get_elected()
                        election_results["stv"] = get_winners(elected_stv)
                        election_results["borda"] = get_winners(elected_borda)
                        election_results["ranked_pairs"] = get_winners(elected_ranked_pairs)
                    if type(pp) == ScoreProfile:
                        elected_cumulative = Cumulative(pp, n_seats = dc["num_winners_per_district"], tiebreak='random').get_elected()
                        election_results["cumulative"] = get_winners(elected_cumulative)
                else:
                    # plurality, irv
                    
                    election_results = {}
                    if type(pp) == RankProfile:
                        elected_plurality = Plurality(pp, n_seats=1, tiebreak='random').get_elected()
                        elected_irv = STV(pp, n_seats=1, tiebreak='random').get_elected()
                        election_results['plurality'] = get_winners(elected_plurality)
                        election_results['irv'] = get_winners(elected_irv)
                    else:
                        election_results['plurality'] = ["~"]
                        election_results['irv'] = ['~']
                all_election_results.append(election_results)
            
            # save election results to file
            out_path = election_results_folder / f"{run_name}_{dc['num_districts']}_districts_{dc['num_winners_per_district']}_winners_election_results.json"
            json_info = {
                "run_name" : run_name,
                "num_districts": dc["num_districts"],
                "num_winners_per_district": dc["num_winners_per_district"],
                "election_results": all_election_results,
                "profile_files" : profile_files_paths,
                "preference_profile_generative_model" : generative_mode
            }
            with open(out_path, "w") as f:
                json.dump(json_info, f, indent=4)


def main(config):
    #generate_districting_plans(config)
    #create_settings_files(config)
    #generate_preference_profiles(config)
    config["run_name"] = "attempt_5"
    simulate_elections(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)