from joblib import Parallel, delayed
from pathlib import Path
import json
import zipfile
import tempfile
import os

from votekit import RankProfile, ScoreProfile
from votekit.elections import Plurality, FastSTV as STV, Borda, Cumulative, RankedPairs

# Simulate election helper functions 
def get_profile(filepath, generative_mode):
    return ScoreProfile.from_csv(filepath) if generative_mode == "name-cumulative" else RankProfile.from_csv(filepath)

def get_winners(elected_candidates):
    return [winner for winners in elected_candidates for winner in winners]

def process_pp(zip_path, csv_file, generative_mode : str, num_winners : int):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        data = zf.read(csv_file)
    # write a tmp file of the csv file data. Once exit with block, tmp file will be deleted
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='wb') as tmp:
        tmp.write(data) 
        tmp.flush() # ensures the write hits disk
        pp = get_profile(tmp.name, generative_mode)
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
    return (election_results, csv_file)

def _simulate_elections(config) :
    """Run elections over all voter profiles. 

    For a ScoreProfile, cumulative election is used for MMD (Multi Member Districts)
    For a RankProfile, STV, Borda, and Ranked Pairs for MMD; Plurality and IRV for SMD (Single Member Districts)
    Election results are stored in a json per plan and preference profile sample for that plan
    plan_results : { (plan_idx, pp_sample) : {districts : [{district_num, election_results}]}
    
    Args:
        config : json file containing all necessary parameters for generating preference profile
    
    """ 
    run_name = config["run_name"]
    for generative_mode in ['name-cumulative', 'slate-pl']:
        election_results_folder = Path(f"outputs/{run_name}/election_results/{generative_mode}")
        election_results_folder.mkdir(parents=True, exist_ok=True)
        for dc in config["districting_configs"]:
            if generative_mode == 'name-cumulative' and dc.num_winners_per_district == 1:
                continue
            # TODO: Add this to the filepath creator function and remove folder nesting
            zip_path = Path(f"outputs/{run_name}/profiles/{dc.num_districts}_districts/{generative_mode}.zip")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                profile_names = zf.namelist()
            print(f"{generative_mode} - {dc.num_districts} : {len(profile_names)}")
            all_election_results = Parallel(n_jobs=-1)(delayed(process_pp)(zip_path, pp, generative_mode, dc.num_winners_per_district) for pp in profile_names)
            # save election results to file
            out_path = election_results_folder / f"{run_name}_{dc.num_districts}_districts_{dc.num_winners_per_district}_winners_election_results.json"
            election_results, fps = zip(*all_election_results)
            json_info = {
                "run_name" : run_name,
                "num_districts": dc.num_districts,
                "num_winners_per_district": dc.num_winners_per_district,
                "election_results": election_results,
                "profile_files" : fps,
                "preference_profile_generative_model" : generative_mode
            }
            with open(out_path, "w") as f:
                json.dump(json_info, f, indent=4)