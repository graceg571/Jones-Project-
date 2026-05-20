import json
from pyben import PyBenDecoder
import geopandas as gpd
from tqdm import tqdm

POP_COL = "TOTPOP"
VAP_COL = "VAP"
BVAP_COL = "BVAP"

#Create Settings Helper functions 
def _calculate_bloc_proportions(focus_group_vap : int, total_vap : int, focus_group_prefix : str, other_group_prefix: str) -> dict:
    if total_vap <= 0:
        raise ValueError(f"Voting Age Population is {total_vap}. There's no one to vote!")
    focus_group_prop = round(focus_group_vap / total_vap, 2)
    return {focus_group_prefix : focus_group_prop, other_group_prefix : 1-focus_group_prop}

def _make_slates_per_bloc(num_candidates_per_slate : int, group_1_prefix : str, group_2_prefix : str) -> dict:
    group_1_candidates = []
    group_2_candidates = []
    for cand_num in range(num_candidates_per_slate):
        group_1_candidates.append(group_1_prefix + str(cand_num+1))
        group_2_candidates.append(group_2_prefix + str(cand_num+1))
    return {group_1_prefix : group_1_candidates, group_2_prefix : group_2_candidates}

def _fill_settings_dict(district_label : int, plan_label : int, districting_config : dict, config : dict, bloc_proportions : dict, slate_candidates : dict) -> dict:
    settings = {
                    "district" : str(district_label),
                    "sample" : str(plan_label),
                    "bloc_proportions" : bloc_proportions,
                    "slate_to_candidates" : slate_candidates,
                    "cohesion_parameters" : config["cohesion_values"],
                    "alphas" : config["alpha_values"],
                    "num_voters" : config["total_population"] // districting_config["num_districts"],
                    "total_seats" : districting_config["num_districts"] * districting_config["num_winners_per_district"], 
                    "chain_length" : config["chain_length"],
                    # TODO: determine the number of reps (preference profiles to generate per settings file) to be equal across districting configurations, right now its up to the user
                    # "num_reps" : config["chain_length"] if districting_config["num_districts"] == 1 else 1
                    
                }
    return settings

def _decode_plan_assignments(filepath)->list:
    all_plans = []
    for assignment in PyBenDecoder(filepath):
        all_plans.append(assignment)
    return all_plans

def _create_settings_files(config):
    """
    Writes a setting file per plan and district with all necessary parameters to generate a preference profile. 
        
    Args:
        config : json file containing all necessary parameters for generating preference profile
    
    """
    run_name = config["run_name"]
    # read the graph and extract demographic data of interest 
    gdf = gpd.read_file(config["shapefile_path"])
    demo_df = gdf[[POP_COL, VAP_COL, BVAP_COL]].copy()

    for dc_idx, dc in enumerate(config["districting_configs"]):
        all_plans = _decode_plan_assignments(config["plan_files"][dc_idx])

        for plan_idx, assignment in tqdm(enumerate(all_plans), total=len(all_plans), desc=f"Creating settings files for each district of {dc['num_districts']}-district plans"):
            # get the VAP population and focus/interest group VAP population for this plan
            demo_df["district"] = assignment 
            district_demos = demo_df.groupby("district")[[POP_COL, VAP_COL, BVAP_COL]].sum()

            # iterate through each district's demographics to generate a settings file 
            for district_label, row in district_demos.iterrows():
                black_bloc_symbol = "B"
                nonblack_bloc_symbol = "NB"
                bloc_proportions = _calculate_bloc_proportions(row[BVAP_COL], row[POP_COL], black_bloc_symbol, nonblack_bloc_symbol)
                slate_candidates = _make_slates_per_bloc(dc["num_candidates_per_slate"], black_bloc_symbol, nonblack_bloc_symbol)
                settings = _fill_settings_dict(district_label, plan_idx, dc, config, bloc_proportions, slate_candidates)
                out_path = config["settings_folder"] / f"{run_name}_{dc['num_districts']}_districts_settings_sample_{plan_idx}_district_{district_label+1}.json"
                with open(out_path, "w") as f:
                    json.dump(settings, f, indent=4)
