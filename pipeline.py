import json
import argparse 
from pathlib import Path

from pipeline_steps.district_generator import _generate_districting_plans
from pipeline_steps.settings_writer import _create_settings_files
from pipeline_steps.preference_profile_generator import _generate_preference_profiles
from pipeline_steps.election_simulator import _simulate_elections
from pipeline_steps.results_visualization import _visualize_bcounts_results
from pipeline_steps.config import DistrictingConfig

#TODO: remove hard coded values from here and modules above 
POP_COL = "TOTPOP"
VAP_COL = "VAP"
BVAP_COL = "BVAP"



def main(config):
    _generate_districting_plans(config)
    _create_settings_files(config)
    _generate_preference_profiles(config)
    _simulate_elections(config)
    _visualize_bcounts_results(config)

def _create_filepaths(config) -> dict:
    run_name = config["run_name"]
    dc_paths = {
        "plans_folder" :  Path(f"outputs/{run_name}/districts"),
        "plan_files" : {},
        "settings_folder" : Path(f"outputs/{run_name}/settings/"),
        "profile_folder" : Path(f"outputs/{run_name}/profiles/")
    }

    for i, dc in enumerate(config["districting_configs"]):
        dc_paths["plan_files"][i] = dc_paths["plans_folder"]/ f"{run_name}_{dc.num_districts}_districts.jsonl.ben"
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

    # make data structures for each districting configuration
    config["districting_configs"] = [DistrictingConfig(**dc) for dc in config["districting_configs"]]
    config = _create_filepaths(config)
    
    # calculate number of district plans per configuration
    # NUM REPS for equalization of sample sizes per districting configuration is only relevant where we don't use the markov chain (1 district)
    # TODO: Calculate the number of preference profile samples per districting configuration so they're the same
    main(config)