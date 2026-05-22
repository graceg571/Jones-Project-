from votekit.ballot_generator import (BlocSlateConfig, slate_pl_profile_generator, name_cumulative_profile_generator)
import json
import re
from pathlib import Path
import zipfile
import io 

# Generate Preference Profile Helper functions 
def use_generative_model(generative_mode, config, total_points=0):
    generator = None
    if generative_mode == "slate-pl":
        generator = slate_pl_profile_generator(config)
    elif generative_mode == "name-cumulative":
        generator = name_cumulative_profile_generator(config, total_points=total_points)
    else:
        raise ValueError("Non-compatible generative mode for preference profile given")
    return generator

def _build_blocSlateConfig_from_settings(settings_filepath) -> BlocSlateConfig:
    settings_file = open(settings_filepath, 'r')
    settings = json.load(settings_file)
    bloc_config = BlocSlateConfig(
        n_voters=settings["num_voters"],
        slate_to_candidates=settings["slate_to_candidates"],
        bloc_proportions=settings["bloc_proportions"],
        cohesion_mapping=settings["cohesion_parameters"],
    )
    bloc_config.set_dirichlet_alphas(settings["alphas"])
    settings_file.close()
    return bloc_config

def _generate_pp_per_settings(bloc_config : BlocSlateConfig, generative_modes : list, num_reps : int, settings_filename, profile_folder, num_winners : int):
    # TODO: make generative modes user defined, can move filepath logic out of this function, add metadata to pp so not dependent on filename
    # TODO: reduce the nesting structure of files and save all pp to one result - use metadata of pp to determine how made instead of folder structre
    # for generative_mode in generative_modes:
    #     for pp_rep in range(num_reps):
    #         profile_output = profile_folder / f"{settings_filename.replace('settings', 'profile')}_rep_v{pp_rep+1}_using_{generative_mode}.csv"
    #         n_points = num_winners if generative_mode == "name-cumulative" else 0
    #         # TODO: do not generate a score pp for single member districts -- no election rule applicable right now 
    #         profile = use_generative_model(generative_mode, bloc_config, total_points=n_points)
    #         profile.to_csv(profile_output)
    pass

def _generate_preference_profiles(config):
    """Generate a preference profile for each district within a districting plan. 
    
    Uses the slate-pl (ranked profile) and name-cumulative (scored profile) models to generate preference profiles.
    Preference Profiles are saved to a csv 
            
    Args:
        config : json file that dictates how many preference profiles to generate per district per plan
    """
    run_name = config["run_name"]

    for dc in config["districting_configs"]:
        for generative_mode in ['slate-pl', 'name-cumulative']:
            if generative_mode == 'name-cumulative' and dc.num_winners_per_district == 1:
                continue
            settings_folder = Path(f"outputs/{run_name}/settings/{dc.num_districts}_districts")
            zip_path = Path(f"outputs/{run_name}/profiles/{dc.num_districts}_districts/{generative_mode}.zip")
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            for rep in range(dc.num_reps):
                for settings_path in settings_folder.glob("*.json"):
                    bloc_config = _build_blocSlateConfig_from_settings(settings_path)
                    n_points = dc.num_winners_per_district if generative_mode == "name-cumulative" else 0
                    profile = use_generative_model(generative_mode, bloc_config, total_points=n_points)

                    setting_file_stem = Path(settings_path).stem
                    profile_output_name = f"{setting_file_stem.replace('settings', 'profile')}_rep_v{rep+1}.csv"
                    # zip profile files together using gzip compression
                    with zipfile.ZipFile(zip_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(profile_output_name, profile.to_csv())
                    
                    


                    
