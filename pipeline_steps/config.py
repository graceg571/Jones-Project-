from dataclasses import dataclass

@dataclass 
class DistrictingConfig:
    num_districts: int
    num_winners_per_district: int
    num_candidates_per_slate: int
    num_reps: int = 1