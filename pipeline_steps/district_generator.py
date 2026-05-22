from gerrychain import Graph, Partition, MarkovChain
from pathlib import Path
import geopandas as gpd
import networkx as nx
from gerrychain.updaters import Tally
from functools import partial
from gerrychain.proposals import recom 
from gerrychain.accept import always_accept
from gerrychain.constraints import contiguous
from pyben import PyBenEncoder

POP_COL = "TOTPOP"
VAP_COL = "VAP"
BVAP_COL = "BVAP"

# District Generator Helper Functions
def _run_chain(graph : Graph, num_districts : int, population_column : str, population_tolerance : float, chain_length : int) -> MarkovChain:
    
    # create intital partition of graph (initial state of Markov chain)
    my_updaters = {
        "population": Tally(population_column),
    }
    partition = Partition.from_random_assignment(
        graph = graph,
        n_parts = num_districts,
        epsilon = population_tolerance,
        pop_col = population_column,
        updaters = my_updaters
    )
    ideal_pop = sum(partition["population"].values()) / num_districts

    # run the markov chain with the Recom proposal (merge 2 adjacent districts and re-split with a spanning tree)
    proposal = partial(
        recom,
        pop_col = population_column,
        pop_target = ideal_pop,
        epsilon = population_tolerance,
    )
    chain = MarkovChain(
        proposal=proposal,
        constraints=[],
        accept=always_accept,
        initial_state=partition,
        total_steps=chain_length,
    )

    return chain

def _build_dual_graph(filepath : str, ) -> Graph:
    gdf = gpd.read_file(filepath)
    # build the dual graph (all blocs are connected = single connected component)
    graph = Graph.from_geodataframe(gdf)
    # relabel nodes as 0-indexed integers
    graph = Graph.from_networkx(nx.convert_node_labels_to_integers(graph, first_label=0))
    return graph

def _save_district_assignment_vectors(filepath : str, plans : MarkovChain | list[int], graph_node_order : list):
    with PyBenEncoder(filepath, overwrite=True) as encoder:
        if isinstance(plans, MarkovChain):
            for partition in plans.with_progress_bar():
                assignment_series = partition.assignment.to_series()
                ordered_assignment = assignment_series.loc[graph_node_order].astype(int).tolist()
                encoder.write(ordered_assignment)
        else: # partition assignment is already list of integers
            encoder.write(plans)

def _generate_districting_plans(config):
    """ Generates and stores districting plans using MCMC with the Recom proposal
        
    Args:
        config : json file containing all necessary parameters to work with MCMC
    
    """
    run_name = config["run_name"]

    graph = _build_dual_graph(config["shapefile_path"])
    graph_node_order = list(graph.nodes) # used to reorder district assignments later 
    
    # save graph for settings writer
    graph_path = Path(f"outputs/{run_name}/{run_name}_graph.json")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_json(graph_path)
    config["graph_path"] = graph_path

    # generate districting plans with MCMC
    district_nums = [dc.num_districts for dc in config["districting_configs"]]

    for num_districts in district_nums:
        plans = None
        if num_districts == 1:
            print("Only one district, skipping Markov chain generation since no districting plan yet")
            plans = [0] * len(graph_node_order)
        else:
            plans = _run_chain(graph, num_districts, POP_COL, config["population_tolerance"], config["chain_length"])
        
        output_path = config["plans_folder"] / f"{run_name}_{num_districts}_districts.jsonl.ben"
        
        _save_district_assignment_vectors(output_path, plans, graph_node_order)

