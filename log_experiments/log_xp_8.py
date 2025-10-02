import os
import json
from pathlib import Path
import numpy as np


def extract_total_costs(base_path: str = "../outputs/tsg/xp_8"):
    """
    Extract total_cost from all log.json files in xp_8 experiments.
    Compares centralized vs coordinated costs across runs.
    """
    base_path = Path(base_path)
    
    # Dictionary to store results: {experiment_name: {centralized: [...], coordinated: [...]}}
    experiments = {}
    
    # Iterate through all experiment folders in xp_8
    for exp_folder in sorted(base_path.iterdir()):
        if not exp_folder.is_dir():
            continue
            
        exp_name = exp_folder.name
        experiments[exp_name] = {
            'centralized': [],
            'coordinated': []
        }
        
        # Check centralized and coordinated subfolders
        for approach in ['centralized', 'coordinated']:
            approach_path = exp_folder / approach
            
            if not approach_path.exists():
                continue
            
            # Iterate through all run folders (run_1, run_2, etc.)
            for run_folder in sorted(approach_path.iterdir()):
                if not run_folder.is_dir() or not run_folder.name.startswith('run_'):
                    continue
                
                log_file = run_folder / 'log.json'
                
                if log_file.exists():
                    try:
                        with open(log_file, 'r') as f:
                            data = json.load(f)
                            total_cost = data['results']['total_cost']
                            experiments[exp_name][approach].append(total_cost)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error reading {log_file}: {e}")
    
    return experiments

def calculate_cost_increases(experiments: dict):
    """
    Calculate percentage increases and statistics for coordinated vs centralized.
    """
    results = {}
    
    for exp_name, costs in experiments.items():
        centralized_costs = costs['centralized']
        coordinated_costs = costs['coordinated']
        
        if not centralized_costs or not coordinated_costs:
            print(f"\nWarning: Missing data for {exp_name}")
            continue
        
        # Calculate percentage increases for each run
        percentage_increases = []
        for cent_cost, coord_cost in zip(centralized_costs, coordinated_costs):
            if cent_cost != 0:
                pct_increase = ((coord_cost - cent_cost) / cent_cost) * 100
                percentage_increases.append(pct_increase)
        
        results[exp_name] = {
            'centralized_costs': centralized_costs,
            'coordinated_costs': coordinated_costs,
            'percentage_increases': percentage_increases,
            'mean_increase': np.mean(percentage_increases) if percentage_increases else None,
            'std_increase': np.std(percentage_increases) if percentage_increases else None
        }
    
    return results

def print_results(results: dict):
    """
    Print formatted results.
    """
    print("\n" + "="*80)
    print("EXPERIMENT COST ANALYSIS: Centralized vs Coordinated")
    print("="*80)
    
    for exp_name, data in results.items():
        print(f"\n{'─'*80}")
        print(f"Experiment: {exp_name}")
        print(f"{'─'*80}")
        
        print(f"\nCentralized costs: {data['centralized_costs']}")
        print(f"Coordinated costs: {data['coordinated_costs']}")
        
        print(f"\nPercentage increases per run:")
        for i, pct in enumerate(data['percentage_increases'], 1):
            print(f"  Run {i}: {pct:+.2f}%")
        
        if data['mean_increase'] is not None:
            print(f"\nMean cost increase: {data['mean_increase']:+.2f}%")
            print(f"Std deviation: {data['std_increase']:.2f}%")

def main():
    # Extract costs from all experiments
    experiments = extract_total_costs()
    
    # Calculate statistics
    results = calculate_cost_increases(experiments)
    
    # Print formatted output
    print_results(results)
    
    # Optionally save to file
    output_file = "../outputs/tsg/xp_8/cost_comparison_log.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
