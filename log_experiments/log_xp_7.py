import json
import glob
import numpy as np
import os
import re

RESULTS_DIR = "../outputs/tsg/xp_7/"

def extract_alpha_from_folder_name(folder_name: str) -> str:
    """Extract alpha_mode from folder name like 'value_elicitation_alpha_mode_unimodal'."""
    # Match pattern like alpha_mode_unimodal or alpha_mode_bimodal
    match = re.search(r"alpha_mode_(\w+)", folder_name)
    if match:
        # Return the mode name (e.g., 'unimodal', 'bimodal')
        mode_name = match.group(1)
        return mode_name
    return "unknown"

def log_xp_7(output_folder):
    # Get all subdirectories (each corresponds to a different alpha_mode)
    subdirs = [d for d in os.listdir(output_folder) 
               if os.path.isdir(os.path.join(output_folder, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {output_folder}")
        return
    
    print(f"Found {len(subdirs)} subdirectories in {output_folder}")
    
    # Collect results across all alpha experiments
    results_by_alpha = {}
    
    # Process each alpha subfolder
    for subdir in subdirs:
        subdir_path = os.path.join(output_folder, subdir)
        
        # Extract alpha from the subfolder name itself
        alpha = extract_alpha_from_folder_name(subdir)
        
        if alpha == "unknown":
            print(f"Could not extract alpha from folder: {subdir}")
            continue
        
        print(f"\nProcessing: {subdir} (alpha_mode={alpha})")
        print(f"  Full path: {subdir_path}")
        
        # Debug: show what's in this folder
        try:
            contents = os.listdir(subdir_path)
            print(f"  Contents: {contents}")
        except Exception as e:
            print(f"  Error listing contents: {e}")
            continue
        
        # Find all log.json files in subfolders (structure: alpha_folder/centralized/run_X/log.json)
        log_files = glob.glob(os.path.join(subdir_path, "*/*/log.json"))
        
        if not log_files:
            print(f"  No log.json files found matching pattern: {subdir_path}/*/*/log.json")
            # Try to find ANY json files
            all_json = glob.glob(os.path.join(subdir_path, "**/*.json"), recursive=True)
            if all_json:
                print(f"  But found these JSON files: {all_json}")
            continue
        
        print(f"  Found {len(log_files)} log.json files")
        
        values = []
        for log_file in log_files:
            try:
                with open(log_file) as f:
                    data = json.load(f)
                
                # Check if results section exists and has value_elicitation
                if "results" not in data:
                    print(f"  No 'results' section in {log_file}")
                    continue
                    
                if "value_elicitation" not in data["results"]:
                    print(f"  No 'value_elicitation' in {log_file}")
                    continue
                
                value_elicitation = data["results"]["value_elicitation"]
                values.append(value_elicitation)
                
            except Exception as e:
                print(f"  Error processing {log_file}: {e}")
                continue
        
        if values:
            results_by_alpha[alpha] = values
            avg = float(np.mean(values))
            std = float(np.std(values))
            print(f"  Found {len(values)} runs: {avg:.2f}% ± {std:.2f}%")
    
    if not results_by_alpha:
        print("\nNo valid value_elicitation results found")
        return
    
    # Compute overall summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    summary = {}
    for alpha, vals in results_by_alpha.items():
        avg = float(np.mean(vals))
        std = float(np.std(vals))
        summary[alpha] = {
            "n_runs": len(vals),
            "avg_value_elicitation": avg,
            "std_value_elicitation": std,
            "all_values": vals
        }
        print(f"[alpha_mode={alpha}] Value elicitation: {avg:.2f}% ± {std:.2f}% (n={len(vals)})")
    
    # Save overall summary
    summary_path = os.path.join(output_folder, "value_information_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    log_xp_7(RESULTS_DIR)
