#!/usr/bin/env python3
"""
Analyze results from multiple CSDI runs.
Calculates mean and standard deviation for timing and metrics (RMSE, MAE, CRPS).
"""

import json
import numpy as np
import os
from datetime import datetime

def analyze_results(results_file="./results/all_runs_results.json"):
    """
    Analyze results from multiple runs and calculate statistics.
    """
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    # Load results data
    with open(results_file, "r") as f:
        all_runs_data = json.load(f)
    
    if len(all_runs_data) == 0:
        print("No runs found in results file.")
        return
    
    # Group data by missing ratio
    grouped_data = {}
    for run in all_runs_data:
        missing_ratio = run["missing_ratio"]
        if missing_ratio not in grouped_data:
            grouped_data[missing_ratio] = []
        grouped_data[missing_ratio].append(run)
    
    # Calculate statistics for each missing ratio
    summary_stats = {
        "analysis_timestamp": datetime.now().isoformat(),
        "number_of_missing_ratios": len(grouped_data),
        "total_runs": len(all_runs_data),
        "results_by_missing_ratio": {}
    }
    
    print("\n" + "="*80)
    print("CSDI Results Summary")
    print("="*80)
    
    for missing_ratio in sorted(grouped_data.keys()):
        runs = grouped_data[missing_ratio]
        
        # Timing stats
        training_times = [run["training_time_seconds"] for run in runs]
        testing_times = [run["testing_time_seconds"] for run in runs]
        
        # Metrics stats (filter out None values)
        mse_values = [run["mse"] for run in runs if run.get("mse") is not None]
        rmse_values = [run["rmse"] for run in runs if run.get("rmse") is not None]
        mae_values = [run["mae"] for run in runs if run.get("mae") is not None]
        crps_values = [run["crps"] for run in runs if run.get("crps") is not None]
        crps_sum_values = [run["crps_sum"] for run in runs if run.get("crps_sum") is not None]
        
        def calc_stats(values):
            if len(values) == 0:
                return {"mean": None, "std": None, "values": []}
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            return {"mean": mean, "std": std, "values": values}
        
        ratio_stats = {
            "number_of_runs": len(runs),
            "training_time": calc_stats(training_times),
            "testing_time": calc_stats(testing_times),
            "mse": calc_stats(mse_values),
            "rmse": calc_stats(rmse_values),
            "mae": calc_stats(mae_values),
            "crps": calc_stats(crps_values),
            "crps_sum": calc_stats(crps_sum_values),
        }
        
        summary_stats["results_by_missing_ratio"][str(missing_ratio)] = ratio_stats
        
        # Print summary for this missing ratio
        print(f"\nMissing Ratio: {missing_ratio}")
        print("-"*40)
        print(f"  Number of runs: {len(runs)}")
        if mse_values:
            print(f"  MSE:   {ratio_stats['mse']['mean']:.6f} ± {ratio_stats['mse']['std']:.6f}")
        if rmse_values:
            print(f"  RMSE:  {ratio_stats['rmse']['mean']:.6f} ± {ratio_stats['rmse']['std']:.6f}")
        if mae_values:
            print(f"  MAE:   {ratio_stats['mae']['mean']:.6f} ± {ratio_stats['mae']['std']:.6f}")
        if crps_values:
            print(f"  CRPS:  {ratio_stats['crps']['mean']:.6f} ± {ratio_stats['crps']['std']:.6f}")
        if crps_sum_values:
            print(f"  CRPS_sum: {ratio_stats['crps_sum']['mean']:.6f} ± {ratio_stats['crps_sum']['std']:.6f}")
        print(f"  Training time: {ratio_stats['training_time']['mean']:.2f}s ± {ratio_stats['training_time']['std']:.2f}s")
        print(f"  Testing time:  {ratio_stats['testing_time']['mean']:.2f}s ± {ratio_stats['testing_time']['std']:.2f}s")
    
    print("\n" + "="*80)
    
    # Ensure results directory exists
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    summary_file = os.path.join(results_dir, "results_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=4)
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    analyze_results()
