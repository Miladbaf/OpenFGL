#!/usr/bin/env python3
"""
Compare all FedALA methods including FedALA-R
Loads fedala_results.npy and fedala_r_results.npy
"""

import numpy as np

print("="*80)
print("FEDALA METHODS COMPARISON")
print("="*80)

# Load both result files
try:
    results_main = np.load('fedala_results.npy', allow_pickle=True).item()
    print("✓ Loaded fedala_results.npy")
except:
    print("✗ Could not load fedala_results.npy")
    results_main = {}

try:
    results_r = np.load('fedala_r_results.npy', allow_pickle=True).item()
    print("✓ Loaded fedala_r_results.npy")
except:
    print("✗ Could not load fedala_r_results.npy")
    results_r = {}

# Combine results
datasets = ['Cora', 'CiteSeer', 'PubMed']
all_methods = ['fedavg', 'fedala', 'fedala_mp', 'fedala_r']

# Reorganize to dataset-first structure
combined = {}
for dataset in datasets:
    combined[dataset] = {}
    
    # Get results from main file
    for method in ['fedavg', 'fedala', 'fedala_mp']:
        if method in results_main and dataset in results_main[method]:
            combined[dataset][method] = results_main[method][dataset]
        else:
            combined[dataset][method] = []
    
    # Get FedALA-R results
    if 'fedala_r' in results_r and dataset in results_r['fedala_r']:
        combined[dataset]['fedala_r'] = results_r['fedala_r'][dataset]
    else:
        combined[dataset]['fedala_r'] = []

print("\n" + "="*80)
print("DETAILED RESULTS BY DATASET")
print("="*80)

for dataset in datasets:
    print(f"\n{dataset}:")
    print("-"*80)
    
    for method in all_methods:
        accs = combined[dataset][method]
        valid_accs = [a for a in accs if not np.isnan(a)]
        
        if len(valid_accs) == 0:
            print(f"  {method:12s}: NO DATA")
        else:
            mean_acc = np.mean(valid_accs)
            std_acc = np.std(valid_accs) if len(valid_accs) > 1 else 0.0
            print(f"  {method:12s}: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

print("\n" + "="*80)
print("COMPARISON TABLE (Mean Accuracy)")
print("="*80)

# Create comparison table
print(f"\n{'Method':<15} {'Cora':<12} {'CiteSeer':<12} {'PubMed':<12} {'Average':<12}")
print("-"*80)

method_means = {}
for method in all_methods:
    accs_all = []
    row = [method]
    
    for dataset in datasets:
        accs = combined[dataset][method]
        valid_accs = [a for a in accs if not np.isnan(a)]
        
        if len(valid_accs) > 0:
            mean_acc = np.mean(valid_accs)
            row.append(f"{mean_acc:.4f}")
            accs_all.append(mean_acc)
        else:
            row.append("N/A")
    
    # Overall average
    if accs_all:
        overall_mean = np.mean(accs_all)
        row.append(f"{overall_mean:.4f}")
        method_means[method] = overall_mean
    else:
        row.append("N/A")
    
    print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

print("\n" + "="*80)
print("IMPROVEMENTS OVER FedAvg (%)")
print("="*80)

# Get FedAvg baseline
fedavg_results = {}
for dataset in datasets:
    accs = combined[dataset]['fedavg']
    valid_accs = [a for a in accs if not np.isnan(a)]
    if len(valid_accs) > 0:
        fedavg_results[dataset] = np.mean(valid_accs)

if fedavg_results:
    print(f"\n{'Method':<15} {'Cora':<12} {'CiteSeer':<12} {'PubMed':<12} {'Average':<12}")
    print("-"*80)
    
    for method in ['fedala', 'fedala_mp', 'fedala_r']:
        improvements = []
        row = [method]
        
        for dataset in datasets:
            if dataset in fedavg_results:
                accs = combined[dataset][method]
                valid_accs = [a for a in accs if not np.isnan(a)]
                
                if len(valid_accs) > 0:
                    mean_acc = np.mean(valid_accs)
                    improvement = ((mean_acc - fedavg_results[dataset]) / fedavg_results[dataset]) * 100
                    row.append(f"{improvement:+.2f}%")
                    improvements.append(improvement)
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        
        # Average improvement
        if improvements:
            avg_improvement = np.mean(improvements)
            row.append(f"{avg_improvement:+.2f}%")
        else:
            row.append("N/A")
        
        print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")

print("\n" + "="*80)
print("BEST METHOD PER DATASET")
print("="*80)

for dataset in datasets:
    best_method = None
    best_acc = -1
    
    for method in all_methods:
        accs = combined[dataset][method]
        valid_accs = [a for a in accs if not np.isnan(a)]
        
        if len(valid_accs) > 0:
            mean_acc = np.mean(valid_accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_method = method
    
    if best_method:
        improvement = 0
        if dataset in fedavg_results and best_method != 'fedavg':
            improvement = ((best_acc - fedavg_results[dataset]) / fedavg_results[dataset]) * 100
        
        print(f"{dataset:12s}: {best_method:12s} ({best_acc:.4f} = {best_acc*100:.2f}%)", end='')
        if improvement > 0:
            print(f" [+{improvement:.2f}% vs FedAvg]")
        else:
            print()

print("\n" + "="*80)
print("RANKING BY OVERALL PERFORMANCE")
print("="*80)

# Rank methods by average performance
if method_means:
    ranked = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6} {'Method':<15} {'Average Accuracy':<20} {'vs FedAvg':<12}")
    print("-"*80)
    
    fedavg_mean = method_means.get('fedavg', 0)
    for rank, (method, mean_acc) in enumerate(ranked, 1):
        improvement = ""
        if method != 'fedavg' and fedavg_mean > 0:
            imp = ((mean_acc - fedavg_mean) / fedavg_mean) * 100
            improvement = f"{imp:+.2f}%"
        
        print(f"{rank:<6} {method:<15} {mean_acc:.4f} ({mean_acc*100:.2f}%)    {improvement:<12}")

print("\n" + "="*80)
print("✅ Analysis Complete!")
print("="*80)
