# FedALA & FedALA-R on OpenFGL

This repository is a research fork of Open Federated Graph Learning (OpenFGL) that adds and evaluates:
- FedALA (fl_algorithm = fedala)
- FedALA-R (fl_algorithm = fedala_r) — residual-based variant

Repo: https://github.com/Miladbaf/OpenFGL/tree/main
OpenFGL paper: https://arxiv.org/abs/2408.16288

## Contents (what to run)

Main scripts:
- run_all_fedala_methods.py: runs FedAvg + FedALA + FedALA-R across datasets and seeds; saves fedala_complete_results.npy
- run_fedala_experiments.py: FedALA experiments
- run_fedala_r.py: FedALA-R experiments
- run_fedala_comparison_5clients.py: quick 5-client comparison
- run_scalability_analysis.py + generate_scalability_figure.py: scalability runs + plots
- run_per_client_analysis.py: per-client analysis + plots
- MIA_analysis.py: black-box membership inference privacy audit utilities


GraphFL scripts:
- ihsan/data_download_multi.py: Downloads datasets related to GraphFL study.
- ihsan/run_baseline_experiments.py: Runs the baseline experiments for GraphFL study.
- ihsan/run_graphfl_fedala_grid.py: Runs the FedAVG-FedALA-FedALAR experiments 
- ihsan/run_scalability_graphfl_clients: The previous scripts focus on a single client size. This script applie a cross-client analysis.
- ihsan/run_mia_graphcls.py: Runs the MIA analysis for GraphFL task.
- The remaining files and scripts in the "ihsan" folder are used to process the obtained data for presentation.
  
## Installation

1) Clone
   git clone https://github.com/Miladbaf/OpenFGL.git
   cd OpenFGL

2) Create environment + activate (choose ONE)

   Linux/macOS (bash/zsh):
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

   Windows PowerShell:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip

3) Install PyTorch + PyTorch Geometric (required)

   pip install torch torchvision torchaudio
   pip install torch-geometric

## Reproducing main results

A) Full multi-dataset, multi-seed run:
   python run_all_fedala_methods.py

Default settings inside run_all_fedala_methods.py:
- Datasets: Cora, CiteSeer, PubMed
- Seeds: 42, 123, 456
- Clients: 5
- Simulation mode: subgraph_fl_louvain
- Model: gcn
- Training: num_rounds=100, local_epoch=5, lr=0.01, weight_decay=5e-4
- Metric: accuracy

Outputs:
- Console summary (mean ± std, improvements over FedAvg)
- fedala_complete_results.npy

## Acknowledgements

Built on Open Federated Graph Learning (OpenFGL):
- https://github.com/zyl24/OpenFGL
- https://arxiv.org/abs/2408.16288
