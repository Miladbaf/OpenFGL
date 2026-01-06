# FedALA & FedALA-R on OpenFGL

This repository is a research fork of Open Federated Graph Learning (OpenFGL) that adds and evaluates:
- FedALA (fl_algorithm = fedala)
- FedALA-R (fl_algorithm = fedala_r) — residual-based variant

Repo: https://github.com/Miladbaf/OpenFGL/tree/main
Upstream OpenFGL: https://github.com/zyl24/OpenFGL
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

Common outputs:
- *.npy caches (e.g., fedala_results.npy, fedala_r_results.npy, scalability_results.npy)
- *.png / *.pdf figures (e.g., fedala_comparison.png, scalability_analysis.png, per_client_analysis.png)

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

3) Install this repo (uses pyproject.toml)
   pip install -e .

4) Install PyTorch + PyTorch Geometric (required)
   CPU-only quick start:
   pip install torch torchvision torchaudio
   pip install torch-geometric

   CUDA users:
   Install the PyTorch build matching your CUDA version, then install torch-geometric following the official PyG install matrix.

Optional (recommended): record versions for reproducibility
   python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)"
   python -c "import torch_geometric; print('pyg', torch_geometric.__version__)"

## Quick start (OpenFGL-style)

Minimal example (edit as needed):
- Set args.root to where you want datasets cached/downloaded (default: data)

Example:
  python -c "import openfgl.config as config; from openfgl.flcore.trainer import FGLTrainer; args=config.args; args.root='data'; args.dataset=['Cora']; args.simulation_mode='subgraph_fl_louvain'; args.num_clients=5; args.model=['gcn']; args.metrics=['accuracy']; args.fl_algorithm='fedala_r'; trainer=FGLTrainer(args); trainer.train()"

## Reproducing main results (recommended)

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

Note (PyTorch 2.6+): this script patches torch.load() to default weights_only=False for checkpoint compatibility.

B) Targeted runs:
   python run_fedala_experiments.py
   python run_fedala_r.py
   python run_fedala_comparison_5clients.py
   python run_scalability_analysis.py
   python generate_scalability_figure.py
   python run_per_client_analysis.py

## Aggregation / plotting

After experiments:
   python analyze_results.py
   python compare_all_results.py
   python generate_comparison_table.py

## Privacy audit (membership inference)

Run:
   python MIA_analysis.py

## Reproducibility checklist

1) Log commit hash:
   git rev-parse HEAD

2) Freeze environment:
   pip freeze > requirements_lock.txt

3) Report (in paper/appendix): OS, CPU/GPU, Python version, torch version, torch-geometric version.

## Acknowledgements

Built on Open Federated Graph Learning (OpenFGL):
- https://github.com/zyl24/OpenFGL
- https://arxiv.org/abs/2408.16288
