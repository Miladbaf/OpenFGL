import numpy as np
path = r"/OpenFGL/ihsan/results_runtime_graphfl_clients.npy"
obj = np.load(path, allow_pickle=True).item()
runs = obj["runs"]
print(type(runs), len(runs))
print(runs[0].keys())