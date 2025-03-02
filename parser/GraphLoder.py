import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import os

from parser.utils import get_a_new2



class GraphDataset(Dataset):
    def __init__(self, lp_dir, solution_dir, num_solutions=5):
        self.lp_files = [os.path.join(lp_dir, f) for f in os.listdir(lp_dir) if f.endswith('.lp')]
        self.solution_dir = solution_dir
        self.num_solutions = num_solutions

    def __len__(self):
        return len(self.lp_files)

    def __getitem__(self, idx):
        lp_path = self.lp_files[idx]
        
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(lp_path)
        
        edge_index = A.coalesce().indices()
        edge_attr = A.coalesce().values().unsqueeze(1)
        
        sol_file = os.path.join(self.solution_dir, 
                              f"{os.path.basename(lp_path).replace('.lp', '')}_sols.npy")
        solutions = torch.from_numpy(np.load(sol_file)).float()
        
        return Data(
            x=v_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            constraint_nodes=c_nodes,
            y=solutions[:self.num_solutions],
            b_vars=b_vars,
            num_vars=v_nodes.size(0),
            num_cons=c_nodes.size(0)
        )

def collate_fn(batch):
    return batch