import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from parser.utils import get_problem_data

class GraphDataset(Dataset):
    def __init__(self, root_dir):
        self.entries = []

        for task_dir in os.listdir(root_dir):
            task_path = os.path.join(root_dir, task_dir)
            if not os.path.isdir(task_path):
                continue  

            csv_file = os.path.join(task_path, 'solutions.csv')
            if os.path.exists(csv_file): 
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row['lp_file'] = os.path.join(task_path, row['lp_file'])
                        row['solution_file'] = os.path.join(task_path, row['solution_file'])
                        row['solve_times_file'] = os.path.join(task_path, row['solve_times_file'])
                        self.entries.append(row)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]

        A, var_map, var_features, cons_features, binary_vars = get_problem_data(row['lp_file'])

        solutions = np.load(row['solution_file'])
        solve_times = np.load(row['solve_times_file'])

        A = A.coalesce()  
        edge_index = A.indices() 
        edge_attr = A.values() 

        if binary_vars.numel() == 0:
            binary_vars = torch.zeros(var_features.shape[0], dtype=torch.bool)

        return {
            'edge_index': edge_index, 
            'edge_attr': edge_attr, 
            'var_features': torch.tensor(var_features, dtype=torch.float32),  
            'cons_features': torch.tensor(cons_features, dtype=torch.float32), 
            'binary_vars': binary_vars,  
            'solutions': torch.tensor(solutions, dtype=torch.float32), 
            'solve_times': torch.tensor(solve_times, dtype=torch.float32) 
        }

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    

    dataset = GraphDataset(csv_file='solutions/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        print(batch['edge_index'].shape) 
        print(batch['solutions'].shape)  
        break
