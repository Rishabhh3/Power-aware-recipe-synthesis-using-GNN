import os
import io
import pandas as pd
import torch
from torch.utils.data import Dataset

class StreamingPowerDataset(Dataset):
    def __init__(self, csv_file, tensor_dir):
        """
        Initializes the streaming dataset by loading the labels mapping.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.tensor_dir = tensor_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Extract Physical Target Metrics
        power = float(self.data_frame.iloc[idx]['power'])
        area = float(self.data_frame.iloc[idx]['area'])
        delay = float(self.data_frame.iloc[idx]['delay'])
        targets = torch.tensor([power, area, delay], dtype=torch.float32)

        # 2. Extract Text Recipe String
        recipe = self.data_frame.iloc[idx]['recipe_string']

        # 3. Stream the Compressed Graph Tensor natively using BytesIO to prevent OOM
        run_id = self.data_frame.iloc[idx]['run_id']
        tensor_path = os.path.join(self.tensor_dir, f"run_{run_id}.pt")
        
        with open(tensor_path, 'rb') as f:
            tensor_buffer = io.BytesIO(f.read())
        
        graph_data = torch.load(tensor_buffer, weights_only=False)

        return graph_data, recipe, targets

# Optional test execution block
if __name__ == "__main__":
    dataset = StreamingPowerDataset(csv_file="data/labels.csv", tensor_dir="data/tensors")
    print(f"Dataset Size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample_graph, sample_recipe, sample_targets = dataset[0]
        print(f"Sample Targets (P, A, D): {sample_targets}")
        print(f"Sample Recipe: {sample_recipe}")
        print(f"Sample Graph Nodes: {sample_graph.x.shape[0]}")