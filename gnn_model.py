import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class MultiObjectiveGNN(nn.Module):
    def __init__(self, num_node_features=3, num_operations=7, recipe_length=20):
        super(MultiObjectiveGNN, self).__init__()
        
        # ==========================================
        # LOBE 1: AIG Embedding Network (Graph)
        # ==========================================
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        
        # Projects combined Mean/Max pooling (64 + 64 = 128) down to 128
        self.graph_fc = nn.Linear(128, 128) 

        # ==========================================
        # LOBE 2: Recipe Embedding Network (Sequence)
        # ==========================================
        # Embeds the 7 possible ABC operations
        self.recipe_embedding = nn.Embedding(num_embeddings=num_operations, embedding_dim=16)
        
        # 1D Convolution to catch sequential dependencies
        self.recipe_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Flattens sequence into a 64-dimensional vector
        self.recipe_fc = nn.Linear(32 * recipe_length, 64)

        # ==========================================
        # LOBE 3: Graph-Based Regression (Output)
        # ==========================================
        # 128 (Graph) + 64 (Recipe) = 192 Input Features
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 3) # Predicts: [Power, Area, Delay]

    def forward(self, x, edge_index, batch, recipe_sequence):
        # --- Process Lobe 1 (Graph) ---
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # "Bulls-Eye" Combined Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_embed = torch.cat([x_mean, x_max], dim=1)
        graph_embed = F.relu(self.graph_fc(graph_embed))

        # --- Process Lobe 2 (Recipe) ---
        # recipe_sequence shape: [Batch, 20]
        r = self.recipe_embedding(recipe_sequence) # Shape: [Batch, 20, 16]
        r = r.permute(0, 2, 1)                     # Shape: [Batch, 16, 20] for Conv1d
        
        r = F.relu(self.recipe_conv(r))            # Shape: [Batch, 32, 20]
        r = r.view(r.size(0), -1)                  # Flatten
        recipe_embed = F.relu(self.recipe_fc(r))   # Shape: [Batch, 64]

        # --- Process Lobe 3 (Regression) ---
        combined = torch.cat([graph_embed, recipe_embed], dim=1) # Shape: [Batch, 192]
        
        out = F.relu(self.fc1(combined))
        out = F.relu(self.fc2(out))
        predictions = self.output_layer(out)
        
        return predictions