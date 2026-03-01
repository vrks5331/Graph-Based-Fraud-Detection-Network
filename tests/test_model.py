import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn_model import FraudGNN


@pytest.fixture
def mock_graph_data():
    """Create a small mock PyG Data object for testing the model."""
    # 5 nodes, 3 features each
    x = torch.randn(5, 3)
    
    # 4 edges (directed for message passing)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4],
        [1, 0, 2, 1, 4, 3]
    ], dtype=torch.long)
    
    # Edge features (3 features per edge)
    edge_attr = torch.randn(6, 3)
    
    # Labels (binary classification)
    y = torch.tensor([0, 1, 0, 0, 1], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def test_model_initialization():
    """Test that the model initializes correctly with given dimensions."""
    model = FraudGNN(in_channels=3, hidden_channels=16, out_channels=2)
    
    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 16
    assert model.conv2.in_channels == 16
    assert model.conv2.out_channels == 16 # Not 2, the classifier handles the output class count
    assert model.classifier.in_features == 16
    assert model.classifier.out_features == 2


def test_model_forward_pass(mock_graph_data):
    """Test that the model can process a graph and output correct shape logits."""
    model = FraudGNN(in_channels=3, hidden_channels=16, out_channels=2)
    model.eval()
    
    with torch.no_grad():
        out = model(mock_graph_data.x, mock_graph_data.edge_index, mock_graph_data.edge_attr)
        
    # Output should be [num_nodes, num_classes]
    assert out.shape == (5, 2)
    
    # Output should not contain NaNs
    assert not torch.isnan(out).any()


def test_model_training_step(mock_graph_data):
    """Test that the model can compute loss and backpropagate."""
    model = FraudGNN(in_channels=3, hidden_channels=16, out_channels=2)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    out = model(mock_graph_data.x, mock_graph_data.edge_index, mock_graph_data.edge_attr)
    
    # Compute loss
    loss = criterion(out, mock_graph_data.y)
    
    # Backpropagate
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients were computed for the first layer
    assert model.conv1.lin_l.weight.grad is not None
    
    # Take a step
    optimizer.step()
