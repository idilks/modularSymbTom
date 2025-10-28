#!/usr/bin/env python3
"""Test script to verify wandb heatmap integration works"""

import numpy as np
import tempfile
import os
import sys
sys.path.append('codebase')

from codebase.utils import plot

# Mock wandb logger for testing
class MockWandbLogger:
    def __init__(self):
        self.wandb = self
        self.logged_data = {}
    
    def log(self, data):
        self.logged_data.update(data)
        print(f"Would log to wandb: {list(data.keys())}")
    
    def Image(self, plt_figure):
        return "MockImage"
    
    def plotly_chart(self, fig):
        return "MockPlotlyChart"

def test_heatmap_integration():
    print("Testing wandb heatmap integration...")
    
    # Create fake causal scores (layers x heads)
    n_layers, n_heads = 6, 8
    fake_scores = np.random.rand(n_layers, n_heads) * 0.5
    
    # Add some high-scoring heads to make it interesting
    fake_scores[2, 3] = 0.9  # layer 2, head 3
    fake_scores[4, 1] = 0.8  # layer 4, head 1
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_logger = MockWandbLogger()
        
        # Test the plot function with wandb integration
        plot(
            result_array=fake_scores,
            save_folder=temp_dir,
            metric_name="Test Causal Scores",
            xlabel_name="Head Index", 
            ylabel_name="Layer Index",
            wandb_logger=mock_logger
        )
        
        # Check if files were created
        expected_file = os.path.join(temp_dir, "Test Causal Scores_heatmap.png")
        assert os.path.exists(expected_file), f"Heatmap PNG not saved at {expected_file}"
        
        # Check if wandb logging was attempted
        print(f"Logged data keys: {list(mock_logger.logged_data.keys())}")
        assert len(mock_logger.logged_data) > 0, "No data logged to wandb"
        
        print("✓ Test passed! Wandb heatmap integration working")

if __name__ == "__main__":
    test_heatmap_integration()