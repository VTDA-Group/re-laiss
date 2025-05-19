"""
Basic usage examples for reLAISS.

This script demonstrates the basic functionality of reLAISS including:
- Finding optimal number of neighbors
- Running nearest neighbor search
- Using Monte Carlo simulations
- Adjusting feature weights
"""

import os
import relaiss as rl

def main():
    # Create output directories
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./sfddata-master', exist_ok=True)
    
    # Initialize the client
    client = rl.ReLAISS()
    
    # Load reference data
    # Note: SFD dust maps will be automatically downloaded if not present
    client.load_reference(
        path_to_sfd_folder='./sfddata-master',  # Directory for SFD dust maps
        host_features=[],  # Empty list means host features are disabled
    )
    
    # Example 1: Find optimal number of neighbors
    print("\nExample 1: Finding optimal number of neighbors")
    client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',  # Using the test transient
        n=40,  # Search in a larger pool
        suggest_neighbor_num=True,  # Only suggest optimal number, don't return neighbors
        plot=True,  # Show the distance elbow plot
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    
    # Example 2: Basic nearest neighbor search
    print("\nExample 2: Basic nearest neighbor search")
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',  # Using the test transient
        n=5,  # Number of neighbors to return
        suggest_neighbor_num=False,  # Return actual neighbors
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors:")
    print(neighbors_df)
    
    # Example 3: Using Monte Carlo simulations and feature weighting
    print("\nExample 3: Using Monte Carlo simulations and feature weighting")
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',  # Using the test transient
        n=5,
        num_mc_simulations=20,  # Number of Monte Carlo simulations
        weight_lc_feats_factor=3.0,  # Up-weight lightcurve features
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors with MC simulations:")
    print(neighbors_df)

if __name__ == "__main__":
    main() 