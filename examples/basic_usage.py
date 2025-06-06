"""
Basic usage examples for reLAISS.

This script demonstrates the basic functionality of reLAISS including:
- Finding optimal number of neighbors
- Running nearest neighbor search
- Using Monte Carlo simulations
- Adjusting feature weights
- Basic anomaly detection
"""

import os
import relaiss as rl

def main():
    # Create output directories
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./sfddata-master', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./timeseries', exist_ok=True)
    
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
        num_sims=20,  # Number of Monte Carlo simulations
        weight_lc_feats_factor=3.0,  # Up-weight lightcurve features
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors with MC simulations:")
    print(neighbors_df)
    
    # Example 4: Basic anomaly detection
    print("\nExample 4: Basic anomaly detection")
    from relaiss.anomaly import train_AD_model, anomaly_detection
    
    # First, train an anomaly detection model
    print("Training anomaly detection model...")
    model_path = train_AD_model(
        lc_features=client.lc_features,
        host_features=client.host_features,
        path_to_dataset_bank=client.bank_csv,
        path_to_sfd_folder='./sfddata-master',
        path_to_models_directory="./models",
        n_estimators=100,  # Using smaller value for faster execution
        contamination=0.02,  # Expected proportion of anomalies
        max_samples=256,  # Max samples per tree
        force_retrain=False  # Only retrain if model doesn't exist
    )
    print(f"Anomaly detection model saved to: {model_path}")


    # Run anomaly detection on a transient
    (mjd_scores, anom_scores, norm_scores) = anomaly_detection(
        client=client,
        transient_ztf_id="ZTF21abbzjeq",
        lc_features=client.lc_features,
        host_features=[],
        path_to_timeseries_folder="./laiss_final/timeseries",
        path_to_sfd_folder='./data/sfddata-master',
        path_to_dataset_bank=client.bank_csv,
        path_to_models_directory="./laiss_final/models",
        path_to_figure_directory="./laiss_final/figures/direct_example",
        save_figures=True,
        return_scores=True,
        anom_thresh=50,
        force_retrain=True,
        preprocessed_df=None
    )

    print("Anomaly detection figures saved to ./figures/AD/")

if __name__ == "__main__":
    main() 
