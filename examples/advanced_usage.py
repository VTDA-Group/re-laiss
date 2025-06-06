"""
Advanced usage examples for reLAISS.

This script demonstrates advanced features of reLAISS including:
- Using theorized lightcurves
- Swapping host galaxies
- Using PCA for dimensionality reduction
- Setting maximum neighbor distances
- Tweaking ANNOY parameters
- Making corner plots
- Advanced anomaly detection with parameter tuning
- Host swapping in anomaly detection
"""

import os
import pandas as pd
import numpy as np
import relaiss as rl
import astropy.units as u

def create_theorized_lightcurve():
    """Create a simple theorized lightcurve for demonstration."""
    # Create time points
    times = np.linspace(0, 100, 50) * u.day
    # Create magnitudes (simple gaussian)
    mags = 20 + 2 * np.exp(-(times.value - 50)**2 / 100)
    # Create errors
    errors = np.ones_like(mags) * 0.1
    
    # Create DataFrame in ANTARES format
    df = pd.DataFrame({
        'ant_mjd': times.to(u.day).value,
        'ant_mag': mags,
        'ant_magerr': errors,
        'ant_passband': ['g' if i % 2 == 0 else 'R' for i in range(len(times))]
    })
    return df

def main():
    # Create output directories
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./sfddata-master', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./timeseries', exist_ok=True)
    
    # Initialize the client
    client = rl.ReLAISS()
    
    # Example 1: Using PCA for dimensionality reduction
    print("\nExample 1: Using PCA")
    client.load_reference(
        path_to_sfd_folder='./sfddata-master',
        use_pca=True,
        num_pca_components=20,  # Keep 20 PCA components
    )
    
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',  # Using the test transient
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors using PCA:")
    print(neighbors_df)
    
    # Example 2: Using theorized lightcurve
    print("\nExample 2: Using theorized lightcurve")
    theorized_lc = create_theorized_lightcurve()
    
    # Need to provide a host galaxy when using theorized lightcurve
    neighbors_df = client.find_neighbors(
        theorized_lightcurve_df=theorized_lc,
        host_ztf_id='ZTF21abbzjeq',  # Use this transient's host
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors for theorized lightcurve:")
    print(neighbors_df)
    
    # Example 3: Swapping host galaxies
    print("\nExample 3: Swapping host galaxies")
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        host_ztf_id='ZTF21aakswqr',  # Use a different host
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors with swapped host:")
    print(neighbors_df)
    
    # Example 4: Setting maximum neighbor distance
    print("\nExample 4: Setting maximum neighbor distance")
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        max_neighbor_dist=0.5,  # Only return neighbors within this distance
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors within distance threshold:")
    print(neighbors_df)
    
    # Example 5: Tweaking ANNOY parameters
    print("\nExample 5: Tweaking ANNOY parameters")
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        search_k=2000,  # Increase search_k for more accurate results
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    print("\nNearest neighbors with tweaked ANNOY parameters:")
    print(neighbors_df)
    
    # Example 6: Making corner plots
    print("\nExample 6: Making corner plots")
    # Get neighbors from a new search
    neighbors_df = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',
        n=5,
        plot=True,
        save_figures=True,
        path_to_figure_directory='./figures'
    )
    
    # Get primer_dict separately
    from relaiss.search import primer
    primer_dict = primer(
        lc_ztf_id='ZTF21abbzjeq',
        theorized_lightcurve_df=None,
        host_ztf_id=None,
        dataset_bank_path=client.bank_csv,
        path_to_timeseries_folder='./',
        path_to_sfd_folder=client.path_to_sfd_folder,
        lc_features=client.lc_features,
        host_features=client.host_features,
        num_sims=0,
        save_timeseries=False,
    )
    
    # Create corner plots using the primer_dict
    from relaiss.plotting import corner_plot
    corner_plot(
        neighbors_df=neighbors_df,
        primer_dict=primer_dict,
        path_to_dataset_bank=client.bank_csv,
        path_to_figure_directory='./figures',
        save_plots=True
    )
    
    # Example 7: Advanced anomaly detection with parameter tuning
    # TODO! 

if __name__ == "__main__":
    main() 
