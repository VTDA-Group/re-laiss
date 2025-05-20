import numpy as np
import pandas as pd
import annoy
import os
from pathlib import Path
import tempfile

# Test fixture for building an Annoy index for testing
def build_test_annoy_index(test_databank_path, lc_features=None, host_features=None):
    """Build an Annoy index from the test dataset bank for testing.
    
    Parameters
    ----------
    test_databank_path : Path
        Path to the test dataset bank CSV file
    lc_features : list[str], optional
        Lightcurve features to use, by default will use some basic features
    host_features : list[str], optional
        Host features to use, by default will use some basic features
    
    Returns
    -------
    tuple
        (index, index_path, object_ids) - the annoy index, path to temp file, and array of object ids
    """
    # Default features if none provided
    if lc_features is None:
        lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    
    if host_features is None:
        host_features = ['host_ra', 'host_dec']
    
    # Load test data bank
    df_bank = pd.read_csv(test_databank_path)
    df_bank = df_bank.set_index("ztf_object_id", drop=False)
    
    # Extract features
    features = lc_features + host_features
    df_features = df_bank[features]
    
    # Create feature array and standardize
    feat_arr = np.array(df_features)
    feat_arr_scaled = (feat_arr - np.mean(feat_arr, axis=0)) / np.std(feat_arr, axis=0)
    
    # Create annoy index
    index_dim = feat_arr.shape[1]
    index = annoy.AnnoyIndex(index_dim, "manhattan")
    
    # Add items to index
    for i, obj_id in enumerate(df_bank.index):
        index.add_item(i, feat_arr_scaled[i])
    
    # Build index with 10 trees (fewer for tests to be faster)
    index.build(10)
    
    # Create temp file to save index
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "test_index.ann")
    index.save(index_path)
    
    return index, index_path, np.array(df_bank.index)

def find_neighbors(index, idx_arr, query_vector, n=5):
    """Find neighbors using the test Annoy index.
    
    Parameters
    ----------
    index : annoy.AnnoyIndex
        The Annoy index to query
    idx_arr : numpy.ndarray
        Array of object IDs
    query_vector : numpy.ndarray
        Query vector
    n : int, optional
        Number of neighbors to return, by default 5
    
    Returns
    -------
    tuple
        (ids, distances) - arrays of neighbor IDs and distances
    """
    # Query the index
    neighbor_indices, distances = index.get_nns_by_vector(
        query_vector, n, include_distances=True
    )
    
    # Get ZTF IDs of neighbors
    neighbor_ids = idx_arr[neighbor_indices]
    
    return neighbor_ids, distances 