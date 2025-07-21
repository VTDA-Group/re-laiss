import numpy as np
import pandas as pd
import ngtpy as ngt
import os
from pathlib import Path
import tempfile

# Test fixture for building an NGT index for testing
def build_test_ngt_index(test_databank_path, lc_features=None, host_features=None):
    """Build an NGT index from the test dataset bank for testing.
    
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
        (index, index_path, object_ids) - the ngt index, path to temp file, and array of object ids
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
    
    # Create temp directory for NGT index
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "test_index.ngt")
    
    # Create NGT index
    index_dim = feat_arr.shape[1]
    ngt.create(index_path.encode(), index_dim, distance_type="L2")
    index = ngt.Index(index_path.encode())
    
    # Add items to index
    for i, obj_id in enumerate(df_bank.index):
        index.insert(feat_arr_scaled[i].astype(np.float32))
    
    # Build index
    index.build_index()
    
    return index, index_path, np.array(df_bank.index)

def find_neighbors(index, idx_arr, query_vector, n=5):
    """Find neighbors using the test NGT index.
    
    Parameters
    ----------
    index : ngt.Index
        The NGT index to query
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
    res = index.search(query_vector.astype(np.float32), n)
    neighbor_indices, distances = zip(*res)
    
    # Get ZTF IDs of neighbors
    neighbor_ids = idx_arr[neighbor_indices]
    
    return neighbor_ids, distances 
