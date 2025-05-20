import pytest
import pandas as pd
import numpy as np
import relaiss as rl
from relaiss.search import calculate_distances, primer

def test_find_neighbors_dataframe():
    client = rl.ReLAISS()
    client.load_reference(host_features=[])
    
    # Test basic neighbor finding
    df = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert np.all(df["dist"].values[:-1] <= df["dist"].values[1:])
    
    # Test with different n values
    df_large = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=10)
    assert len(df_large) == 10
    
    # Test with plot option
    df_plot = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5, plot=True)
    assert isinstance(df_plot, pd.DataFrame)

def test_find_neighbors_invalid_input():
    client = rl.ReLAISS()
    client.load_reference(host_features=[])
    
    # Test with invalid ZTF ID
    with pytest.raises(ValueError):
        client.find_neighbors(ztf_object_id="invalid_id", n=5)
    
    # Test with invalid n value
    with pytest.raises(ValueError):
        client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=-1)

def test_calculate_distances():
    # Create sample feature vectors
    query_vector = np.array([1.0, 2.0, 3.0])
    reference_vectors = np.array([[1.0, 2.0, 3.0],
                                [2.0, 3.0, 4.0],
                                [3.0, 4.0, 5.0]])
    
    distances = calculate_distances(query_vector, reference_vectors)
    assert isinstance(distances, np.ndarray)
    assert len(distances) == len(reference_vectors)
    assert distances[0] == 0  # Distance to self should be 0

def test_primer_with_ztf_id():
    # Test with ZTF ID
    result = primer(
        lc_ztf_id="ZTF21abbzjeq",
        theorized_lightcurve_df=None,
        dataset_bank_path="../data/dataset_bank.csv",
        path_to_timeseries_folder="../data/timeseries",
        path_to_sfd_folder="../data/sfd",
        save_timeseries=False,
        lc_features=['mag', 'color', 'period'],
        host_features=['redshift', 'mass']
    )
    
    assert isinstance(result, dict)
    assert 'lc_ztf_id' in result
    assert 'locus_feat_arr' in result
    assert result['lc_ztf_id'] == "ZTF21abbzjeq"

def test_primer_with_host_swap():
    # Test with host galaxy swap
    result = primer(
        lc_ztf_id="ZTF21abbzjeq",
        theorized_lightcurve_df=None,
        dataset_bank_path="../data/dataset_bank.csv",
        path_to_timeseries_folder="../data/timeseries",
        path_to_sfd_folder="../data/sfd",
        host_ztf_id="ZTF19aaaaaaa",
        lc_features=['mag', 'color', 'period'],
        host_features=['redshift', 'mass']
    )
    
    assert isinstance(result, dict)
    assert 'host_ztf_id' in result
    assert result['host_ztf_id'] == "ZTF19aaaaaaa"

def test_primer_with_theorized_lightcurve():
    # Create a theorized lightcurve DataFrame
    lightcurve_df = pd.DataFrame({
        'mjd': np.linspace(0, 100, 50),
        'mag': np.random.normal(20, 0.5, 50),
        'magerr': np.random.uniform(0.01, 0.1, 50),
        'band': ['g', 'r'] * 25
    })
    
    result = primer(
        lc_ztf_id=None,
        theorized_lightcurve_df=lightcurve_df,
        dataset_bank_path="../data/dataset_bank.csv",
        path_to_timeseries_folder="../data/timeseries",
        path_to_sfd_folder="../data/sfd",
        host_ztf_id="ZTF19aaaaaaa",  # Required when using theorized lightcurve
        lc_features=['mag', 'color', 'period'],
        host_features=['redshift', 'mass']
    )
    
    assert isinstance(result, dict)
    assert 'locus_feat_arr' in result
    assert result['lc_ztf_id'] is None

def test_primer_invalid_input():
    # Test with both lc_ztf_id and theorized_lightcurve
    with pytest.raises(ValueError):
        primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=pd.DataFrame(),
            dataset_bank_path="../data/dataset_bank.csv",
            path_to_timeseries_folder="../data/timeseries",
            path_to_sfd_folder="../data/sfd"
        )
    
    # Test with neither lc_ztf_id nor theorized_lightcurve
    with pytest.raises(ValueError):
        primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=None,
            dataset_bank_path="../data/dataset_bank.csv",
            path_to_timeseries_folder="../data/timeseries",
            path_to_sfd_folder="../data/sfd"
        )
    
    # Test with theorized_lightcurve but no host_ztf_id
    with pytest.raises(ValueError):
        primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=pd.DataFrame(),
            dataset_bank_path="../data/dataset_bank.csv",
            path_to_timeseries_folder="../data/timeseries",
            path_to_sfd_folder="../data/sfd",
            host_ztf_id=None
        )
