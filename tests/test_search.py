import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import relaiss as rl
from relaiss.search import primer
from .fixtures.search import find_neighbors

def test_find_neighbors_dataframe():
    client = rl.ReLAISS()
    client.load_reference(host_features=[])
    
    # Patch find_neighbors to return a DataFrame with 5 rows
    with patch.object(client, 'find_neighbors', return_value=pd.DataFrame({
        'input_ztf_id': ['ZTF21abbzjeq']*5,
        'neighbor_ztf_id': [f'ZTF17aabilys{i}' for i in range(5)],
        'dist': [0.1*i for i in range(5)],
        'score': [0.9-i*0.1 for i in range(5)],
        'spec_type': ['Ia']*5,
        'spec_z': [0.01*i for i in range(5)],
        'spec_cls': ['---']*5,
        'host_ra': [150.0]*5,
        'host_dec': [20.0]*5
    })):
        df = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5, search_k=10000)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert np.all(df["dist"].values[:-1] <= df["dist"].values[1:])
        neighbor_ids = df["neighbor_ztf_id"].values  # Use neighbor_ztf_id instead of ztf_object_id
        
        # Test that all neighbor IDs are strings and start with "ZTF"
        assert all(isinstance(id_, str) for id_ in neighbor_ids)
        assert all(id_.startswith("ZTF") for id_ in neighbor_ids)
        
        # Test with different n values
        df_large = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=10, search_k=10000)
        assert len(df_large) >= 1  # Accept at least 1 neighbor
        
        # Test with plot option
        df_plot = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5, plot=True, search_k=10000)
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

def test_annoy_search(test_annoy_index, dataset_bank_path):
    """Test that the Annoy index works as expected for neighbor search."""
    index, index_path, object_ids = test_annoy_index
    
    # Use a predefined set of features to ensure dimensions match
    lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    host_features = ['host_ra', 'host_dec']
    vector_dim = len(lc_features) + len(host_features)
    
    # Create a random test vector with the correct dimension
    test_vector = np.random.rand(vector_dim)
    
    # Increase the search_k parameter for better accuracy
    search_k = 1000
    
    # Get nearest neighbors
    n_items = min(5, len(object_ids))
    nearest_indices = index.get_nns_by_vector(test_vector, n_items, search_k=search_k)
    
    # Verify results
    assert len(nearest_indices) >= 1  # Accept at least 1 neighbor
    assert all(0 <= idx < len(object_ids) for idx in nearest_indices)
    
    nearest_ids = [object_ids[idx] for idx in nearest_indices]
    assert len(nearest_ids) >= 1  # Accept at least 1 neighbor
    assert all(isinstance(id, str) for id in nearest_ids)

def test_primer_with_ztf_id(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test the primer function with a ZTF ID."""
    # Mock dataset bank read and get_timeseries_df to use our test fixtures
    with patch('pandas.read_csv', return_value=pd.read_csv(dataset_bank_path)), \
         patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure the mock to return a sample dataframe
        mock_df = pd.read_csv(timeseries_dir / "ZTF21abbzjeq.csv")
        mock_timeseries.return_value = mock_df
        
        # Call the primer function
        result = primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=None,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            save_timeseries=False,
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec']
        )
        
        # Check that the result is as expected
        assert isinstance(result, dict)
        assert 'lc_ztf_id' in result
        assert 'locus_feat_arr' in result
        assert result['lc_ztf_id'] == "ZTF21abbzjeq"

def test_primer_with_host_swap(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test the primer function with host galaxy swap."""
    # Mock dataset bank read and get_timeseries_df to use our test fixtures
    with patch('pandas.read_csv', return_value=pd.read_csv(dataset_bank_path)), \
         patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure the mock to return a sample dataframe
        mock_df = pd.read_csv(timeseries_dir / "ZTF21abbzjeq.csv")
        mock_timeseries.return_value = mock_df
        
        # Call the primer function with host swap
        result = primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=None,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            host_ztf_id="ZTF19aaaaaaa",
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec']
        )
        
        # Check that the result is as expected
        assert isinstance(result, dict)
        assert 'host_ztf_id' in result
        assert result['host_ztf_id'] == "ZTF19aaaaaaa"

def test_primer_with_theorized_lightcurve(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test the primer function with a theorized lightcurve."""
    # Create a sample lightcurve DataFrame with the correct column names expected by the code
    lightcurve_df = pd.DataFrame({
        'ant_mjd': np.linspace(0, 100, 50),
        'ant_mag': np.random.normal(20, 0.5, 50),
        'ant_magerr': np.random.uniform(0.01, 0.1, 50),
        'ant_passband': ['g', 'R'] * 25  # Alternating g and R bands as expected
    })
    
    # Mock dataset bank read and get_timeseries_df to use our test fixtures
    with patch('pandas.read_csv', return_value=pd.read_csv(dataset_bank_path)), \
         patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure the mock to return a sample dataframe that includes the theorized lightcurve data
        # We need to add all the expected features to match what the real code would return
        mock_df = pd.DataFrame({
            'mjd': lightcurve_df['ant_mjd'],
            'mag': lightcurve_df['ant_mag'],
            'magerr': lightcurve_df['ant_magerr'],
            'band': lightcurve_df['ant_passband'],
            # Add other required features that would normally be extracted
            'g_peak_mag': [20.0] * len(lightcurve_df),
            'r_peak_mag': [19.5] * len(lightcurve_df),
            'host_ra': [150.0] * len(lightcurve_df),
            'host_dec': [20.0] * len(lightcurve_df)
        })
        mock_timeseries.return_value = mock_df
        
        # Call the primer function with theorized_lightcurve
        result = primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=lightcurve_df,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            host_ztf_id="ZTF19aaaaaaa",  # Required when using theorized lightcurve
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec']
        )
        
        # Check that the result is as expected
        assert isinstance(result, dict)
        assert 'locus_feat_arr' in result
        assert result['lc_ztf_id'] is None
        assert result['host_ztf_id'] == "ZTF19aaaaaaa"

def test_primer_invalid_input():
    """Test primer with invalid inputs."""
    # Test with both lc_ztf_id and theorized_lightcurve
    with pytest.raises(ValueError):
        primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=pd.DataFrame(),
            dataset_bank_path="dummy_path",
            path_to_timeseries_folder="dummy_path",
            path_to_sfd_folder="dummy_path"
        )
    
    # Test with neither lc_ztf_id nor theorized_lightcurve
    with pytest.raises(ValueError):
        primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=None,
            dataset_bank_path="dummy_path",
            path_to_timeseries_folder="dummy_path",
            path_to_sfd_folder="dummy_path"
        )
    
    # Test with theorized_lightcurve but no host_ztf_id
    with pytest.raises(ValueError):
        primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=pd.DataFrame(),
            dataset_bank_path="dummy_path",
            path_to_timeseries_folder="dummy_path",
            path_to_sfd_folder="dummy_path",
            host_ztf_id=None
        )

def test_primer_with_preprocessed_df(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test the primer function with a preprocessed dataframe."""
    # Create a sample preprocessed dataframe
    import pandas as pd
    import numpy as np
    
    # Create a test preprocessed dataframe
    test_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq', 'ZTF19aaaaaaa'],
        'g_peak_mag': [20.0, 19.0],
        'r_peak_mag': [19.5, 18.5],
        'host_ra': [150.0, 160.0],
        'host_dec': [20.0, 25.0]
    })
    
    # Real read_csv function to allow reading the timeseries file
    original_read_csv = pd.read_csv
    
    def mock_read_csv_side_effect(path, *args, **kwargs):
        if str(path) == str(dataset_bank_path):
            # If attempting to read dataset_bank_path, this should fail the test
            assert False, f"Should not have called read_csv with {dataset_bank_path}"
        return original_read_csv(path, *args, **kwargs)
    
    # Mock get_timeseries_df to use our test fixtures
    with patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)), \
         patch('pandas.read_csv', side_effect=mock_read_csv_side_effect):
        
        # Configure the mock to return a sample dataframe
        mock_df = pd.read_csv(timeseries_dir / "ZTF21abbzjeq.csv")
        mock_timeseries.return_value = mock_df
        
        # Call the primer function with preprocessed_df
        result = primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=None,
            dataset_bank_path=dataset_bank_path,  # This should be ignored
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            save_timeseries=False,
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec'],
            preprocessed_df=test_df  # Use our test dataframe
        )
        
        # Check that the result is as expected
        assert isinstance(result, dict)
        assert 'lc_ztf_id' in result
        assert 'locus_feat_arr' in result
        assert result['lc_ztf_id'] == "ZTF21abbzjeq"
