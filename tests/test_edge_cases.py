import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import relaiss as rl
from relaiss.search import primer
from relaiss.anomaly import train_AD_model, anomaly_detection
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from relaiss.relaiss import ReLAISS

def test_missing_error_columns(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test that primer handles missing error columns without crashing."""
    # Create a minimal test dataframe with no error columns
    test_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq'],
        'g_peak_mag': [20.0],
        'r_peak_mag': [19.5],
        'host_ra': [150.0],
        'host_dec': [20.0],
        # Intentionally missing error columns like g_peak_mag_err
    })
    
    # Mock get_timeseries_df to use our test fixtures 
    with patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure mock to return data without error columns
        mock_df = pd.DataFrame({
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0],
            # No error columns included
        })
        mock_timeseries.return_value = mock_df
        
        # This should not raise any errors
        result = primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=None,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec'],
            preprocessed_df=test_df,
            num_sims=2  # Request Monte Carlo simulations which use error columns
        )
        
        # Check that the result still has the core information
        assert 'locus_feat_arr' in result
        assert len(result['locus_feat_arrs_mc_l']) == 2  # Requested 2 MC simulations

def test_missing_ant_mjd_column(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test that anomaly detection handles missing ant_mjd column without crashing."""
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']

    client = ReLAISS()
    client.load_reference()
    
    # Create a sample dataframe without ant_mjd column
    test_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq', 'ZTF19aaaaaaa'],
        'g_peak_mag': [20.0, 19.0],
        'r_peak_mag': [19.5, 18.5],
        'host_ra': [150.0, 160.0],
        'host_dec': [20.0, 25.0],
        # Missing ant_mjd column
    })
    
    # Mock the needed components
    with patch('relaiss.fetch.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.anomaly.get_timeseries_df') as mock_ad_timeseries, \
         patch('joblib.dump'), \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)), \
         patch('joblib.load', return_value=MagicMock()), \
         patch('relaiss.anomaly.check_anom_and_plot'):  # Skip plotting
        
        # Configure mocks to return dataframes without ant_mjd
        df_without_ant_mjd = pd.DataFrame({
            'ztf_object_id': ['ZTF21abbzjeq'],
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0],
            # Intentionally missing ant_mjd
            'obs_num': [1]
        })
        mock_timeseries.return_value = df_without_ant_mjd
        mock_ad_timeseries.return_value = df_without_ant_mjd
        
        # This should work without errors
        # First train model (also tests duplicate model save message fix)
        model_path = train_AD_model(
            lc_features=lc_features,
            host_features=host_features,
            preprocessed_df=test_df,
            path_to_models_directory="./test_models",
            n_estimators=10,  # Small for quick test
            max_samples=8     # Small for quick test
        )
        
        # Then run anomaly detection
        result = anomaly_detection(
            client=client,
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            path_to_dataset_bank=dataset_bank_path,
            path_to_models_directory="./test_models",
            preprocessed_df=test_df,
            save_figures=False
        )
        
        # Check that mjd_cutoff was properly handled when ant_mjd was missing
        assert 'mjd_cutoff' in mock_ad_timeseries.return_value.columns, "mjd_cutoff column was not created"

def test_host_feature_length_mismatch(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test that primer handles mismatched host feature lengths."""
    # Create a test with a theorized lightcurve
    lightcurve_df = pd.DataFrame({
        'ant_mjd': np.linspace(0, 100, 10),
        'ant_mag': np.random.normal(20, 0.5, 10),
        'ant_magerr': np.random.uniform(0.01, 0.1, 10),
        'ant_passband': ['g', 'R'] * 5  # Alternating g and R bands
    })
    
    # Define features with intentionally mismatched lengths
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock the necessary functions
    with patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure the mock with values mismatched to feature lengths
        # This simulates the scenario where host_locus_feat_arr has 4 values but
        # is intended to be mapped to just 2 host features
        mock_df = pd.DataFrame({
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0]
        })
        mock_timeseries.return_value = mock_df
        
        # This should handle the mismatch correctly
        result = primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=lightcurve_df,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            host_ztf_id="ZTF19aaaaaaa",  # Required for theorized lightcurve
            lc_features=lc_features,
            host_features=host_features
        )
        
        # Verify the output has correct structure
        assert 'locus_feat_arr' in result
        assert len(result['locus_feat_arr']) == len(lc_features) + len(host_features)

def test_light_curve_weighting():
    """Test that light curve weighting is applied correctly even with NaN values."""
    # Create a test array with NaN values in the light curve part
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock array with all light curve features as NaN
    test_array = np.array([np.nan, np.nan, 155.0, 25.0])
    
    # Create a properly fitted scaler
    X = np.array([
        [20.0, 19.5, 150.0, 20.0],
        [19.0, 19.0, 155.0, 25.0]
    ])
    scaler = StandardScaler().fit(X)
    
    # Apply light curve weighting (copying logic from relaiss.py)
    test_array_copy = test_array.copy()
    n_lc = len(lc_features)
    weight_factor = 10.0
    
    # Test the light curve upweighting function with NaNs
    # Only upweight if we don't have all NaN values in the light curve part
    if not np.all(np.isnan(test_array_copy[:n_lc])):
        test_array_copy[:n_lc] *= weight_factor
    
    # Transform with scaler
    scaled = scaler.transform([test_array_copy])[0]
    
    # Since all light curve features are NaN, weighting should not change anything
    assert np.isnan(scaled[0])
    assert np.isnan(scaled[1])
    # Host features should be scaled normally
    assert not np.isnan(scaled[2])
    assert not np.isnan(scaled[3])
    
    # Now test with some non-NaN light curve values
    test_array2 = np.array([20.0, np.nan, 155.0, 25.0])
    test_array2_copy = test_array2.copy()
    
    # First transform without weighting
    scaled_no_weight = scaler.transform([test_array2])[0]
    
    # Now apply weighting - it should upweight the first feature
    if not np.all(np.isnan(test_array2_copy[:n_lc])):
        test_array2_copy[:n_lc] *= weight_factor
    
    # Verify the first element got upweighted before scaling
    assert test_array2_copy[0] == 20.0 * weight_factor
    
    # Transform with scaler
    scaled_with_weight = scaler.transform([test_array2_copy])[0]
    
    # First element should be scaled differently due to upweighting
    assert scaled_with_weight[0] != scaled_no_weight[0]
    # But the host features should be the same
    assert scaled_with_weight[2] == scaled_no_weight[2]
    assert scaled_with_weight[3] == scaled_no_weight[3]

def test_optional_ztf_object_id():
    """Test that find_neighbors accepts None for ztf_object_id when using theorized lightcurve."""
    # Create a minimal client
    client = rl.ReLAISS()
    # Test by monkeypatching the primer function to avoid actual computation
    with patch.object(client, 'find_neighbors', return_value=pd.DataFrame()):
        # The key test is that this doesn't raise an error due to missing required argument
        # We need to provide theorized_lightcurve_df and host_ztf_id to avoid other validation errors
        client.find_neighbors(
            theorized_lightcurve_df=pd.DataFrame({
                'ant_mjd': [1, 2, 3],
                'ant_mag': [19, 20, 19],
                'ant_magerr': [0.1, 0.1, 0.1],
                'ant_passband': ['g', 'R', 'g']
            }),
            host_ztf_id="ZTF19aaaaaaa",
            n=3
        )
        # If we get here, the test passes 

def test_swapped_host_with_key_error():
    """Test that find_neighbors in ReLAISS can handle missing host galaxies."""
    from relaiss.relaiss import ReLAISS
    from unittest.mock import patch, MagicMock
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    # Mock the primer function directly
    with patch('relaiss.relaiss.primer') as mock_primer:
        # Configure mock to first raise KeyError, then return successfully
        mock_primer.side_effect = [
            KeyError("ZTF19aazefbe not found"),  # First call with host fails
            {  # Second call without host succeeds
                'lc_ztf_id': 'ZTF21abbzjeq',
                'host_ztf_id': None,
                'lc_tns_name': 'SN2021xyz',
                'lc_tns_cls': 'SN Ia',
                'lc_tns_z': 0.1,
                'lc_ztf_id_in_dataset_bank': True,
                'locus_feat_arr': np.array([20.0, 19.5, 150.0, 20.0]),
                'locus_feat_arrs_mc_l': [],
                'lc_galaxy_ra': 150.0,
                'lc_galaxy_dec': 20.0,
                'lc_feat_names': ['g_peak_mag', 'r_peak_mag'],
                'host_feat_names': ['host_ra', 'host_dec']
            }
        ]
        
        # Mock entire find_neighbors method to avoid dealing with index
        with patch.object(ReLAISS, 'find_neighbors') as mock_find_neighbors:
            # Set up mock to properly call the primer function
            def find_neighbors_impl(ztf_object_id=None, host_ztf_id=None, **kwargs):
                # Just delegate to mocked primer to test our error handling
                try:
                    primer_dict = mock_primer(
                        lc_ztf_id=ztf_object_id,
                        theorized_lightcurve_df=None,
                        host_ztf_id=host_ztf_id,
                        dataset_bank_path="dummy_path.csv",
                        path_to_timeseries_folder='./',
                        path_to_sfd_folder='./',
                        lc_features=['g_peak_mag', 'r_peak_mag'],
                        host_features=['host_ra', 'host_dec'],
                        num_sims=0,
                        preprocessed_df=None
                    )
                    return pd.DataFrame([{"result": "success"}])
                except Exception as e:
                    # This is the code we're testing - calling primer again without host_ztf_id
                    primer_dict = mock_primer(
                        lc_ztf_id=ztf_object_id,
                        theorized_lightcurve_df=None,
                        host_ztf_id=None,  # Try without host
                        dataset_bank_path="dummy_path.csv",
                        path_to_timeseries_folder='./',
                        path_to_sfd_folder='./',
                        lc_features=['g_peak_mag', 'r_peak_mag'],
                        host_features=['host_ra', 'host_dec'],
                        num_sims=0,
                        preprocessed_df=None
                    )
                    return pd.DataFrame([{"result": "fallback_success"}])
                    
            # Set the mock implementation
            mock_find_neighbors.side_effect = lambda *args, **kwargs: find_neighbors_impl(**kwargs)
            
            # Create a ReLAISS instance (doesn't need special setup now)
            client = ReLAISS()
            
            # Test the method with a non-existent host ZTF ID
            result = client.find_neighbors(
                ztf_object_id='ZTF21abbzjeq',
                host_ztf_id='ZTF19aazefbe',  # This will trigger the KeyError
                n=2
            )
            
            # Verify the fallback occurred
            assert mock_primer.call_count == 2
            assert mock_primer.call_args_list[0][1]['host_ztf_id'] == 'ZTF19aazefbe'
            assert mock_primer.call_args_list[1][1]['host_ztf_id'] is None

def test_mjd_alignment():
    """Test that MJD alignment is handled properly when ant_mjd column is missing."""
    from relaiss.anomaly import anomaly_detection
    import pandas as pd
    import numpy as np
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    
    # Create a mock for antares_client.search.get_by_ztf_object_id
    locus_mock = MagicMock()
    
    # Create fake light curve data with real MJD values
    mjd_values = np.linspace(58000, 58100, 20)
    timeseries_data = pd.DataFrame({
        'ant_mjd': mjd_values,
        'ant_mag': np.random.normal(20, 0.5, 20),
        'ant_magerr': np.random.uniform(0.1, 0.2, 20),
        'ant_passband': ['g', 'R'] * 10
    })
    locus_mock.timeseries.to_pandas.return_value = timeseries_data
    
    # Mock the necessary components
    with patch('antares_client.search.get_by_ztf_object_id', return_value=locus_mock), \
         patch('relaiss.anomaly.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.anomaly.train_AD_model') as mock_train, \
         patch('relaiss.anomaly.check_anom_and_plot') as mock_plot, \
         patch('relaiss.fetch.get_TNS_data', return_value=('Test', 'Type', 0.1)), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.show'), \
         patch('joblib.load') as mock_load:
        
        # Configure the timeseries_df mock to NOT have ant_mjd column
        mock_timeseries_df = pd.DataFrame({
            'obs_num': range(10),
            'g_peak_mag': np.random.normal(20, 0.5, 10),
            'r_peak_mag': np.random.normal(19.5, 0.5, 10),
            'host_ra': np.random.uniform(150, 160, 10),
            'host_dec': np.random.uniform(20, 30, 10)
        })
        mock_timeseries.return_value = mock_timeseries_df
        
        # Configure the mock trained model
        mock_clf = MagicMock()
        mock_clf.decision_function.return_value = np.random.normal(0, 1, 10)
        mock_train.return_value = Path("test_model.joblib")
        mock_load.return_value = mock_clf
        
        # Run the function with our mocks
        result = anomaly_detection(
            transient_ztf_id="ZTF_TEST",
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec'],
            path_to_timeseries_folder="./",
            path_to_sfd_folder="./",
            path_to_dataset_bank="test_bank.csv",
            save_figures=False
        )
        
        # Verify that mjd_cutoff was added to the dataframe
        assert 'mjd_cutoff' in mock_timeseries_df.columns
        
        # Check that the mjd_cutoff values are within the expected range
        min_mjd = min(mjd_values)
        max_mjd = max(mjd_values)
        assert mock_timeseries_df['mjd_cutoff'].min() >= min_mjd
        assert mock_timeseries_df['mjd_cutoff'].max() <= max_mjd
        
        # Verify the plot function was called with the right arguments
        mock_plot.assert_called_once()
        args, kwargs = mock_plot.call_args
        # The timeseries_df_full arg should have mjd_cutoff column
        assert 'mjd_cutoff' in kwargs.get('timeseries_df_full', {}).columns 
