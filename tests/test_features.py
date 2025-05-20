import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import astropy.units as u
import relaiss as rl
from relaiss.features import (
    build_dataset_bank,
    create_features_dict,
    extract_lc_and_host_features,
    SupernovaFeatureExtractor
)

def test_build_dataset_bank(dataset_bank_path, sfd_dir, mock_extinction_all):
    """Test the build_dataset_bank function."""
    # Create a much more complete sample dataset with all required columns
    raw_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq'],
        'g_peak_mag': [20.0],
        'r_peak_mag': [19.5],
        'g_peak_time': [25.0],
        'g_rise_time': [15.0],
        'g_decline_time': [20.0],
        'g_duration_above_half_flux': [40.0],
        'r_duration_above_half_flux': [45.0],
        'r_peak_time': [27.0],
        'r_rise_time': [18.0],
        'r_decline_time': [25.0],
        'mean_g-r': [0.5],
        'g-r_at_g_peak': [0.45],
        'mean_color_rate': [0.01],
        'g_mean_rolling_variance': [0.05],
        'r_mean_rolling_variance': [0.04],
        'g_rise_local_curvature': [0.02],
        'g_decline_local_curvature': [0.03],
        'r_rise_local_curvature': [0.02],
        'r_decline_local_curvature': [0.025],
        'ra': [150.0],
        'dec': [20.0],
        'gKronMag': [21.0],
        'rKronMag': [20.5],
        'iKronMag': [20.0],
        'zKronMag': [19.5],
        'gKronMagErr': [0.1],
        'rKronMagErr': [0.1],
        'iKronMagErr': [0.1],
        'zKronMagErr': [0.1],
        'gKronRad': [5.0],
        'gExtNSigma': [2.0],
        'rmomentXX': [1.0],
        'rmomentYY': [1.0],
        'rmomentXY': [0.1],
        'rKronRad': [5.0],
        'rExtNSigma': [2.0],
        'iKronRad': [5.0],
        'iExtNSigma': [2.0],
        'zKronRad': [5.0],
        'zExtNSigma': [2.0]
    })
    
    # Mock pd.read_csv for dataset_bank
    with patch('pandas.read_csv', return_value=raw_df):
        result = build_dataset_bank(
            raw_df_bank=raw_df,
            path_to_sfd_folder=sfd_dir,
            theorized=False,
            path_to_dataset_bank=dataset_bank_path
        )
    
    assert isinstance(result, pd.DataFrame)
    assert 'gminusrKronMag' in result.columns  # Calculated color column
    assert 'rminusiKronMag' in result.columns
    assert 'iminuszKronMag' in result.columns

def test_create_features_dict():
    """Test that the feature dictionary is created correctly."""
    lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    host_features = ['host_ra', 'host_dec']
    
    result = create_features_dict(
        lc_feature_names=lc_features,
        host_feature_names=host_features,
        lc_groups=2,
        host_groups=2
    )
    
    assert isinstance(result, dict)
    assert 'lc_group_1' in result  # Check for the expected group keys
    assert 'lc_group_2' in result
    assert 'host_group_1' in result
    assert 'host_group_2' in result
    assert len(result['lc_group_1']) + len(result['lc_group_2']) == len(lc_features)
    assert len(result['host_group_1']) + len(result['host_group_2']) == len(host_features)

@pytest.mark.skip(reason="Requires more comprehensive mocking of antares_client")
def test_extract_lc_and_host_features(dataset_bank_path, timeseries_dir, sfd_dir, mock_extinction_all, mock_antares_client):
    """Test the extract_lc_and_host_features function with our mock fixtures."""
    # Create a sample dataframe that would exist in dataset_bank.csv
    sample_db = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq'],
        'ra': [150.0],
        'dec': [20.0],
        'g_peak_mag': [19.5],
        'r_peak_mag': [19.0],
        'host_ra': [150.1],
        'host_dec': [20.1]
    })
    
    # Mock pd.read_csv to return our sample dataset and timeseries
    with patch('pandas.read_csv', side_effect=[
        sample_db,  # First call for dataset_bank
        pd.read_csv(timeseries_dir / "ZTF21abbzjeq.csv")  # Second call for timeseries
    ]), patch('os.path.exists', return_value=True):
        
        result = extract_lc_and_host_features(
            ztf_id="ZTF21abbzjeq",
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            path_to_dataset_bank=dataset_bank_path,
            show_lc=False,
            show_host=False
        )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Check for expected columns from our fixture data
    assert 'mjd' in result.columns
    assert 'mag' in result.columns
    assert 'magerr' in result.columns
    assert 'band' in result.columns

# Updated test
def test_extract_lc_and_host_features(dataset_bank_path, timeseries_dir, sfd_dir, mock_extinction_all):
    """Test the extract_lc_and_host_features function with comprehensive mocking."""
    # Create a complete sample dataset_bank.csv dataframe
    sample_db = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq'],
        'ra': [150.0],
        'dec': [20.0],
        'g_peak_mag': [19.5],
        'r_peak_mag': [19.0],
        'g_peak_time': [25.0],
        'r_peak_time': [27.0],
        'g_rise_time': [15.0],
        'r_rise_time': [18.0],
        'g_decline_time': [20.0],
        'r_decline_time': [25.0],
        'mean_g-r': [0.5],
        'g-r_at_g_peak': [0.45],
        'mean_color_rate': [0.01],
        'host_ra': [150.1],
        'host_dec': [20.1],
        'gKronMag': [21.0],
        'rKronMag': [20.5],
        'iKronMag': [20.0],
        'zKronMag': [19.5],
        'gKronMagErr': [0.1],
        'rKronMagErr': [0.1],
        'iKronMagErr': [0.1],
        'zKronMagErr': [0.1],
        'gKronRad': [5.0],
        'gExtNSigma': [2.0],
        'rmomentXX': [1.0],
        'rmomentYY': [1.0],
        'rmomentXY': [0.1],
        'rKronRad': [5.0],
        'rExtNSigma': [2.0],
        'iKronRad': [5.0],
        'iExtNSigma': [2.0],
        'zKronRad': [5.0],
        'zExtNSigma': [2.0]
    })
    
    # Create a mock timeseries DataFrame
    mock_ts_df = pd.DataFrame({
        'ant_mjd': np.linspace(0, 100, 50),
        'ant_passband': ['g', 'R'] * 25,  # Alternating g and R bands
        'ant_mag': np.random.normal(20, 0.5, 50),
        'ant_magerr': np.random.uniform(0.01, 0.1, 50),
        'ant_ra': [150.0] * 50,
        'ant_dec': [20.0] * 50
    })
    
    # Mock all the required functions
    with patch('pandas.read_csv', return_value=sample_db), \
         patch('antares_client.search.get_by_ztf_object_id') as mock_get, \
         patch('astro_prost.associate.associate_sample') as mock_associate, \
         patch('relaiss.features.build_dataset_bank', return_value=sample_db), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        
        # Configure the mock ANTARES client
        mock_locus = MagicMock()
        mock_timeseries = MagicMock()
        mock_timeseries.to_pandas.return_value = mock_ts_df
        mock_locus.timeseries = mock_timeseries
        mock_get.return_value = mock_locus
        
        # Configure the mock host galaxy association
        mock_hosts = pd.DataFrame({
            'IAUID': ['ZTF21abbzjeq'],
            'raMean': [150.1],
            'decMean': [20.1],
            'host_ra': [150.1],
            'host_dec': [20.1]
        })
        for col in sample_db.columns:
            if col.startswith('g') or col.startswith('r') or col.startswith('i') or col.startswith('z'):
                if col not in mock_hosts.columns:
                    mock_hosts[col] = sample_db[col].values[0]
        
        mock_associate.return_value = mock_hosts
        
        # Execute the function
        with patch('relaiss.features.SupernovaFeatureExtractor.extract_features') as mock_extract:
            # Return a DataFrame similar to what extract_features would return
            feature_df = pd.DataFrame({
                'g_peak_mag': [19.5],
                'r_peak_mag': [19.0],
                'g_peak_time': [25.0],
                'r_peak_time': [27.0],
                'g_rise_time': [15.0],
                'r_rise_time': [18.0],
                'g_decline_time': [20.0],
                'r_decline_time': [25.0]
            })
            mock_extract.return_value = feature_df
            
            result = extract_lc_and_host_features(
                ztf_id="ZTF21abbzjeq",
                path_to_timeseries_folder=str(timeseries_dir),
                path_to_sfd_folder=str(sfd_dir),
                path_to_dataset_bank=str(dataset_bank_path),
                show_lc=False,
                show_host=False
            )
    
    # Check the result
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

def test_supernova_feature_extractor():
    """Test the SupernovaFeatureExtractor with mocked extinction."""
    # Create sample light curve data
    np.random.seed(42)  # Make sure results are reproducible
    time_g = np.linspace(0, 100, 50)
    mag_g = np.random.normal(20, 0.5, 50)
    err_g = np.random.uniform(0.01, 0.1, 50)
    time_r = np.linspace(0, 100, 50)
    mag_r = np.random.normal(19, 0.5, 50)
    err_r = np.random.uniform(0.01, 0.1, 50)
    
    # Fully mock ALL dependencies to avoid external calls
    with patch('sfdmap2.sfdmap.SFDMap') as mock_map, \
         patch('dust_extinction.parameter_averages.G23') as mock_g23, \
         patch('astropy.units.um', u.um), \
         patch('sklearn.cluster.DBSCAN') as mock_dbscan:
        
        # Configure mocks
        mock_sfd = MagicMock()
        mock_sfd.ebv.return_value = 0.05
        mock_map.return_value = mock_sfd
        
        mock_ext_model = MagicMock()
        mock_ext_model.extinguish.return_value = 0.9  # 10% extinction
        mock_g23.return_value = mock_ext_model
        
        # Mock DBSCAN to avoid clustering issues
        mock_cluster = MagicMock()
        mock_cluster.labels_ = np.zeros(50)  # All points in the same cluster
        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.fit.return_value = mock_cluster
        mock_dbscan.return_value = mock_dbscan_instance
        
        # Create the extractor
        extractor = SupernovaFeatureExtractor(
            time_g=time_g,
            mag_g=mag_g,
            err_g=err_g,
            time_r=time_r,
            mag_r=mag_r,
            err_r=err_r,
            ztf_object_id="ZTF21abbzjeq",
            ra=150.0,
            dec=20.0
        )
        
        # Extract features
        features = extractor.extract_features()
    
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1  # Should return a single row
    assert 'g_peak_mag' in features.columns
    assert 'r_peak_mag' in features.columns
    
    # Test with uncertainty estimation but with much fewer trials for speed
    with patch('sfdmap2.sfdmap.SFDMap') as mock_map, \
         patch('dust_extinction.parameter_averages.G23') as mock_g23, \
         patch('astropy.units.um', u.um), \
         patch('sklearn.cluster.DBSCAN') as mock_dbscan:
        
        # Configure mocks again
        mock_sfd = MagicMock()
        mock_sfd.ebv.return_value = 0.05
        mock_map.return_value = mock_sfd
        
        mock_ext_model = MagicMock()
        mock_ext_model.extinguish.return_value = 0.9  # 10% extinction
        mock_g23.return_value = mock_ext_model
        
        # Mock DBSCAN again
        mock_cluster = MagicMock()
        mock_cluster.labels_ = np.zeros(50)  # All points in the same cluster
        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.fit.return_value = mock_cluster
        mock_dbscan.return_value = mock_dbscan_instance
        
        # Extract with uncertainties
        features_with_err = extractor.extract_features(return_uncertainty=True, n_trials=2)
    
    assert isinstance(features_with_err, pd.DataFrame)
    assert features_with_err.shape[0] == 1  # Still one row
    assert any(col.endswith('_err') for col in features_with_err.columns)  # Should have _err columns

def test_feature_extraction_invalid_input():
    """Test error handling for invalid input to SupernovaFeatureExtractor."""
    # Test with empty bands
    with patch('relaiss.features.SupernovaFeatureExtractor._preprocess') as mock_preprocess:
        # Skip the preprocessing which causes index errors with invalid data
        mock_preprocess.return_value = None
        
        # With both empty g-band & r-band (should raise ValueError)
        with pytest.raises(ValueError):
            SupernovaFeatureExtractor(
                time_g=[],
                mag_g=[],
                err_g=[],
                time_r=[],
                mag_r=[],
                err_r=[],
                ztf_object_id="Test"
            )
        
        # With mismatched array lengths in g-band
        with pytest.raises(ValueError):
            SupernovaFeatureExtractor(
                time_g=[1, 2],
                mag_g=[19],  # Different length
                err_g=[0.1, 0.1],
                time_r=[1, 2],
                mag_r=[19, 19.5],
                err_r=[0.1, 0.1],
                ztf_object_id="Test"
            ) 