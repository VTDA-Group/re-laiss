import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import astropy.units as u
import relaiss as rl
from relaiss.features import SupernovaFeatureExtractor

def test_supernova_feature_extractor_mock():
    """Test that SupernovaFeatureExtractor works with mocked dust extinction."""
    np.random.seed(42)
    time_g = np.linspace(0, 100, 50)
    mag_g = np.random.normal(20, 0.5, 50)
    err_g = np.random.uniform(0.01, 0.1, 50)
    time_r = np.linspace(0, 100, 50)
    mag_r = np.random.normal(19, 0.5, 50)
    err_r = np.random.uniform(0.01, 0.1, 50)
    
    with patch('sfdmap2.sfdmap.SFDMap') as mock_map, \
         patch('dust_extinction.parameter_averages.G23') as mock_g23, \
         patch('astropy.units.um', u.um), \
         patch('sklearn.cluster.DBSCAN') as mock_dbscan:
        
        mock_sfd = MagicMock()
        mock_sfd.ebv.return_value = 0.05
        mock_map.return_value = mock_sfd
        
        mock_ext_model = MagicMock()
        mock_ext_model.extinguish.return_value = 0.9
        mock_g23.return_value = mock_ext_model
        
        mock_cluster = MagicMock()
        mock_cluster.labels_ = np.zeros(50)
        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.fit.return_value = mock_cluster
        mock_dbscan.return_value = mock_dbscan_instance
        
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
        
        features = extractor.extract_features()
        
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == 1
        assert 'g_peak_mag' in features.columns
        assert 'r_peak_mag' in features.columns
        
        features_with_err = extractor.extract_features(
            return_uncertainty=True, 
            n_trials=2
        )
        
        assert isinstance(features_with_err, pd.DataFrame)
        assert features_with_err.shape[0] == 1
        assert any(col.endswith('_err') for col in features_with_err.columns) 