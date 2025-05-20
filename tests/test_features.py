import pytest
import pandas as pd
import numpy as np
import relaiss as rl
from relaiss.features import (
    build_dataset_bank,
    create_features_dict,
    extract_lc_and_host_features,
    SupernovaFeatureExtractor
)

def test_build_dataset_bank():
    # Create sample data
    raw_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq'],
        'mag': [20.0],
        'color': [0.5],
        'period': [5.0],
        'ra': [150.0],
        'dec': [20.0],
        'gKronMag': [21.0],
        'rKronMag': [20.5],
        'iKronMag': [20.0],
        'zKronMag': [19.5],
        'gKronMagErr': [0.1],
        'rKronMagErr': [0.1],
        'iKronMagErr': [0.1],
        'zKronMagErr': [0.1]
    })
    
    result = build_dataset_bank(
        raw_df_bank=raw_df,
        path_to_sfd_folder="../data/sfd",
        theorized=False,
        path_to_dataset_bank="../data/dataset_bank.csv"
    )
    
    assert isinstance(result, pd.DataFrame)
    assert 'gminusrKronMag' in result.columns
    assert 'rminusiKronMag' in result.columns
    assert 'iminuszKronMag' in result.columns

def test_create_features_dict():
    lc_features = ['mag', 'color', 'period']
    host_features = ['redshift', 'mass']
    
    result = create_features_dict(
        lc_feature_names=lc_features,
        host_feature_names=host_features,
        lc_groups=2,
        host_groups=2
    )
    
    assert isinstance(result, dict)
    assert 'lc_features' in result
    assert 'host_features' in result
    assert len(result['lc_features']) == len(lc_features)
    assert len(result['host_features']) == len(host_features)

def test_extract_lc_and_host_features():
    result = extract_lc_and_host_features(
        ztf_id="ZTF21abbzjeq",
        path_to_timeseries_folder="../data/timeseries",
        path_to_sfd_folder="../data/sfd",
        path_to_dataset_bank="../data/dataset_bank.csv",
        show_lc=False,
        show_host=False
    )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'mag' in result.columns

def test_supernova_feature_extractor():
    # Create sample light curve data
    time_g = np.linspace(0, 100, 50)
    mag_g = np.random.normal(20, 0.5, 50)
    err_g = np.random.uniform(0.01, 0.1, 50)
    time_r = np.linspace(0, 100, 50)
    mag_r = np.random.normal(19, 0.5, 50)
    err_r = np.random.uniform(0.01, 0.1, 50)
    
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
    assert isinstance(features, dict)
    assert 'g_mean_mag' in features
    assert 'r_mean_mag' in features
    
    # Test with uncertainty estimation
    features, uncertainties = extractor.extract_features(return_uncertainty=True, n_trials=5)
    assert isinstance(features, dict)
    assert isinstance(uncertainties, dict)
    assert len(features) == len(uncertainties)

def test_feature_extraction_invalid_input():
    # Test with invalid data
    with pytest.raises(ValueError):
        extractor = SupernovaFeatureExtractor(
            time_g=[],  # Empty data
            mag_g=[],
            err_g=[],
            time_r=[1, 2, 3],
            mag_r=[19, 19.5, 20],
            err_r=[0.1, 0.1, 0.1]
        )
    
    # Test with mismatched array lengths
    with pytest.raises(ValueError):
        extractor = SupernovaFeatureExtractor(
            time_g=[1, 2],
            mag_g=[19],  # Different length
            err_g=[0.1, 0.1],
            time_r=[1, 2],
            mag_r=[19, 19.5],
            err_r=[0.1, 0.1]
        ) 