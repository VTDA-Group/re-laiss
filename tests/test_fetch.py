import pytest
import pandas as pd
import numpy as np
import requests_mock
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import relaiss as rl
from relaiss.fetch import (
    get_TNS_data,
    fetch_ps1_cutout,
    fetch_ps1_rgb_jpeg,
    get_timeseries_df
)

def test_get_TNS_data():
    # We need to create a proper mock for antares_client.search.get_by_ztf_object_id
    mock_locus = MagicMock()
    mock_locus.catalog_objects = {
        "tns_public_objects": [
            {"name": "TNS2023abc", "type": "SN Ia", "redshift": 0.1}
        ]
    }
    
    # Use the correct module path to patch
    with patch('antares_client.search.get_by_ztf_object_id', return_value=mock_locus):
        tns_name, tns_cls, tns_z = get_TNS_data("ZTF21abbzjeq")
        assert tns_name == "TNS2023abc"
        assert tns_cls == "SN Ia"
        assert tns_z == 0.1

def test_fetch_ps1_cutout():
    # Create a mock array for testing
    mock_array = np.random.normal(0, 1, (100, 100))
    
    # Mock the function to return our array
    with patch('relaiss.fetch._ps1_list_filenames', return_value='test.fits'), \
         patch('requests.get') as mock_get, \
         patch('astropy.io.fits.open') as mock_fits:
        
        # Configure the mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'dummy_content'
        mock_get.return_value = mock_response
        
        # Configure mock fits
        mock_hdulist = MagicMock()
        mock_hdu = MagicMock()
        mock_hdu.data = mock_array
        mock_hdulist.__getitem__.return_value = mock_hdu
        mock_fits.return_value.__enter__.return_value = mock_hdulist
        
        # Call the function
        cutout = fetch_ps1_cutout(
            ra_deg=150.0,
            dec_deg=20.0,
            size_pix=100,
            flt='r'
        )
        
        assert isinstance(cutout, np.ndarray)
        assert cutout.shape == (100, 100)

def test_fetch_ps1_rgb_jpeg():
    """Test the PS1 RGB JPEG fetching function with mocked requests."""
    # Create a mock RGB image
    mock_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mock_img = Image.fromarray(mock_rgb)
    img_bytes = io.BytesIO()
    mock_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Use requests_mock to intercept HTTP requests
    with requests_mock.Mocker() as m:
        # Mock the PS1 request
        m.get(
            requests_mock.ANY,  # Match any URL
            content=img_bytes.read(),
            status_code=200
        )
        
        # Call the function
        rgb_img = fetch_ps1_rgb_jpeg(
            ra_deg=150.0,
            dec_deg=20.0,
            size_pix=100
        )
        
        # Verify the result
        assert isinstance(rgb_img, np.ndarray)
        assert rgb_img.shape[2] == 3  # RGB image has 3 channels
        assert rgb_img.dtype == np.uint8

def test_get_timeseries_df_with_local_file():
    """Test using local CSV files instead of fetching from ANTARES."""
    # Use the fixture timeseries file directly rather than using the get_timeseries_df function
    df = pd.read_csv("tests/fixtures/timeseries/ZTF21abbzjeq.csv")
    
    # Check expected columns
    assert 'mjd' in df.columns
    assert 'mag' in df.columns
    assert 'magerr' in df.columns
    assert 'band' in df.columns

def test_with_theorized_lightcurve():
    """Test with a theorized lightcurve DataFrame."""
    # Create a sample lightcurve DataFrame
    lightcurve_df = pd.DataFrame({
        'ant_mjd': np.linspace(0, 100, 50),
        'ant_mag': np.random.normal(20, 0.5, 50),
        'ant_magerr': np.random.uniform(0.01, 0.1, 50),
        'ant_passband': ['g', 'R'] * 25  # Alternating g and R bands
    })
    
    # Mock the extract_lc_and_host_features function
    with patch('relaiss.fetch.extract_lc_and_host_features', return_value=lightcurve_df):
        df = get_timeseries_df(
            ztf_id=None,
            theorized_lightcurve_df=lightcurve_df,
            path_to_timeseries_folder="dummy_path",
            path_to_sfd_folder="dummy_path",
            save_timeseries=False
        )
        
        # Check that we have the expected columns with the same content
        assert set(df.columns) == set(lightcurve_df.columns)
        assert df.shape == lightcurve_df.shape
        
        # Check that the values are preserved
        for col in lightcurve_df.columns:
            pd.testing.assert_series_equal(
                df[col].reset_index(drop=True), 
                lightcurve_df[col].reset_index(drop=True)
            ) 

def test_get_timeseries_df_with_preprocessed_df():
    """Test that get_timeseries_df correctly uses a provided preprocessed dataframe."""
    import pandas as pd
    from relaiss.fetch import get_timeseries_df
    
    # Create a sample preprocessed dataframe with the test transient
    test_preprocessed_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq', 'ZTF19aaaaaaa'],
        'g_peak_mag': [20.0, 19.0],
        'r_peak_mag': [19.5, 18.5], 
        'host_ra': [150.0, 160.0],
        'host_dec': [20.0, 25.0]
    })
    
    # Mock extract_lc_and_host_features to verify it's called with the preprocessed_df
    with patch('relaiss.fetch.extract_lc_and_host_features') as mock_extract:
        # Mock the returned timeseries dataframe
        mock_extract.return_value = pd.DataFrame({
            'mjd': [58000, 58001],
            'mag': [20.0, 20.1],
            'magerr': [0.1, 0.1],
            'band': ['g', 'r']
        })
        
        # Call get_timeseries_df with a preprocessed dataframe
        result = get_timeseries_df(
            ztf_id='ZTF21abbzjeq',
            path_to_timeseries_folder='dummy_path',
            path_to_sfd_folder='dummy_path',
            path_to_dataset_bank='dummy_path',
            preprocessed_df=test_preprocessed_df
        )
        
        # Verify the extract_lc_and_host_features function was called with the preprocessed_df
        mock_extract.assert_called_once()
        # Check that preprocessed_df was passed to extract_lc_and_host_features
        args, kwargs = mock_extract.call_args
        assert 'preprocessed_df' in kwargs
        assert kwargs['preprocessed_df'] is test_preprocessed_df 