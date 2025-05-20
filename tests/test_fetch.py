import pytest
import pandas as pd
import numpy as np
import relaiss as rl
from relaiss.fetch import (
    get_TNS_data,
    fetch_ps1_cutout,
    fetch_ps1_rgb_jpeg,
    get_timeseries_df
)

def test_get_TNS_data():
    # Test with valid ZTF ID
    tns_name, tns_cls, tns_z = get_TNS_data("ZTF21abbzjeq")
    assert isinstance(tns_name, str)
    assert isinstance(tns_cls, str)
    assert isinstance(tns_z, (float, int))
    
    # Test with invalid ZTF ID
    tns_name, tns_cls, tns_z = get_TNS_data("invalid_id")
    assert tns_name == "No TNS"
    assert tns_cls == "---"
    assert tns_z == -99

def test_fetch_ps1_cutout():
    # Test with valid coordinates
    cutout = fetch_ps1_cutout(
        ra_deg=150.0,
        dec_deg=20.0,
        size_pix=100,
        flt='r'
    )
    assert isinstance(cutout, np.ndarray)
    assert cutout.shape == (100, 100)
    
    # Test with invalid coordinates
    with pytest.raises(RuntimeError):
        fetch_ps1_cutout(
            ra_deg=1000.0,  # Invalid RA
            dec_deg=20.0,
            size_pix=100,
            flt='r'
        )

def test_fetch_ps1_rgb_jpeg():
    # Test with valid coordinates
    rgb_img = fetch_ps1_rgb_jpeg(
        ra_deg=150.0,
        dec_deg=20.0,
        size_pix=100
    )
    assert isinstance(rgb_img, np.ndarray)
    assert rgb_img.shape == (100, 100, 3)
    assert rgb_img.dtype == np.uint8
    
    # Test with invalid coordinates
    with pytest.raises(RuntimeError):
        fetch_ps1_rgb_jpeg(
            ra_deg=1000.0,  # Invalid RA
            dec_deg=20.0,
            size_pix=100
        )

def test_get_timeseries_df():
    # Test with valid ZTF ID
    df = get_timeseries_df(
        ztf_id="ZTF21abbzjeq",
        path_to_timeseries_folder="../data/timeseries",
        path_to_sfd_folder="../data/sfd",
        save_timeseries=False
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    # Test with theorized lightcurve
    lightcurve_df = pd.DataFrame({
        'mjd': np.linspace(0, 100, 50),
        'mag': np.random.normal(20, 0.5, 50),
        'magerr': np.random.uniform(0.01, 0.1, 50),
        'band': ['g', 'r'] * 25
    })
    
    df = get_timeseries_df(
        ztf_id=None,
        theorized_lightcurve_df=lightcurve_df,
        path_to_timeseries_folder="../data/timeseries",
        path_to_sfd_folder="../data/sfd",
        save_timeseries=False
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    # Test with invalid ZTF ID
    with pytest.raises(Exception):  # Could be various exceptions depending on implementation
        get_timeseries_df(
            ztf_id="invalid_id",
            path_to_timeseries_folder="../data/timeseries",
            path_to_sfd_folder="../data/sfd"
        ) 