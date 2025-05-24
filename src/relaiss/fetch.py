import io
import os

import antares_client
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from PIL import Image

from .features import extract_lc_and_host_features
from .utils import (
    compute_dataframe_hash,
    get_cache_key,
    load_cached_dataframe,
    cache_dataframe,
)


def get_TNS_data(ztf_id):
    """Fetch the TNS cross-match for a given ZTF object.

    Parameters
    ----------
    ztf_id : str
        ZTF object ID, e.g. ``"ZTF23abcxyz"``.

    Returns
    -------
    tuple[str, str, float]
        *(tns_name, tns_type, tns_redshift)*.  Values default to
        ``("No TNS", "---", -99)`` when no match or metadata are present.
    """
    locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id)
    try:
        tns = locus.catalog_objects["tns_public_objects"][0]
        tns_name, tns_cls, tns_z = tns["name"], tns["type"], tns["redshift"]
    except:
        tns_name, tns_cls, tns_z = "No TNS", "---", -99
    if tns_cls == "":
        tns_cls, tns_ann_z = "---", -99
    return tns_name, tns_cls, tns_z

def _ps1_list_filenames(ra_deg, dec_deg, flt):
    """Return the first PS1 stacked-image FITS filename at (RA, Dec).

    Parameters
    ----------
    ra_deg, dec_deg : float
        ICRS coordinates in degrees.
    flt : str
        PS1 filter letter (``'g' 'r' 'i' 'z' 'y'``).

    Returns
    -------
    str | None
        Filename, e.g. ``'tess-skycell1001.012-i.fits'``, or *None* when absent.
    """
    url = (
        "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        f"?ra={ra_deg}&dec={dec_deg}&filters={flt}"
    )
    for line in requests.get(url, timeout=20).text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for tok in line.split():
            if tok.endswith(".fits"):
                return tok
    return None


def fetch_ps1_cutout(ra_deg, dec_deg, *, size_pix=100, flt="r"):
    """Download a single-filter PS1 FITS cut-out around *(RA, Dec)*.

    Parameters
    ----------
    ra_deg, dec_deg : float
        ICRS coordinates (degrees).
    size_pix : int, default 100
        Width/height of the square cut-out in PS1 pixels.
    flt : str, default 'r'
        PS1 filter.

    Returns
    -------
    numpy.ndarray
        2-D float array (grayscale image).

    Raises
    ------
    RuntimeError
        When the target lies outside the PS1 footprint or no data exist.
    """
    fits_name = _ps1_list_filenames(ra_deg, dec_deg, flt)
    if fits_name is None:
        raise RuntimeError(f"No {flt}-band stack at this position")

    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        f"?ra={ra_deg}&dec={dec_deg}&size={size_pix}"
        f"&format=fits&filters={flt}&red={fits_name}"
    )
    r = requests.get(url, timeout=40)
    if r.status_code == 400:
        raise RuntimeError("Outside PS1 footprint or no data in this filter")
    r.raise_for_status()

    with fits.open(io.BytesIO(r.content)) as hdul:
        data = hdul[0].data.astype(float)

    if data is None or data.size == 0 or (data != data).all():
        raise RuntimeError("Empty FITS array returned")

    data[data != data] = 0.0
    return data


def fetch_ps1_rgb_jpeg(ra_deg, dec_deg, *, size_pix=100):
    """Fetch an RGB JPEG cut-out (g/r/i) from PS1.

    Falls back via *raising* ``RuntimeError`` when PS1 lacks colour data.

    Parameters
    ----------
    ra_deg, dec_deg : float
        ICRS coordinates (degrees).
    size_pix : int, default 100
        Square cut-out size in pixels.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 array in RGB order.
    """
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        f"?ra={ra_deg}&dec={dec_deg}&size={size_pix}"
        f"&format=jpeg&filters=grizy&red=i&green=r&blue=g&autoscale=99.5"
    )
    r = requests.get(url, timeout=40)
    if r.status_code == 400:
        raise RuntimeError("Outside PS1 footprint or no colour data here")
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.array(img)

def get_timeseries_df(
    ztf_id,
    path_to_timeseries_folder,
    path_to_sfd_folder,
    theorized_lightcurve_df=None,
    save_timeseries=False,
    path_to_dataset_bank=None,
    building_for_AD=False,
    swapped_host=False,
    preprocessed_df=None,
):
    """Retrieve or build a fully-hydrated time-series feature DataFrame.

    Checks disk cache; otherwise calls
    ``extract_lc_and_host_features`` and optionally writes the CSV.

    Parameters
    ----------
    ztf_id : str
    path_to_timeseries_folder : str | Path
    path_to_sfd_folder : str | Path
    theorized_lightcurve_df : pandas.DataFrame | None
        If provided, builds features for a simulated LC.
    save_timeseries : bool, default False
        Persist CSV to disk.
    path_to_dataset_bank : str | Path | None
        Reference bank for imputers.
    building_for_AD : bool, default False
    swapped_host : bool, default False
    preprocessed_df : pandas.DataFrame | None, default None
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.

    Returns
    -------
    pandas.DataFrame
        Feature rows ready for indexing or AD.
    """
    # When preprocessed_df is provided, skip cache and file checks
    # and always use extract_lc_and_host_features to ensure consistency
    if preprocessed_df is not None:
        print("Using provided preprocessed dataframe to extract time series features...")
        return extract_lc_and_host_features(
            ztf_id=ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_folder=path_to_sfd_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            show_lc=False,
            show_host=False,
            store_csv=save_timeseries,
            building_for_AD=building_for_AD,
            swapped_host=swapped_host,
            preprocessed_df=preprocessed_df,
        )

    # Generate cache key based on input parameters
    cache_params = {
        'ztf_id': ztf_id,
        'path_to_sfd_folder': str(path_to_sfd_folder),
        'path_to_dataset_bank': str(path_to_dataset_bank) if path_to_dataset_bank else None,
        'building_for_AD': building_for_AD,
        'swapped_host': swapped_host,
    }

    if theorized_lightcurve_df is not None:
        cache_params['theorized_df_hash'] = compute_dataframe_hash(theorized_lightcurve_df)

    cache_key = get_cache_key('timeseries', **cache_params)

    # Try to load from cache
    cached_df = load_cached_dataframe(cache_key)
    if cached_df is not None:
        if not building_for_AD:
            print("Loading timeseries features from cache...")
        return cached_df

    if theorized_lightcurve_df is not None:
        print("Extracting full lightcurve features for theorized lightcurve...")
        timeseries_df = extract_lc_and_host_features(
            ztf_id=ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_folder=path_to_sfd_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            show_lc=False,
            show_host=True,
            store_csv=save_timeseries,
            swapped_host=swapped_host,
        )
    else:
        # Check if CSV exists in timeseries folder
        csv_path = os.path.join(path_to_timeseries_folder, f"{ztf_id}_timeseries.csv")
        if os.path.exists(csv_path):
            print(f"Loading timeseries from {csv_path}...")
            timeseries_df = pd.read_csv(csv_path)
        else:
            print("Extracting full lightcurve features...")
            timeseries_df = extract_lc_and_host_features(
                ztf_id=ztf_id,
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_folder=path_to_sfd_folder,
                path_to_dataset_bank=path_to_dataset_bank,
                show_lc=False,
                show_host=True,
                store_csv=save_timeseries,
                building_for_AD=building_for_AD,
                swapped_host=swapped_host,
            )

    # Cache the processed DataFrame
    if not building_for_AD:
        print("Caching timeseries features...")
        cache_dataframe(timeseries_df, cache_key)

    return timeseries_df
