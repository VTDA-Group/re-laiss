import numpy as np
import pandas as pd
import os
import sys
import warnings
from contextlib import contextmanager
import io
import logging
import requests
import antares_client
from astropy.io import fits
from PIL import Image
from .features import extract_lc_and_host_features

@contextmanager
def suppress_output():
    """Context-manager that silences *everything* except CRITICAL logs.

    Temporarily redirects ``stdout``/``stderr`` to ``os.devnull``, ignores
    warnings, and disables the root logger for messages < ``logging.CRITICAL``.
    Restores all streams and the logger state on exit.

    Yields
    ------
    None
        Used only for the ``with`` context block.

    Examples
    --------
    >>> with re_suppress_output():
    ...     noisy_function()
    """
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull

        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            logging.disable(logging.NOTSET)
            sys.stdout, sys.stderr = old_stdout, old_stderr

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
    path_to_sfd_data_folder,
    theorized_lightcurve_df=None,
    save_timeseries=False,
    path_to_dataset_bank=None,
    building_for_AD=False,
    swapped_host=False,
):
    """Retrieve or build a fully-hydrated time-series feature DataFrame.

    Checks disk cache; otherwise calls
    ``extract_lc_and_host_features`` and optionally writes the CSV.

    Parameters
    ----------
    ztf_id : str
    path_to_timeseries_folder : str | Path
    path_to_sfd_data_folder : str | Path
    theorized_lightcurve_df : pandas.DataFrame | None
        If provided, builds features for a simulated LC.
    save_timeseries : bool, default False
        Persist CSV to disk.
    path_to_dataset_bank : str | Path | None
        Reference bank for imputers.
    building_for_AD : bool, default False
    swapped_host : bool, default False

    Returns
    -------
    pandas.DataFrame
        Feature rows ready for indexing or AD.
    """
    if theorized_lightcurve_df is not None:
        print("Extracting full lightcurve features for theorized lightcurve...")
        timeseries_df = extract_lc_and_host_features(
            ztf_id=ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            show_lc=False,
            show_host=True,
            store_csv=save_timeseries,
            swapped_host=swapped_host,
        )
        return timeseries_df

    # Check if timeseries already made (but must rebuild for AD regardless)
    if (
        os.path.exists(f"{path_to_timeseries_folder}/{ztf_id}_timeseries.csv")
        and not building_for_AD
    ):
        timeseries_df = pd.read_csv(
            f"{path_to_timeseries_folder}/{ztf_id}_timeseries.csv"
        )
        print(f"Timeseries dataframe for {ztf_id} is already made. Continue!\n")
    else:
        # If timeseries is not made or building for AD, create timeseries by extracting features
        if not building_for_AD:
            print(
                f"Timeseries dataframe does not exist. Re-extracting lightcurve and host features for {ztf_id}."
            )
        timeseries_df = extract_lc_and_host_features(
            ztf_id=ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            show_lc=False,
            show_host=True,
            store_csv=save_timeseries,
            building_for_AD=building_for_AD,
            swapped_host=swapped_host,
        )
    return timeseries_df
