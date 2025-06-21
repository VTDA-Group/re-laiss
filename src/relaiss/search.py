import numpy as np
import pandas as pd
from . import constants
from .fetch import get_timeseries_df, get_TNS_data


def _load_dataset_bank(path, preprocessed_df=None):
    """
    Load and normalize the dataset bank, returning a DataFrame indexed by ztf_object_id.
    """
    if preprocessed_df is not None:
        df = preprocessed_df.copy()
    else:
        df = pd.read_csv(path, low_memory=False)
    if 'ZTFID' in df.columns:
        df = df.rename(columns={'ZTFID': 'ztf_object_id'})
    if 'ztf_object_id' not in df.columns:
        raise ValueError("Dataset bank must contain a 'ztf_object_id' column.")
    return df.set_index('ztf_object_id', drop=True)


def _get_bank_features(df_bank, ztf_id, features):
    """
    Attempt to retrieve features and coords for a given ID from the bank.
    Returns (array, coords_dict). Empty array if missing.
    """
    try:
        row = df_bank.loc[ztf_id]
    except KeyError:
        return np.array([]), {'lc_galaxy_ra': np.nan, 'lc_galaxy_dec': np.nan}
    arr = row.reindex(features).fillna(np.nan).values
    coords = {}
    if {'raMean', 'decMean'}.issubset(df_bank.columns):
        coords = {'lc_galaxy_ra': row.get('raMean', np.nan), 'lc_galaxy_dec': row.get('decMean', np.nan)}
    elif {'host_ra', 'host_dec'}.issubset(row.index):
        coords = {'lc_galaxy_ra': row.get('host_ra', np.nan), 'lc_galaxy_dec': row.get('host_dec', np.nan)}
    else:
        coords = {'lc_galaxy_ra': np.nan, 'lc_galaxy_dec': np.nan}
    return arr, coords


def _extract_timeseries(ztf_id, lc_df, features, dataset_bank_path,
                        timeseries_folder, sfd_folder,
                        save_timeseries, swapped_host, preprocessed_df):
    """
    Safely fetch a timeseries; always returns (array, coords_dict).
    Drops NaNs and returns available data.
    """
    ts = get_timeseries_df(
        ztf_id=ztf_id,
        theorized_lightcurve_df=lc_df,
        path_to_timeseries_folder=timeseries_folder,
        path_to_sfd_folder=sfd_folder,
        path_to_dataset_bank=dataset_bank_path,
        save_timeseries=save_timeseries,
        swapped_host=swapped_host,
        preprocessed_df=preprocessed_df,
    )
    if ts is None or not hasattr(ts, 'dropna'):
        return np.array([]), {'lc_galaxy_ra': np.nan, 'lc_galaxy_dec': np.nan}
    df_clean = ts.dropna(subset=features) if features else ts.copy()
    if df_clean.empty:
        return np.array([]), {'lc_galaxy_ra': np.nan, 'lc_galaxy_dec': np.nan}
    arr = df_clean[features].iloc[-1].values if features else np.array([])
    if {'raMean', 'decMean'}.issubset(df_clean.columns):
        coords = {'lc_galaxy_ra': df_clean['raMean'].iloc[0], 'lc_galaxy_dec': df_clean['decMean'].iloc[0]}
    elif {'host_ra', 'host_dec'}.issubset(df_clean.columns):
        coords = {'lc_galaxy_ra': df_clean['host_ra'].iloc[0], 'lc_galaxy_dec': df_clean['host_dec'].iloc[0]}
    else:
        coords = {'lc_galaxy_ra': np.nan, 'lc_galaxy_dec': np.nan}
    return arr, coords


def primer(
    lc_ztf_id=None,
    theorized_lightcurve_df=None,
    dataset_bank_path=None,
    path_to_timeseries_folder=None,
    path_to_sfd_folder=None,
    save_timeseries=False,
    host_ztf_id=None,
    lc_features=None,
    host_features=None,
    num_sims=0,
    preprocessed_df=None,
    random_seed=42,
    drop_nan_features=True,
):
    """
    Assemble combined feature array; drops NaNs by default.
    Always returns a dict with feature array (possibly reduced) without aborting.
    """
    if (lc_ztf_id is None) == (theorized_lightcurve_df is None):
        raise ValueError("Provide exactly one of lc_ztf_id or theorized_lightcurve_df.")
    if theorized_lightcurve_df is not None and host_ztf_id is None:
        raise ValueError("Providing theorized_lightcurve_df requires host_ztf_id.")
    lc_features = lc_features or []
    host_features = host_features or []
    feature_names = lc_features + host_features
    df_bank = _load_dataset_bank(dataset_bank_path, preprocessed_df)

    def get_entity(ztf_id, features):
        arr, coords = _get_bank_features(df_bank, ztf_id, features)
        # only fallback if features requested but bank returned empty
        if features and arr.size == 0:
            arr_ts, coords_ts = _extract_timeseries(
                ztf_id,
                None if features is host_features or theorized_lightcurve_df is not None else theorized_lightcurve_df,
                features,
                dataset_bank_path,
                path_to_timeseries_folder,
                path_to_sfd_folder,
                save_timeseries,
                swapped_host=(features is host_features),
                preprocessed_df=preprocessed_df,
            )
            arr = arr_ts
            coords.update(coords_ts)
        name, cls, z = get_TNS_data(ztf_id) if ztf_id else ("No TNS", "---", -99)
        return arr, coords, (name, cls, z), ztf_id

    lc_arr, lc_coords, lc_tns, lc_id = get_entity(lc_ztf_id, lc_features)
    if host_ztf_id:
        host_arr, host_coords, host_tns, host_id = get_entity(host_ztf_id, host_features)
    else:
        host_arr = np.array([])
        host_coords = {'lc_galaxy_ra': np.nan, 'lc_galaxy_dec': np.nan}
        host_tns = (None, None, None)
        host_id = None

    combined = np.concatenate([lc_arr, host_arr]) if host_arr.size else lc_arr
    if drop_nan_features and combined.size:
        mask = ~pd.isna(combined)
        combined = combined[mask]
        feature_names = [f for f, m in zip(feature_names, mask) if m]

    output = {
        'lc_ztf_id': lc_id,
        'lc_tns_name': lc_tns[0], 'lc_tns_cls': lc_tns[1], 'lc_tns_z': lc_tns[2],
        'lc_galaxy_ra': lc_coords.get('lc_galaxy_ra', np.nan),
        'lc_galaxy_dec': lc_coords.get('lc_galaxy_dec', np.nan),
        'host_ztf_id': host_id,
        'host_tns_name': host_tns[0], 'host_tns_cls': host_tns[1], 'host_tns_z': host_tns[2],
        'host_galaxy_ra': host_coords.get('lc_galaxy_ra', np.nan),
        'host_galaxy_dec': host_coords.get('lc_galaxy_dec', np.nan),
        'locus_feat_arr': combined,
        'locus_feat_arrs_mc_l': [],
        'lc_feat_names': lc_features,
        'host_feat_names': host_features,
    }
    np.random.seed(random_seed)
    for _ in range(num_sims):
        s = pd.Series(combined, index=feature_names).copy()
        for feat, err in constants.err_lookup.items():
            if feat in s.index and err in s.index:
                s[feat] += np.random.normal(0, s[err])
        output['locus_feat_arrs_mc_l'].append(s.values)

    return output

