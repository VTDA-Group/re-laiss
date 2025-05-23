import os
import pickle
import time
from pathlib import Path

import antares_client
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pyod.models.iforest import IForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.ensemble import IsolationForest

from .fetch import get_timeseries_df, get_TNS_data
from .features import build_dataset_bank


def train_AD_model(
    lc_features,
    host_features,
    path_to_dataset_bank=None,
    preprocessed_df=None,
    path_to_sfd_folder=None,
    path_to_models_directory="../models",
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=False,
):
    """Train an Isolation Forest model for anomaly detection.
    
    Parameters
    ----------
    lc_features : list[str]
        Names of lightcurve features to use.
    host_features : list[str]
        Names of host galaxy features to use.
    path_to_dataset_bank : str | Path | None, optional
        Path to raw dataset bank CSV. Not used if preprocessed_df is provided.
    preprocessed_df : pandas.DataFrame | None, optional
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.
    path_to_sfd_folder : str | Path | None, optional
        Path to SFD dust maps.
    path_to_models_directory : str | Path, default "../models"
        Directory to save trained models.
    n_estimators : int, default 500
        Number of trees in the Isolation Forest.
    contamination : float, default 0.02
        Expected fraction of outliers in the dataset.
    max_samples : int, default 1024
        Number of samples to draw for each tree.
    force_retrain : bool, default False
        Whether to retrain even if a saved model exists.
        
    Returns
    -------
    str
        Path to the saved model file.
        
    Notes
    -----
    Either path_to_dataset_bank or preprocessed_df must be provided.
    If both are provided, preprocessed_df takes precedence.
    """
    if preprocessed_df is None and path_to_dataset_bank is None:
        raise ValueError("Either path_to_dataset_bank or preprocessed_df must be provided")
    
    # Create models directory if it doesn't exist
    os.makedirs(path_to_models_directory, exist_ok=True)
    
    # Generate model filename based on parameters including feature counts
    num_lc_features = len(lc_features)
    num_host_features = len(host_features)
    model_name = f"IForest_n={n_estimators}_c={contamination}_m={max_samples}_lc={num_lc_features}_host={num_host_features}.pkl"
    model_path = os.path.join(path_to_models_directory, model_name)
    
    # Check if model already exists
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing model from {model_path}")
        return model_path
    
    print("Training new Isolation Forest model...")
    
    # Get features from preprocessed dataframe or load and process raw data
    if preprocessed_df is not None:
        print("Using provided preprocessed dataframe")
        df = preprocessed_df
    else:
        print("Loading and preprocessing dataset bank...")
        raw_df = pd.read_csv(path_to_dataset_bank, low_memory=False)
        df = build_dataset_bank(
            raw_df,
            path_to_sfd_folder=path_to_sfd_folder,
            building_entire_df_bank=True,
            building_for_AD=True
        )
    
    # Extract features
    feature_cols = lc_features + host_features
    X = df[feature_cols].values
    
    # Train model
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path


def anomaly_detection(
    transient_ztf_id,
    lc_features,
    host_features,
    path_to_timeseries_folder,
    path_to_sfd_folder,
    path_to_dataset_bank,
    host_ztf_id_to_swap_in=None,
    path_to_models_directory="../models",
    path_to_figure_directory="../figures",
    save_figures=True,
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=False,
    preprocessed_df=None,
):
    """Run anomaly detection for a single transient (with optional host swap).

    Generates an AD probability plot and calls
    :func:`check_anom_and_plot`.

    Parameters
    ----------
    transient_ztf_id : str
        Target object ID.
    host_ztf_id_to_swap_in : str | None
        Replace host features before scoring.
    lc_features, host_features : list[str]
    path_* : folders for intermediates, models, and figures.
    save_figures : bool, default True
    n_estimators, contamination, max_samples : Isolation-Forest params.
    force_retrain : bool, default False
        Pass-through to :func:`train_AD_model`.
    preprocessed_df : pandas.DataFrame | None, optional
        Pre-processed dataframe with imputed features. If provided, this is used
        instead of loading and processing the raw dataset bank.

    Returns
    -------
    None
    """

    print("Running Anomaly Detection:\n")

    # Train the model (if necessary)
    path_to_trained_model = train_AD_model(
        lc_features,
        host_features,
        path_to_dataset_bank,
        preprocessed_df=preprocessed_df,
        path_to_sfd_folder=path_to_sfd_folder,
        path_to_models_directory=path_to_models_directory,
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        force_retrain=force_retrain,
    )

    # Load the model
    clf = joblib.load(path_to_trained_model)

    # If no preprocessed_df was provided, try to find a cached one
    if preprocessed_df is None:
        # Try to find the cached preprocessed dataframe used for training
        from .utils import get_cache_dir
        cache_dir = Path(get_cache_dir())
        
        for cache_file in cache_dir.glob("*.pkl"):
            if "dataset_bank" in str(cache_file) and not cache_file.name.startswith("timeseries"):
                try:
                    cached_df = joblib.load(cache_file)
                    if isinstance(cached_df, pd.DataFrame):
                        preprocessed_df = cached_df
                        print("Using cached preprocessed dataframe for feature extraction")
                        break
                except:
                    continue

    # Load the timeseries dataframe
    print("\nRebuilding timeseries dataframe(s) for AD...")
    timeseries_df = get_timeseries_df(
        ztf_id=transient_ztf_id,
        theorized_lightcurve_df=None,
        path_to_timeseries_folder=path_to_timeseries_folder,
        path_to_sfd_folder=path_to_sfd_folder,
        path_to_dataset_bank=path_to_dataset_bank,
        save_timeseries=False,
        building_for_AD=True,
        preprocessed_df=preprocessed_df,
    )

    # Add mjd_cutoff column for plotting
    if 'ant_mjd' in timeseries_df.columns:
        timeseries_df.loc[:, 'mjd_cutoff'] = timeseries_df['ant_mjd']
    else:
        # If ant_mjd doesn't exist, extract MJD values from the reference data
        print("Warning: ant_mjd column not found, using MJD values from light curve")
        # Get the reference light curve data
        locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=transient_ztf_id)
        df_ref = locus.timeseries.to_pandas()
        
        # Extract unique MJD values from the light curve
        mjd_values = np.sort(df_ref['ant_mjd'].unique())
        
        # If timeseries_df is longer than the actual observations, truncate it
        if len(timeseries_df) > len(mjd_values):
            timeseries_df = timeseries_df.iloc[:len(mjd_values)]
        # If it's shorter, pad with the last MJD values
        elif len(timeseries_df) < len(mjd_values):
            mjd_values = mjd_values[:len(timeseries_df)]
            
        # Assign actual MJD values to the anomaly detection points
        timeseries_df.loc[:, 'mjd_cutoff'] = mjd_values

    if host_ztf_id_to_swap_in is not None:
        # Swap in the host galaxy
        swapped_host_timeseries_df = get_timeseries_df(
            ztf_id=host_ztf_id_to_swap_in,
            theorized_lightcurve_df=None,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_folder=path_to_sfd_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            save_timeseries=False,
            building_for_AD=True,
            swapped_host=True,
            preprocessed_df=preprocessed_df,
        )

        host_values = swapped_host_timeseries_df[host_features].iloc[0]
        for col in host_features:
            timeseries_df[col] = host_values[col]

    timeseries_df_filt_feats = timeseries_df[lc_features + host_features]
    input_lightcurve_locus = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=transient_ztf_id
    )

    tns_name, tns_cls, tns_z = get_TNS_data(transient_ztf_id)
    
    # Suppress the UserWarning about feature names
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X has feature names, but IsolationForest was fitted without feature names")
        # Run the anomaly detection check
        check_anom_and_plot(
            clf=clf,
            input_ztf_id=transient_ztf_id,
            swapped_host_ztf_id=host_ztf_id_to_swap_in,
            input_spec_cls=tns_cls,
            input_spec_z=tns_z,
            anom_thresh=70,
            timeseries_df_full=timeseries_df,
            timeseries_df_features_only=timeseries_df_filt_feats,
            ref_info=input_lightcurve_locus,
            savefig=save_figures,
            figure_path=path_to_figure_directory,
        )
    return


def check_anom_and_plot(
    clf,
    input_ztf_id,
    swapped_host_ztf_id,
    input_spec_cls,
    input_spec_z,
    anom_thresh,
    timeseries_df_full,
    timeseries_df_features_only,
    ref_info,
    savefig,
    figure_path,
):
    """Run anomaly-detector probabilities over a time-series and plot results.

    Produces a two-panel figure: light curve with anomaly epoch marked, and
    rolling anomaly/normal probabilities.

    Parameters
    ----------
    clf : sklearn.base.ClassifierMixin
        Trained binary classifier with ``predict_proba``.
    input_ztf_id : str
        ID of the object evaluated.
    swapped_host_ztf_id : str | None
        Alternate host ID (annotated in title).
    input_spec_cls : str | None
        Spectroscopic class label for title.
    input_spec_z : float | str | None
        Redshift for title.
    anom_thresh : float
        Legacy parameter, now overridden to 70%.
    timeseries_df_full : pandas.DataFrame
        Hydrated LC + host features, including ``mjd_cutoff``.
    timeseries_df_features_only : pandas.DataFrame
        Same rows but feature columns only (classifier input).
    ref_info : antares_client.objects.Locus
        ANTARES locus for retrieving original photometry.
    savefig : bool
        Save the plot as ``AD/*.pdf`` inside *figure_path*.
    figure_path : str | Path
        Output directory.

    Returns
    -------
    None
    """
    # Fix SettingWithCopyWarning by using .copy() and .loc
    timeseries_df_full = timeseries_df_full.copy()

    # Get anomaly scores from decision_function (-ve = anomalous, +ve = normal)
    scores = clf.decision_function(timeseries_df_features_only)
    
    # Convert scores to probabilities (0-100 scale)
    pred_prob_anom = np.zeros((len(scores), 2))
    for i, score in enumerate(scores):
        if i > 0:
            alpha = 0.7
            score = alpha * score + (1 - alpha) * scores[i-1]
        normal_prob = 100 * (1 / (1 + np.exp(-score)))
        anomaly_prob = 100 - normal_prob
        pred_prob_anom[i, 0] = round(normal_prob, 1)
        pred_prob_anom[i, 1] = round(anomaly_prob, 1)

    # Apply smoothing to probabilities - we'll use these smoothed values for everything
    pred_prob_anom_smoothed = pd.DataFrame(pred_prob_anom).rolling(window=3, min_periods=1, center=True).mean().values

    # --- Use weighted average approach for anomaly detection ---
    anom_scores = pred_prob_anom_smoothed[:, 1]  # Use smoothed anomaly probabilities
    anom_thresh = 70.0  # Set fixed threshold at 70%
    min_sustained = 3  # Minimum consecutive points to consider
    min_delta = 2.0   # Minimum increase from baseline
    found_anom = False
    anom_start_idx = None
    
    # First pass: find any periods above threshold
    for i in range(len(anom_scores) - min_sustained + 1):
        window = anom_scores[i:i+min_sustained]
        baseline = np.mean(anom_scores[max(0, i-3):i]) if i > 0 else anom_scores[0]
        
        # Calculate weighted average of scores (more weight to higher scores)
        weights = np.array([0.2, 0.35, 0.45])  # More weight to recent points
        weighted_avg = np.average(window, weights=weights)
        
        # Calculate consistency score (penalize high variance)
        score_variance = np.std(window)
        consistency_penalty = min(score_variance / 10.0, 2.0)  # Cap the penalty
        
        # Adjusted score that accounts for:
        # 1. Weighted average of the anomaly scores
        # 2. How much it increased from baseline
        # 3. Consistency of the scores
        adjusted_score = weighted_avg - consistency_penalty
        baseline_increase = weighted_avg - baseline
        
        if (adjusted_score >= anom_thresh + 2.0 and  # Must be well above threshold
            baseline_increase >= min_delta and        # Must be significant increase
            all(x >= anom_thresh - 5 for x in window)):  # Allow small dips
            found_anom = True
            anom_start_idx = i
            anom_mjds = timeseries_df_full.mjd_cutoff.iloc[i:i+min_sustained].values
            print(f"\nDEBUG: Analyzing potential anomaly:")
            print(f"MJD range: {anom_mjds[0]:.1f} to {anom_mjds[-1]:.1f}")
            print(f"Raw scores: {window}")
            print(f"Weighted average: {weighted_avg:.1f}")
            print(f"Score variance: {score_variance:.1f}")
            print(f"Baseline: {baseline:.1f}")
            print(f"Adjusted score: {adjusted_score:.1f}")
            break

    num_anom_epochs = min_sustained if found_anom else 0

    if found_anom:
        # Get the MJD values for the anomalous period
        if 0 <= anom_start_idx < len(timeseries_df_full):
            anom_mjds = timeseries_df_full.mjd_cutoff.iloc[anom_start_idx:anom_start_idx+min_sustained].values
            print(f"\nSignificant anomalous behavior detected!")
            print(f"Number of anomalous epochs: {num_anom_epochs}")
            print(f"MJD range: {anom_mjds[0]:.1f} to {anom_mjds[-1]:.1f}")
            anom_idx_is = True
            mjd_cross_thresh = anom_mjds[0]  # Use start of anomalous period for plotting
        else:
            anom_idx_is = False
            print(f"Warning: Anomaly index {anom_start_idx} out of bounds for DataFrame of length {len(timeseries_df_full)}")
    else:
        # Check if we had any moderately high scores
        max_score = np.max(anom_scores)
        if max_score >= 50.0:  # Show info for scores above 50% even if below 70% threshold
            print(f"\nNote: Some epochs showed elevated scores (max: {max_score:.1f}%), "
                  f"but did not reach the {anom_thresh}% threshold required for anomaly detection.")
            print(f"Summary: No sustained anomalous behavior detected (max score: {max_score:.1f}, num_anom_epochs: {num_anom_epochs})")
        else:
            print(f"\nNo significant anomalous behavior detected for {input_ztf_id}"
                  + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else ""))
            print(f"Summary: Normal behavior (max score: {max_score:.1f}, num_anom_epochs: {num_anom_epochs})")
        anom_idx_is = False

    max_anom_score = np.max(anom_scores) if anom_scores.size > 0 else 0

    # Get the light curve data
    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[["ant_mag"]].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[["ant_mag"]].reset_index().idxmin().ant_mag

    # Calculate the actual range of MJD values in the light curve for setting axis limits
    min_mjd = min(df_ref_g['ant_mjd'].min(), df_ref_r['ant_mjd'].min())
    max_mjd = max(df_ref_g['ant_mjd'].max(), df_ref_r['ant_mjd'].max())
    # Add a small margin
    mjd_margin = (max_mjd - min_mjd) * 0.05
    mjd_range = (min_mjd - mjd_margin, max_mjd + mjd_margin)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 10))
    ax1.invert_yaxis()
    ax1.errorbar(
        x=df_ref_r.ant_mjd,
        y=df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g.ant_mjd,
        y=df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )
    
    # Set the x-axis range to match the light curve data
    ax1.set_xlim(mjd_range)
    
    # Only plot the anomaly data points that fall within the light curve's time range
    mask = (timeseries_df_full.mjd_cutoff >= mjd_range[0]) & (timeseries_df_full.mjd_cutoff <= mjd_range[1])
    if mask.any():
        anomaly_mjd = timeseries_df_full.mjd_cutoff[mask]
        anomaly_prob_normal = pred_prob_anom_smoothed[mask, 0]  # Use smoothed values
        anomaly_prob_anomaly = pred_prob_anom_smoothed[mask, 1]  # Use smoothed values
    else:
        # If no data points fall within the range, create some placeholder data
        print("Warning: No anomaly detection data points fall within the light curve's time range")
        anomaly_mjd = np.array([mjd_range[0], mjd_range[1]])
        anomaly_prob_normal = np.array([50, 50])  # Neutral values
        anomaly_prob_anomaly = np.array([50, 50])
        
        # Add a note to the plot
        ax2.text(0.5, 0.5, "No anomaly data in this time range", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14, color='gray',
                 bbox=dict(facecolor='white', alpha=0.8))

    if anom_idx_is == True:
        # Check if the anomaly MJD is within the visible range
        if mjd_range[0] <= mjd_cross_thresh <= mjd_range[1]:
            # Plot vertical lines for start and end of anomalous period
            ax1.axvline(
                x=mjd_cross_thresh,
                label="Anomaly start",
                color="dodgerblue",
                ls="--",
            )
            
            # Add text annotation
            mjd_anom_per = (mjd_cross_thresh - mjd_range[0]) / (mjd_range[1] - mjd_range[0])
            plt.text(
                mjd_anom_per,
                -0.075,
                f"t$_a$ = {int(mjd_cross_thresh)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
                fontsize=16,
                color="dodgerblue",
            )
        else:
            print(f"Warning: Anomaly at MJD {mjd_cross_thresh:.1f} is outside the plotted range {mjd_range}")
            anom_idx_is = False

    # Plot anomaly probabilities using the filtered values within the light curve's time range
    ax2.plot(
        anomaly_mjd,
        anomaly_prob_normal,
        drawstyle="steps",
        label=r"$p(Normal)$",
    )
    ax2.plot(
        anomaly_mjd,
        anomaly_prob_anomaly,
        drawstyle="steps",
        label=r"$p(Anomaly)$",
    )

    if input_spec_z is None:
        input_spec_z = "None"
    elif isinstance(input_spec_z, float):
        input_spec_z = round(input_spec_z, 3)
    else:
        input_spec_z = input_spec_z
    ax1.set_title(
        rf"{input_ztf_id} ({input_spec_cls}, $z$={input_spec_z})"
        + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else ""),
        pad=25,
    )
    plt.xlabel("MJD")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Probability (%)")

    if anom_idx_is == True:
        ax1.legend(
            loc="upper right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.12),
            frameon=False,
            fontsize=14,
        )
    else:
        ax1.legend(
            loc="upper right",
            ncol=2,
            bbox_to_anchor=(0.75, 1.12),
            frameon=False,
            fontsize=14,
        )
    ax2.legend(
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.87, 1.12),
        frameon=False,
        fontsize=14,
    )

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(figure_path + "/AD", exist_ok=True)
        plt.savefig(
            (
                f"{figure_path}/AD/{input_ztf_id}"
                + (f"_w_host_{swapped_host_ztf_id}" if swapped_host_ztf_id else "")
                + "_AD.pdf"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Saved anomaly detection chart to:"
            + f"{figure_path}/AD/{input_ztf_id}"
            + (f"_w_host_{swapped_host_ztf_id}" if swapped_host_ztf_id else "")
            + "_AD.pdf"
        )
    plt.show()
