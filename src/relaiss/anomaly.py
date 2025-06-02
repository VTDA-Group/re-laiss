import os
import pickle
import time
from pathlib import Path
import antares_client
import matplotlib
# Only use interactive backend if display is available
if os.environ.get("DISPLAY") or os.name == 'nt':  # Windows or X11 display available
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')  # Headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from .fetch import get_timeseries_df, get_TNS_data
from .features import build_dataset_bank
from sklearn.neighbors import NearestNeighbors
from scipy.stats import genpareto
from . import anomaly_config as config

def train_AD_model(
    client,
    lc_features,
    host_features,
    path_to_dataset_bank=None,
    preprocessed_df=None,
    path_to_sfd_folder=None,
    path_to_models_directory="../models",
    force_retrain=True,
):
    """Train a scaler for k-NN distance-based anomaly detection.

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

    # Generate model filename based on features
    num_lc_features = len(lc_features)
    num_host_features = len(host_features)
    model_name = config.MODEL_FILENAME_TEMPLATE.format(
        num_lc=num_lc_features, 
        num_host=num_host_features
    )
    model_path = os.path.join(path_to_models_directory, model_name)

    # Check if model already exists
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing scaler from {model_path}")
        return model_path

    print("Training new feature scaler for k-NN distance anomaly detection...")

    # Get features from preprocessed dataframe or load and process raw data
    built_for_AD = getattr(client, 'built_for_AD', False)
    if (preprocessed_df is not None) and built_for_AD:
        print("Using provided preprocessed dataframe")
        df = preprocessed_df
    else:
        print("Loading and preprocessing dataset bank for AD...")
        raw_df = pd.read_csv(path_to_dataset_bank, low_memory=False)
        df = build_dataset_bank(
            raw_df,
            path_to_sfd_folder=path_to_sfd_folder,
            building_entire_df_bank=True,
            building_for_AD=True
        )
    
    # Use unfiltered training dataset to preserve population diversity
    print("Using unfiltered training dataset to preserve supernova diversity...")
    print(f"Training set size: {len(df):,} objects (no filtering applied)")
    
    # Retain full training set to capture complete range of normal variability

    # Extract features
    feature_names = lc_features + host_features

    # Check if all required features exist in the dataframe
    missing_features = [feat for feat in feature_names if feat not in df.columns]
    if missing_features:
        available_features = list(df.columns)
        print(f"\nERROR: The following features are not available in the dataset:")
        for feat in missing_features:
            print(f"  - {feat}")
        
        print(f"\nAvailable features in dataset ({len(available_features)} total):")
        # Show lightcurve-related features
        lc_related = [f for f in available_features if any(prefix in f for prefix in ['g_', 'r_', 'mean_g', 'mean_r'])]
        if lc_related:
            print(f"  Lightcurve features: {lc_related[:10]}...")
            if len(lc_related) > 10:
                print(f"    ... and {len(lc_related) - 10} more")
        
        # Show host-related features  
        host_related = [f for f in available_features if any(word in f.lower() for word in ['kron', 'mag', 'moment', 'ext'])]
        if host_related:
            print(f"  Host features: {host_related[:10]}...")
            if len(host_related) > 10:
                print(f"    ... and {len(host_related) - 10} more")
        
        print(f"\nSOLUTION: Use only features that exist in your dataset.")
        print(f"You can run the diagnostic script to see all available features:")
        print(f"  python debug_features.py")
        
        raise ValueError(f"Missing features: {missing_features}")

    # Extract features and drop samples with NaN values instead of imputing
    X_df = df[feature_names].copy()
    
    # Check for NaN values and drop samples with any NaN
    if X_df.isnull().any().any():
        initial_size = len(X_df)
        X_df = X_df.dropna()
        final_size = len(X_df)
        dropped_percent = ((initial_size - final_size) / initial_size) * 100
        
        if dropped_percent > config.HIGH_NAN_WARNING_THRESHOLD:
            print(f"WARNING: High NaN rate ({dropped_percent:.1f}%) may bias results. Consider imputation.")
        elif dropped_percent > config.VERY_HIGH_NAN_ERROR_THRESHOLD:
            print(f"ERROR: Very high NaN rate ({dropped_percent:.1f}%) - results may be unreliable!")
    
    X = X_df.values

    # Train feature scaler for k-NN distance computation
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X)

    # Store scaler and a reasonable sample of training features for k-NN queries
    # This balances memory efficiency with functionality
    sample_size = min(config.TRAINING_SAMPLE_SIZE, len(X))
    sample_indices = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Compute k-NN distances on full training set for distribution reference
    X_scaled = scaler.transform(X)
    
    # Adjust k based on available training data
    max_k = min(21, len(X))  # k=20 + 1 for self, but cap at training size
    if max_k < 6:  # Need at least 5 neighbors for meaningful anomaly detection
        print(f"Warning: Very small training set ({len(X)} samples). k-NN anomaly detection may be unreliable.")
        if len(X) == 1:
            # Cannot do k-NN with only 1 sample - use a dummy approach
            max_k = 1
            print("Using distance from single training sample as anomaly metric.")
        else:
            max_k = max(2, len(X) - 1)  # Ensure k < training size
    
    nbrs = NearestNeighbors(n_neighbors=max_k)
    nbrs.fit(X_scaled)
    train_distances, _ = nbrs.kneighbors(X_scaled)
    train_knn_distances = train_distances[:, -1]  # k-th neighbor distance
    
    # Prepare scaled training sample for efficient k-NN queries during inference
    X_sample_scaled = scaler.transform(X_sample)
    
    # Fit a smaller NearestNeighbors object on the sample for memory efficiency
    sample_nbrs = NearestNeighbors(n_neighbors=min(max_k, len(X_sample_scaled)))
    sample_nbrs.fit(X_sample_scaled)
    
    # Store essential data with memory optimization
    model_data = {
        'scaler': scaler,
        'feature_names': feature_names,
        'training_sample_scaled': X_sample_scaled,  # Pre-scaled sample for test k-NN queries
        'sample_nbrs': sample_nbrs,  # Pre-fitted k-NN model for test queries
        'train_knn_distances': train_knn_distances,  # Full distance distribution for GPD
        'training_k': max_k - 1  # Actual k used (subtract 1 for self-neighbor)
    }
    
    joblib.dump(model_data, model_path)
    print(f"Scaler and training data saved to: {model_path}")

    return model_path

def _compute_completeness_weights(feature_matrix):
    """
    Computation of completeness weights.
    
    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Feature matrix (n_epochs, n_features)
        
    Returns
    -------
    numpy.ndarray
        Completeness weights for each epoch
    """
    # Count non-null features per epoch
    non_null_counts = np.sum(~np.isnan(feature_matrix), axis=1)
    total_features = feature_matrix.shape[1]
    
    # Simple fraction of available features per epoch
    frac_available = non_null_counts / max(total_features, 1)
    
    return np.maximum(0.3, 0.3 + 0.7 * frac_available)


def _calc_anomaly_score(knn_d, train_knn_d, thresh_quant=0.995, sf_weight=0.7, xi_shrink=0.7):
    """
    Streamlined anomaly score using GPD tail estimation.
    
    Minimal performant implementation following user audit recommendations.
    
    Parameters
    ----------
    knn_d : float
        k-NN distance for test point
    train_knn_d : array-like
        k-NN distances for training data
    thresh_quant : float, default 0.995
        Quantile threshold for GPD tail fitting
    sf_weight : float, default 0.7
        Power tempering factor for survival function
    xi_shrink : float, default 0.7
        Shape parameter shrinkage factor
    
    Returns
    -------
    float
        Anomaly score as -log10(tail probability)
    """
    thresh = np.quantile(train_knn_d, thresh_quant)
    
    if knn_d <= thresh:
        # Below threshold: use empirical probability
        p = (train_knn_d < knn_d).mean()
        return -np.log10(1 - p + 1e-12)
    else:
        # Above threshold: use GPD tail
        excess = train_knn_d[train_knn_d > thresh] - thresh
        
        if len(excess) < 5:  # Fallback if insufficient tail data
            p = (train_knn_d < knn_d).mean()
            return -np.log10(1 - p + 1e-12)
        
        try:
            # Fit GPD to tail exceedances
            xi, _, beta = genpareto.fit(excess, floc=0)
            
            # Apply regularization: shrink xi and clip
            xi = min(0.25, xi_shrink * xi)  # ξ hard clip to 0.25 + shrinkage
            
            # GPD survival function with power tempering
            p = (1 - thresh_quant) * (genpareto.sf(knn_d - thresh, xi, loc=0, scale=beta) ** sf_weight)
            
            return -np.log10(p + 1e-12)
        
        except:
            # Fallback to empirical method if GPD fitting fails
            p = (train_knn_d < knn_d).mean()
            return -np.log10(1 - p + 1e-12)

def anomaly_detection(
    client,
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
    anom_thresh=50,
    random_seed=42,
    force_retrain=True,
    preprocessed_df=None,
    return_scores=False,
    verbose=False,
):
    """Run k-NN distance-based anomaly detection for a single transient.

    Generates anomaly score plots using k-NN distances with GPD tail modeling.

    Parameters
    ----------
    transient_ztf_id : str
        Target object ID.
    host_ztf_id_to_swap_in : str | None
        Replace host features before scoring.
    lc_features, host_features : list[str]
        Feature lists for lightcurve and host galaxy properties.
    path_* : folders for intermediates, models, and figures.
    save_figures : bool, default True
        Whether to save diagnostic plots.
    anom_thresh : float, default 50
        Anomaly threshold percentage for flagging.
    force_retrain : bool, default False
        Whether to retrain scaler.
    preprocessed_df : pandas.DataFrame | None, optional
        Pre-processed dataframe with imputed features.
    verbose : bool, default False
        Whether to print detailed results summary.

    Returns
    -------
    None or tuple
        Returns scores if return_scores=True, otherwise None.
    """

    # Set plotting style
    try:
        import seaborn as sns
        sns.set_context("talk")
    except ImportError:
        pass  # Seaborn is optional
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Palatino"]
    })

    print("Running reLAISS in anomaly detection mode:\n")

    # Train the scaler (if necessary)
    path_to_trained_model = train_AD_model(
        client,
        lc_features,
        host_features,
        path_to_dataset_bank,
        preprocessed_df=preprocessed_df,
        path_to_sfd_folder=path_to_sfd_folder,
        path_to_models_directory=path_to_models_directory,
        force_retrain=force_retrain,
    )

    # Load the model data
    model_data = joblib.load(path_to_trained_model)
    scaler = model_data['scaler']
    
    # Handle both new optimized format and legacy format
    if 'training_sample_scaled' in model_data:
        training_sample_scaled = model_data['training_sample_scaled']
        sample_nbrs = model_data['sample_nbrs']
        train_knn_distances = model_data['train_knn_distances']
    else:
        # Legacy format with full training_features - convert on the fly
        training_features = model_data['training_features']
        sample_size = min(config.TRAINING_SAMPLE_SIZE, len(training_features))
        sample_indices = np.random.RandomState(random_seed).choice(len(training_features), sample_size, replace=False)
        training_sample = training_features[sample_indices]
        
        # Compute k-NN distances for legacy models
        X_scaled = scaler.transform(training_features)
        training_sample_scaled = scaler.transform(training_sample)
        
        # Apply same safeguards as in training
        max_k = min(21, len(training_features))
        if max_k < 6:
            print(f"Warning: Very small legacy training set ({len(training_features)} samples). k-NN anomaly detection may be unreliable.")
            if len(training_features) == 1:
                max_k = 1
                print("Using distance from single training sample as anomaly metric.")
            else:
                max_k = max(2, len(training_features) - 1)
        
        # Create legacy objects for compatibility
        sample_nbrs = NearestNeighbors(n_neighbors=min(max_k, len(training_sample_scaled)))
        sample_nbrs.fit(training_sample_scaled)
        
        nbrs = NearestNeighbors(n_neighbors=max_k)
        nbrs.fit(X_scaled)
        train_distances, _ = nbrs.kneighbors(X_scaled)
        train_knn_distances = train_distances[:, -1]

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
        save_timeseries=True,
        building_for_AD=True
    )

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
            swapped_host=True
        )

        host_values = swapped_host_timeseries_df[host_features].iloc[0]
        for col in host_features:
            timeseries_df[col] = host_values[col]

    timeseries_df.sort_values(by=['mjd_cutoff'], inplace=True)
    timeseries_df_filt_feats = timeseries_df[lc_features + host_features]
    input_lightcurve_locus = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=transient_ztf_id
    )

    tns_name, tns_cls, tns_z = get_TNS_data(transient_ztf_id)

    # Run the anomaly detection check
    mjd_anom, anom_scores, norm_scores = check_anom_and_plot(
        scaler=scaler,
        training_sample_scaled=training_sample_scaled,
        sample_nbrs=sample_nbrs,
        train_knn_distances=train_knn_distances,
        model_data=model_data,
        input_ztf_id=transient_ztf_id,
        swapped_host_ztf_id=host_ztf_id_to_swap_in,
        input_spec_cls=tns_cls,
        input_spec_z=tns_z,
        anom_thresh=anom_thresh,
        timeseries_df_full=timeseries_df,
        timeseries_df_features_only=timeseries_df_filt_feats,
        ref_info=input_lightcurve_locus,
        savefig=save_figures,
        figure_path=path_to_figure_directory,
        verbose=verbose,
    )
    if not return_scores:
        return
    else:
        return mjd_anom, anom_scores, norm_scores


def check_anom_and_plot(
    scaler,
    training_sample_scaled,
    sample_nbrs,
    train_knn_distances,
    model_data,
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
    N_train_samples=5000,
    pct_low=10,
    pct_high=90,
    mid_shift=0.2,
    pct_width_divisor=3,
    verbose=False,
):
    """Compute anomaly scores over time-series epochs and generate diagnostic plots.

    This function implements a sophisticated anomaly detection pipeline combining:
    1. k-NN distance-based outlier detection in feature space
    2. Generalized Pareto Distribution (GPD) tail modeling for score calibration
    3. Completeness weighting to correct for early-epoch bias
    4. False positive control (removing SNe Ia)
    5. Logistic transform to convert to percentage scores

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted feature scaler for normalization.
    training_sample_scaled : numpy.ndarray
        Pre-scaled sample of training features for test k-NN queries.
    sample_nbrs : sklearn.neighbors.NearestNeighbors
        Pre-fitted k-NN model on training sample for test queries.
    train_knn_distances : numpy.ndarray
        Pre-computed k-NN distances from training data for distribution reference.
    model_data : dict
        Model data dictionary containing training metadata.
    input_ztf_id : str
        ZTF object identifier.
    swapped_host_ztf_id : str | None
        Alternative host galaxy ID for counterfactual analysis.
    input_spec_cls : str | None
        Spectroscopic classification for annotation.
    input_spec_z : float | str | None
        Redshift measurement for annotation.
    anom_thresh : float
        Anomaly threshold percentage for flagging.
    timeseries_df_full : pandas.DataFrame
        Complete timeseries with all features and metadata.
    timeseries_df_features_only : pandas.DataFrame
        Feature-only subset for model input.
    ref_info : antares_client.objects.Locus
        ANTARES object for photometry retrieval.
    savefig : bool
        Whether to save diagnostic plot.
    figure_path : str | Path
        Output directory for plots.
    verbose : bool, default False
        Whether to print detailed results summary.
    pct_low : float, default 10
        Lower percentile for logistic transform calibration.
    pct_high : float, default 90
        Upper percentile for logistic transform calibration.
    mid_shift : float, default 0.2
        Shift for logistic transform calibration.
    pct_width_divisor : float, default 3
        Divisor for width of logistic transform calibration.

    Returns
    -------
    tuple
        (mjd_array, anomaly_scores, normal_scores) for each epoch.

    Notes
    -----
    The anomaly scoring algorithm proceeds in several stages:
    
    1. Scale features and drop nans.
       
    2. Calculate k-NN distances (k=20) in the scaled feature space.
       
    3. Apply epoch-dependent weighting to account for incomplete features in early observations.
       
    4. Use GPD to model the tail distribution of k-NN distances with regularization.
              
    5. Transform log-probabilities percentages using a logistic saturation function calibrated on the
       training distribution.

    """
    timeseries_df_full = timeseries_df_full.copy()
    timeseries_df_full.sort_values(by=['mjd_cutoff'], inplace=True)

    # Drop samples with NaN values (consistent with training)
    feature_df = timeseries_df_features_only.copy()
    
    # Check for NaN values and drop samples with any NaN
    if feature_df.isnull().any().any():
        initial_size = len(feature_df)
        feature_df = feature_df.dropna()
        final_size = len(feature_df)
        dropped_count = initial_size - final_size
        
        print(f"Warning: Dropped {dropped_count} test samples with NaN values")
        print(f"Scoring {final_size} complete test samples")
        
        # Also need to drop corresponding rows from the full timeseries dataframe
        timeseries_df_full = timeseries_df_full.loc[feature_df.index]
    
    feature_matrix = feature_df.values
    
    # Scale features using the trained scaler
    test_scaled = scaler.transform(feature_matrix)
    
    # Get the k value used during training, with fallback to default
    training_k = model_data.get('training_k', config.DEFAULT_K)
    k = min(training_k, len(training_sample_scaled) - 1)  # Ensure k doesn't exceed training size
    k = max(k, 1) # Ensure the code works for small training sets

    print(f"Using k-NN distance anomaly detection (k={k})!")

    # Get distances for test data using pre-fitted k-NN model
    test_distances, _ = sample_nbrs.kneighbors(test_scaled)

    # Use the last available neighbor distance (may be less than k for small training sets)
    test_knn_distances = test_distances[:, -1]
    
    # Adjust distances for early-epoch feature incompleteness
    feature_names = list(timeseries_df_features_only.columns)
    completeness_weights = _compute_completeness_weights(feature_matrix)
    test_knn_distances_weighted = test_knn_distances * completeness_weights
    
    # Apply completeness weighting to training distances for consistent reference
    # Assume most training data is relatively complete, so weight is 90%
    Nwgts = min(len(training_sample_scaled), N_train_samples)
    train_knn_distances_weighted = train_knn_distances[:Nwgts]*0.9
    
    # Calculate reference distribution for natural scaling using weighted distances
    sampled = train_knn_distances_weighted[::100]
    train_log_scores = np.fromiter(
        (_calc_anomaly_score(d, train_knn_distances_weighted) for d in sampled),
        dtype=float,
        count=len(sampled)
    )

    # Apply calibrated logistic transform to convert log-probability scores to percentages
    # Use training distribution for calibration
    score_median = np.median(train_log_scores)
    score_p10 = np.percentile(train_log_scores, pct_low)
    score_p90 = np.percentile(train_log_scores, pct_high)
    
    # Calibrate logistic transform parameters from training distribution
    mid = score_median + mid_shift  # Slight shift to compress normal scores
    width = max(1e-6, (score_p90 - score_p10) / pct_width_divisor)  # Width based on training score spread
    
    # calculate epoch-specific test scores
    refined_scores = np.fromiter(
        (_calc_anomaly_score(d, train_knn_distances_weighted)
        for d in test_knn_distances_weighted),
        dtype=float,
        count=len(test_knn_distances_weighted)
    )

    # Build phase-guard weights -- (don't trust huge anomaly 
    # spikes during the first 12 observations)
    # unfortunately a bit hard-coded for now
    epochs = np.arange(1, len(refined_scores) + 1)
    phase_weights = np.where(
        epochs < 12,
        np.maximum(0.15, (epochs - 5) / 7.0),   # ramp 5 → 12
        1.0
    )

    # apply phase guard
    capped_scores = np.minimum(refined_scores, -np.log10(2e-3))
    anom_score_percentile = phase_weights * capped_scores 
        
    # sigmoid transform and bound between 0 and 100%
    anom_score_percentile = 100.0 / (
        1.0 + np.exp(-(anom_score_percentile - mid) / width)
    )

    # convert to "normal" percentages
    norm_score_percentile = 100 - anom_score_percentile

    anom_score = np.array([round(a, 1) for a in anom_score_percentile])
    norm_score = np.array([round(b, 1) for b in norm_score_percentile])
    
    anom_mjd = timeseries_df_full.mjd_cutoff
    
    num_anom_epochs = len(np.where(anom_score >= anom_thresh)[0])

    try:
        anom_idx = timeseries_df_full.iloc[
            np.where(anom_score >= anom_thresh)[0][0]
        ].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(
            f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {input_ztf_id}"
            + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else ".")
        )
        anom_idx_is = False

    max_anom_score = np.max(anom_score) if anom_score.size > 0 else 0
    final_anom_score = anom_score[-1] if len(anom_score) > 0 else 0
    
    if verbose:
        print(f"\nDetailed results for {input_ztf_id}:")
        print(f"   Final anomaly score: {final_anom_score:.1f}%")
        print(f"   Maximum anomaly score: {max_anom_score:.1f}%")
        print(f"   All epoch scores: {[f'{s:.1f}%' for s in anom_score[:10]]}")  # First 10 epochs
        if len(completeness_weights) > 0:
            print(f"   Completeness weights: {[f'{w:.2f}' for w in completeness_weights[:10]]}")  # First 10 epochs
        print(f"   Total epochs processed: {len(anom_score)}")
    
    # Classification based on final score
    if verbose:
        if final_anom_score >= 80:
            classification = "Near-Certain Anomaly"
        elif final_anom_score >= 70:
            classification = "High-Probability Anomaly"
        elif final_anom_score >= 50:
            classification = "Medium-Probability Anomaly"
        else:
            classification = "Normal Event/Low-Probability Anomaly"
        
        print(f"   Classification: {classification}")
        print(f"   {'='*60}")

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

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 13))
    ax1.invert_yaxis()
    ax1.errorbar(
        x=df_ref_r.ant_mjd,
        y=df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        mec='k',
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g.ant_mjd,
        y=df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        mec='k',
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )

    # Set the x-axis range to match the light curve data
    ax1.set_xlim(mjd_range)

    # Plot both traditional and corrected percentile-based probabilities
    ax2.plot(
        anom_mjd,
        norm_score,
        drawstyle="steps",
        label=r"$p(Normal)$",
        linewidth=2,
    )
    ax2.plot(
        anom_mjd,
        anom_score,
        drawstyle="steps",
        label=r"$p(Anomaly)$",
        linewidth=2,
    )
    
    # Add vertical lines for key anomaly detection moments
    first_above_thresh_idx = np.where(anom_score >= anom_thresh)[0]
    if len(first_above_thresh_idx) > 0:
        first_thresh_mjd = anom_mjd.iloc[first_above_thresh_idx[0]]
        ax1.axvline(first_thresh_mjd, color='tab:blue', alpha=0.8, linewidth=3)
        ax2.axvline(first_thresh_mjd, color='tab:blue', alpha=0.8, linewidth=3)
        
        ax2.annotate(
            fr'MJD$_{{\mathrm{{first}}}}$ = {first_thresh_mjd:.1f}',
            xy=(first_thresh_mjd, 1.05),  # Directly above the line
            xycoords=('data', 'axes fraction'),
            fontsize=16,
            ha='center',
            color='tab:blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='white', alpha=0.8)
        )

    peak_anom_idx = np.argmax(anom_score)
    peak_mjd = anom_mjd.iloc[peak_anom_idx]
    peak_score = anom_score[peak_anom_idx]
    
    ax1.axvline(peak_mjd, color='tab:purple', alpha=0.8, linewidth=3)
    ax2.axvline(peak_mjd, color='tab:purple', alpha=0.8, linewidth=3)
    
    ax2.annotate(
        fr'MJD$_{{\mathrm{{peak}}}}$ = {peak_mjd:.1f}',
        xy=(peak_mjd, 1.05),  # Directly above the line
        xycoords=('data', 'axes fraction'),
        fontsize=16,
        ha='center',
        color='tab:purple',                      
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='white', alpha=0.8)
    )
    
    if input_spec_z is None:
        input_spec_z = "None"
    elif isinstance(input_spec_z, float):
        input_spec_z = round(input_spec_z, 3)
    else:
        input_spec_z = input_spec_z

    pmax_text = r"$p_{\mathrm{max}}$"
    ax1.set_title(
        rf"{input_ztf_id} ({input_spec_cls}, $z$={input_spec_z})"
        + (f" with host from {swapped_host_ztf_id}" if swapped_host_ztf_id else "")
        + f" [{pmax_text}(Anomaly)= {max_anom_score:.1f}%]",
        pad=25,
    )
    plt.xlabel("MJD")
    ax1.set_ylabel("Apparent Magnitude")
    ax2.set_ylabel("Anomaly Score (%)")

    if anom_idx_is == True:
        ax1.legend(
            loc="upper right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            facecolor='white',
            edgecolor='black',
            framealpha=0.6,
            fontsize=16,
        )
    else:
        ax1.legend(
            loc="upper right", 
            ncol=2,
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            facecolor='white',
            edgecolor='black',
            framealpha=0.6,
            fontsize=16,
        )
    ax2.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=0.9,
        fontsize=16,
    )

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        figure_dir = Path(figure_path)
        ad_dir = figure_dir / "AD"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(ad_dir, exist_ok=True)
        
        filename = f"{input_ztf_id}"
        if swapped_host_ztf_id:
            filename += f"_w_host_{swapped_host_ztf_id}"
        filename += "_AD.pdf"
        
        save_path = ad_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved anomaly detection chart to: {save_path}")
    
    # Only show plot in interactive environments
    if os.environ.get("INTERACTIVE", "0") == "1" or (hasattr(plt.get_backend(), 'lower') and 'tk' in plt.get_backend().lower()):
        plt.show()

    return anom_mjd, anom_score, norm_score


