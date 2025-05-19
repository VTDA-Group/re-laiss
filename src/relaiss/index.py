import os
import logging
import joblib

import annoy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def build_indexed_sample(
    data_bank,
    lc_features=[],
    host_features=[],
    use_pca=False,
    num_pca_components=None,
    num_trees=1000,
    path_to_index_directory="",
    save=True,
    force_recreation_of_index=True,
    weight_lc_feats_factor=1.0,
):
    """Create (or load) an ANNOY index over a reference feature bank.

    Parameters
    ----------
    data_bank : Pandas DataFrame
        CSV produced by ``build_data_bank``; must contain ``ztf_object_id``.
    lc_features, host_features : list[str]
        Feature columns to include in the index.
        Provide one or both lists.
    use_pca : bool, default False
        Apply PCA before indexing.
    n_components : int | None
        Dimensionality of PCA space; ignored if *use_pca=False*.
    num_trees : int, default 1000
        Number of random projection trees for ANNOY.
    path_to_index_directory : str | Path, default ""
        Folder for ``*.ann`` plus ``*.npy`` support files.
    save : bool, default True
        Persist index and numpy arrays.
    force_recreation_of_index : bool, default False
        Rebuild even when an index file already exists.
    weight_lc_feats_factor : float, default 1
        Scalar >1 up-weights LC columns relative to host features
        (ignored if *use_pca=True*).

    Returns
    -------
    str
        Stem path (without ``.ann`` extension) of the built/loaded index.

    Raises
    ------
    ValueError
        When feature inputs are invalid or required columns are missing.
    """

    # Confirm that the first column is the ZTF ID, and index by ZTF ID
    if "ztf_object_id" not in data_bank.columns.values:
        if "ZTFID" in data_bank.columns.values:
            data_bank = data_bank.rename(columns={'ZTFID': 'ztf_object_id'})
        else:
            raise ValueError(
                f"Error: Expected 'ztf_object_id' or 'ZTFID' column in dataset bank, but got '{data_bank.columns[0]}' instead."
            )
    data_bank = data_bank.set_index("ztf_object_id")

    print("INDEXING on features:", lc_features + host_features)

    # Ensure proper user input of features
    num_lc_features   = len(lc_features)
    num_host_features = len(host_features)
    if num_lc_features + num_host_features == 0:
        raise ValueError("Error: must provide at least one lightcurve or host feature.")
    if num_lc_features == 0:
        print(
            f"No lightcurve features provided. Running host-only LAISS with {num_host_features} features."
        )
    if num_host_features == 0:
        print(
            f"No host features provided. Running lightcurve-only LAISS with {num_lc_features} features."
        )

    # Filtering dataset bank for provided features
    data_bank = data_bank[lc_features + host_features]
    data_bank = data_bank.dropna()
    
    print(f"\nIndex building - Feature statistics before scaling:")
    print(f"Number of samples: {len(data_bank)}")
    print(f"Feature means: {data_bank.mean().to_dict()}")
    print(f"Feature stds: {data_bank.std().to_dict()}")
    print(f"Any NaN values: {data_bank.isna().any().any()}")
    print(f"Any infinite values: {np.isinf(data_bank.values).any()}")

    # Scale dataset bank features
    feat_arr = np.array(data_bank)
    idx_arr = np.array(data_bank.index)
    scaler = preprocessing.StandardScaler()
    scaler = scaler.fit(feat_arr)
    feat_arr_scaled = scaler.transform(feat_arr)
    
    print(f"\nIndex building - Feature statistics after scaling:")
    print(f"Scaled feature means: {np.mean(feat_arr_scaled, axis=0)}")
    print(f"Scaled feature stds: {np.std(feat_arr_scaled, axis=0)}")
    print(f"Any NaN values in scaled features: {np.isnan(feat_arr_scaled).any()}")
    print(f"Any infinite values in scaled features: {np.isinf(feat_arr_scaled).any()}")

    if not use_pca:
        # Upweight lightcurve features
        num_lc_feats = len(lc_features)
        feat_arr_scaled[:, :num_lc_feats] *= weight_lc_feats_factor
        print(f"\nIndex building - After LC feature weighting (factor={weight_lc_feats_factor}):")
        print(f"Weighted feature means: {np.mean(feat_arr_scaled, axis=0)}")
        print(f"Weighted feature stds: {np.std(feat_arr_scaled, axis=0)}")
        # Save the weighted feature array
        weighted_feat_arr = feat_arr_scaled.copy()

    if use_pca:
        if weight_lc_feats_factor != 1.0:
            print(
                "Ignoring weighted lightcurve feature factor. Not compatible with PCA."
            )
        random_seed = 88
        pcaModel = PCA(n_components=num_pca_components, random_state=random_seed)
        feat_arr_scaled_pca = pcaModel.fit_transform(feat_arr_scaled)
        print(f"\nIndex building - After PCA:")
        print(f"PCA feature means: {np.mean(feat_arr_scaled_pca, axis=0)}")
        print(f"PCA feature stds: {np.std(feat_arr_scaled_pca, axis=0)}")
        print(f"Explained variance ratio: {pcaModel.explained_variance_ratio_}")

    # Save PCA and non-PCA index arrays to binary files
    os.makedirs(path_to_index_directory, exist_ok=True)
    index_stem_name = (
        f"re_laiss_annoy_index_pca{use_pca}"
        + (f"_{num_pca_components}comps" if use_pca else "")
        + f"_{num_lc_features}lc_{num_host_features}host"
    )
    index_stem_name_with_path = path_to_index_directory + "/" + index_stem_name
    if save:
        np.save(f"{index_stem_name_with_path}_idx_arr.npy", idx_arr)
        np.save(f"{index_stem_name_with_path}_feat_arr.npy", feat_arr)
        # Save the scaler
        joblib.dump(scaler, f"{index_stem_name_with_path}_scaler.joblib")
        if use_pca:
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled_pca.npy",
                feat_arr_scaled_pca,
            )
            # Save the PCA model
            joblib.dump(pcaModel, f"{index_stem_name_with_path}_pca.joblib")
        else:
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled.npy",
                feat_arr_scaled,
            )
            # Save the scaled and weighted feature array used to build the index
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled_weighted.npy",
                feat_arr_scaled,
            )

    # Create or load the ANNOY index:
    index_file = f"{index_stem_name_with_path}.ann"
    index_dim = feat_arr_scaled_pca.shape[1] if use_pca else feat_arr_scaled.shape[1]

    # If the ANNOY index already exists, use it
    if os.path.exists(index_file) and not force_recreation_of_index:
        print("Loading previously saved ANNOY index...")
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(
            f"{index_stem_name_with_path}_idx_arr.npy",
            allow_pickle=True,
        )

    # Otherwise, create a new index
    else:
        print(f"Building new ANNOY index with {data_bank.shape[0]} transients...")

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")

        for i in range(len(idx_arr)):
            index.add_item(i, feat_arr_scaled_pca[i] if use_pca else weighted_feat_arr[i])

        index.build(num_trees)

        if save:
            index.save(index_file)

    print("Done!\n")

    return index_stem_name_with_path, scaler, feat_arr_scaled
