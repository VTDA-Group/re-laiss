from helper_func import *
import pandas as pd
import numpy as np
import os
import annoy
import pickle
import corner
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from kneed import KneeLocator
from pyod.models.iforest import IForest
from statsmodels import robust

def re_build_indexed_sample(
    dataset_bank_path,
    lc_features=[],
    host_features=[],
    use_pca=False,
    n_components=None,
    num_trees=1000,
    path_to_index_directory="",
    save=True,
    force_recreation_of_index=False,
    weight_lc_feats_factor=1,
):
    """Create (or load) an ANNOY index over a reference feature bank.

    Parameters
    ----------
    dataset_bank_path : str | Path
        CSV produced by ``re_build_dataset_bank``; first column must be
        ``ztf_object_id``.
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
    df_bank = pd.read_csv(dataset_bank_path)

    # Confirm that the first column is the ZTF ID, and index by ZTF ID
    if df_bank.columns[0] != "ztf_object_id":
        raise ValueError(
            f"Error: Expected first column in dataset bank to be 'ztf_object_id', but got '{df_bank.columns[0]}' instead."
        )
    df_bank = df_bank.set_index("ztf_object_id")

    # Ensure proper user input of features
    num_lc_features = len(lc_features)
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
    df_bank = df_bank[lc_features + host_features]
    df_bank = df_bank.dropna()

    # Scale dataset bank features
    feat_arr = np.array(df_bank)
    idx_arr = np.array(df_bank.index)
    scaler = preprocessing.StandardScaler()
    feat_arr_scaled = scaler.fit_transform(feat_arr)

    if not use_pca:
        # Upweight lightcurve features
        num_lc_feats = len(lc_features)
        feat_arr_scaled[:, :num_lc_feats] *= weight_lc_feats_factor

    if use_pca:
        if weight_lc_feats_factor != 1:
            print(
                "Ignoring weighted lightcurve feature factor. Not compatible with PCA."
            )
        random_seed = 88
        pcaModel = PCA(n_components=n_components, random_state=random_seed)
        feat_arr_scaled_pca = pcaModel.fit_transform(feat_arr_scaled)

    # Save PCA and non-PCA index arrays to binary files
    os.makedirs(path_to_index_directory, exist_ok=True)
    index_stem_name = (
        f"re_laiss_annoy_index_pca{use_pca}"
        + (f"_{n_components}comps" if use_pca else "")
        + f"_{num_lc_features}lc_{num_host_features}host"
    )
    index_stem_name_with_path = path_to_index_directory + "/" + index_stem_name
    if save:
        np.save(f"{index_stem_name_with_path}_idx_arr.npy", idx_arr)
        np.save(f"{index_stem_name_with_path}_feat_arr.npy", feat_arr)
        if use_pca:
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled.npy",
                feat_arr_scaled,
            )
            np.save(
                f"{index_stem_name_with_path}_feat_arr_scaled_pca.npy",
                feat_arr_scaled_pca,
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
        print(f"Building new ANNOY index with {df_bank.shape[0]} transients...")

        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        for i in range(len(idx_arr)):
            index.add_item(i, feat_arr_scaled_pca[i] if use_pca else feat_arr_scaled[i])

        index.build(num_trees)

        if save:
            index.save(index_file)

    print("Done!\n")

    return index_stem_name_with_path


def re_LAISS_primer(
    lc_ztf_id,
    theorized_lightcurve_df,
    dataset_bank_path,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    save_timeseries=False,
    host_ztf_id=None,
    lc_features=[],
    host_features=[],
    num_sims=10,
):
    """Assemble input feature vectors (and MC replicas) for a query object.

    Combines LC + host features—optionally swapping in a different host—and
    returns a dict used later by NN and AD stages.

    Parameters
    ----------
    lc_ztf_id : str | None
        ZTF ID of the transient to query.  Mutually exclusive with
        *theorized_lightcurve_df*.
    theorized_lightcurve_df : pandas.DataFrame | None
        Pre-computed ANTARES-style LC for a theoretical model.
    host_ztf_id : str | None
        If given, replace the query object’s host features with those of this
        transient.
    dataset_bank_path, path_to_timeseries_folder, path_to_sfd_data_folder : str | Path
        Locations for cached data.
    lc_features, host_features : list[str]
        Names of columns to extract.
    num_sims : int, default 10
        Number of Monte-Carlo perturbations for uncertainty propagation.

    Returns
    -------
    dict
        Primer dictionary containing feature arrays, metadata, and MC sims.

    Raises
    ------
    ValueError
        On inconsistent inputs or missing data.
    """
    feature_names = lc_features + host_features
    if lc_ztf_id is not None and theorized_lightcurve_df is not None:
        print(
            "Expected only one of theorized_lightcurve_df and transient_ztf_id. Try again!"
        )
        raise ValueError(
            "Cannot provide both a transient ZTF ID and a theorized lightcurve."
        )
    if lc_ztf_id is None and theorized_lightcurve_df is None:
        print("Requires one of theorized_lightcurve_df or transient_ztf_id. Try again!")
        raise ValueError(
            "Transient ZTF ID and theorized lightcurve cannot both be None."
        )
    if theorized_lightcurve_df is not None and host_ztf_id is None:
        print(
            "Inputing theorized_lightcurve_df requires host_ztf_id_to_swap_in. Try again!"
        )
        raise ValueError(
            "If providing a theorized lightcurve, must also provide a host galaxy ZTF ID."
        )

    host_galaxy_ra = None
    host_galaxy_dec = None
    lc_galaxy_ra = None
    lc_galaxy_dec = None

    # Loop through lightcurve object and host object to create feature array
    for ztf_id, host_loop in [(lc_ztf_id, False), (host_ztf_id, True)]:

        # Skip host loop if host galaxy to swap is not provided
        if host_loop and ztf_id is None:
            continue

        ztf_id_in_dataset_bank = False

        # Check if ztf_id is in dataset bank
        try:
            df_bank = pd.read_csv(dataset_bank_path, index_col=0)
            # Check to make sure all features are in the dataset bank
            missing_cols = [col for col in feature_names if col not in df_bank.columns]
            if missing_cols:
                raise KeyError(
                    f"KeyError: The following columns are not in the raw data provided: {missing_cols}. Abort!"
                )

            locus_feat_arr = df_bank.loc[ztf_id]

            print(f"{ztf_id} is in dataset_bank.")
            ztf_id_in_dataset_bank = True

            df_bank_input_only = df_bank.loc[[ztf_id]]
            if host_loop:
                host_galaxy_ra = df_bank_input_only.iloc[0].host_ra
                host_galaxy_dec = df_bank_input_only.iloc[0].host_dec
            else:
                lc_galaxy_ra = df_bank_input_only.iloc[0].host_ra
                lc_galaxy_dec = df_bank_input_only.iloc[0].host_dec

            if save_timeseries:
                timeseries_df = re_get_timeseries_df(
                    ztf_id=ztf_id,
                    theorized_lightcurve_df=None,
                    path_to_timeseries_folder=path_to_timeseries_folder,
                    path_to_sfd_data_folder=path_to_sfd_data_folder,
                    path_to_dataset_bank=dataset_bank_path,
                    save_timeseries=save_timeseries,
                    swapped_host=host_loop,
                )

        # If ztf_id is not in dataset bank...
        except:
            # Extract timeseries dataframe
            if ztf_id is not None:
                print(f"{ztf_id} is not in dataset_bank.")
            timeseries_df = re_get_timeseries_df(
                ztf_id=ztf_id,
                theorized_lightcurve_df=(
                    theorized_lightcurve_df if not host_loop else None
                ),
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_data_folder=path_to_sfd_data_folder,
                path_to_dataset_bank=dataset_bank_path,
                save_timeseries=save_timeseries,
                swapped_host=host_loop,
            )

            if host_loop:
                host_galaxy_ra = timeseries_df["raMean"].iloc[0]
                host_galaxy_dec = timeseries_df["decMean"].iloc[0]
            else:
                if theorized_lightcurve_df is None:
                    lc_galaxy_ra = timeseries_df["raMean"].iloc[0]
                    lc_galaxy_dec = timeseries_df["decMean"].iloc[0]

            # If timeseries_df is from theorized lightcurve, it only has lightcurve features
            if not host_loop and theorized_lightcurve_df is not None:
                subset_feats_for_checking_na = lc_features
            else:
                subset_feats_for_checking_na = lc_features + host_features

            timeseries_df = timeseries_df.dropna(subset=subset_feats_for_checking_na)
            if timeseries_df.empty:
                raise ValueError(f"{ztf_id} has some NaN features. Abort!")

            # Extract feature array from timeseries dataframe
            if not host_loop and theorized_lightcurve_df is not None:
                # theorized timeseries_df is just lightcurve data, so we must shape it properly
                for host_feature in host_features:
                    timeseries_df[host_feature] = np.nan

            locus_feat_arr_df = pd.DataFrame(timeseries_df.iloc[-1]).T
            locus_feat_arr = locus_feat_arr_df.iloc[0]

        # Pull TNS data for ztf_id
        if ztf_id is not None:
            tns_name, tns_cls, tns_z = re_getTnsData(ztf_id)
        else:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99

        if host_loop:
            host_tns_name, host_tns_cls, host_tns_z = tns_name, tns_cls, tns_z
            host_ztf_id_in_dataset_bank = ztf_id_in_dataset_bank
            host_locus_feat_arr = locus_feat_arr
        else:
            lc_tns_name, lc_tns_cls, lc_tns_z = tns_name, tns_cls, tns_z
            lc_ztf_id_in_dataset_bank = ztf_id_in_dataset_bank
            lc_locus_feat_arr = locus_feat_arr

    # Make final feature array
    lc_feature_err_names = constants.lc_feature_err.copy()
    host_feature_err_names = constants.host_feature_err.copy()
    feature_err_names = lc_feature_err_names + host_feature_err_names

    if host_ztf_id is None:
        # Not swapping out host, use features from lightcurve ztf_id
        locus_feat_df = lc_locus_feat_arr[feature_names + feature_err_names]
    else:
        # Create new feature array with mixed lc and host features
        subset_lc_features = lc_locus_feat_arr[lc_features + lc_feature_err_names]
        subset_host_features = host_locus_feat_arr[
            host_features + host_feature_err_names
        ]

        locus_feat_df = pd.concat([subset_lc_features, subset_host_features], axis=0)

    # Create Monte Carlo copies locus_feat_arrays_l
    np.random.seed(888)
    err_lookup = constants.err_lookup.copy()
    locus_feat_arrs_mc_l = []
    for _ in range(num_sims):
        locus_feat_df_for_mc = locus_feat_df.copy()

        for feat_name, error_name in err_lookup.items():
            if feat_name in feature_names:
                std = locus_feat_df_for_mc[error_name]
                noise = np.random.normal(0, std)
                if not np.isnan(noise):
                    locus_feat_df_for_mc[feat_name] = (
                        locus_feat_df_for_mc[feat_name] + noise
                    )
                else:
                    pass

        locus_feat_arrs_mc_l.append(locus_feat_df_for_mc[feature_names].values)

    # Create true feature array
    locus_feat_arr = locus_feat_df[feature_names].values

    output_dict = {
        # host data is optional, it's only if the user decides to swap in a new host
        "host_ztf_id": host_ztf_id if host_ztf_id is not None else None,
        "host_tns_name": host_tns_name if host_ztf_id is not None else None,
        "host_tns_cls": host_tns_cls if host_ztf_id is not None else None,
        "host_tns_z": host_tns_z if host_ztf_id is not None else None,
        "host_ztf_id_in_dataset_bank": (
            host_ztf_id_in_dataset_bank if host_ztf_id is not None else None
        ),
        "host_galaxy_ra": host_galaxy_ra if host_ztf_id is not None else None,
        "host_galaxy_dec": host_galaxy_dec if host_ztf_id is not None else None,
        "lc_ztf_id": lc_ztf_id,
        "lc_tns_name": lc_tns_name,
        "lc_tns_cls": lc_tns_cls,
        "lc_tns_z": lc_tns_z,
        "lc_ztf_id_in_dataset_bank": lc_ztf_id_in_dataset_bank,
        "locus_feat_arr": locus_feat_arr,
        "locus_feat_arrs_mc_l": locus_feat_arrs_mc_l,
        "lc_galaxy_ra": lc_galaxy_ra,
        "lc_galaxy_dec": lc_galaxy_dec,
        "lc_feat_names": lc_features,
        "host_feat_names": host_features,
    }

    return output_dict


def re_LAISS_nearest_neighbors(
    primer_dict,
    theorized_lightcurve_df,
    path_to_dataset_bank,
    annoy_index_file_stem,
    use_pca=False,
    num_pca_components=15,
    n=8,
    suggest_neighbor_num=False,
    max_neighbor_dist=None,
    search_k=1000,
    weight_lc_feats_factor=1,
    save_figures=True,
    path_to_figure_directory="../figures",
):
   """Query the ANNOY index and plot nearest-neighbor diagnostics.

    Parameters
    ----------
    primer_dict : dict
        Output from :func:`re_LAISS_primer`.
    annoy_index_file_stem : str
        Stem path returned by :func:`re_build_indexed_sample`.
    use_pca, num_pca_components : see above
    n : int, default 8
        Number of neighbours to return.
    suggest_neighbor_num : bool, default False
        If True, plots the distance elbow and exits early.
    max_neighbor_dist : float | None
        Optional cut on L1 distance.
    search_k : int, default 1000
        ANNOY *search_k* parameter.
    weight_lc_feats_factor : float, default 1
        Same interpretation as in ``re_build_indexed_sample``.
    save_figures : bool, default True
        Write LC + host plots and distance-elbow PNGs.
    path_to_figure_directory : str | Path

    Returns
    -------
    pandas.DataFrame | None
        Table summarising neighbours (or *None* if *suggest_neighbor_num=True*).
    """
    start_time = time.time()
    index_file = annoy_index_file_stem + ".ann"

    if n is None or n <= 0:
        raise ValueError("Neighbor number must be a nonzero integer. Abort!")

    plot_label = (
        f"{primer_dict['lc_ztf_id'] if primer_dict['lc_ztf_id'] is not None else 'theorized_lc'}"
        + (
            f"_host_from_{primer_dict['host_ztf_id']}"
            if primer_dict["host_ztf_id"] is not None
            else ""
        )
    )

    # Find neighbors for every Monte Carlo feature array
    scaler = preprocessing.StandardScaler()
    if use_pca:
        print(
            f"Loading previously saved ANNOY PCA={num_pca_components} index:",
            index_file,
            "\n",
        )
    else:
        print("Loading previously saved ANNOY index without PCA:", index_file, "\n")

    bank_feat_arr = np.load(
        annoy_index_file_stem + "_feat_arr.npy",
        allow_pickle=True,
    )
    trained_PCA_feat_arr_scaled = scaler.fit_transform(bank_feat_arr)

    true_and_mc_feat_arrs_l = [primer_dict["locus_feat_arr"]] + primer_dict[
        "locus_feat_arrs_mc_l"
    ]

    neighbor_dist_dict = {}
    if len(primer_dict["locus_feat_arrs_mc_l"]) != 0:
        print("Running Monte Carlo simulation to find possible neighbors...")
    for locus_feat_arr in true_and_mc_feat_arrs_l:
        # Scale locus_feat_arr using the same scaler (fit on dataset bank feature array)
        locus_feat_arr_scaled = scaler.transform([locus_feat_arr])

        if not use_pca:
            # Upweight lightcurve features
            num_lc_feats = len(constants.lc_features_const.copy())
            locus_feat_arr_scaled[:, :num_lc_feats] *= weight_lc_feats_factor

        if use_pca:
            # Transform the scaled locus_feat_arr using the same PCA model
            random_seed = 88
            pca = PCA(n_components=num_pca_components, random_state=random_seed)

            # pca needs to be fit first to the same data as trained
            trained_PCA_feat_arr_scaled_pca = pca.fit_transform(
                trained_PCA_feat_arr_scaled
            )
            locus_feat_arr_pca = pca.transform(locus_feat_arr_scaled)

            index_dim = num_pca_components
            query_vector = locus_feat_arr_pca[0]

        else:
            index_dim = len(locus_feat_arr)
            query_vector = locus_feat_arr_scaled[0]

        # 3. Use the ANNOY index to find nearest neighbors (common to both branches)
        index = annoy.AnnoyIndex(index_dim, metric="manhattan")
        index.load(index_file)
        idx_arr = np.load(f"{annoy_index_file_stem}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(
            query_vector, n=n, search_k=search_k, include_distances=True
        )

        # Store neighbors and distances in dictionary
        for ann_index, ann_dist in zip(ann_indexes, ann_dists):
            if ann_index in neighbor_dist_dict:
                neighbor_dist_dict[ann_index].append(ann_dist)
            else:
                neighbor_dist_dict[ann_index] = [ann_dist]

    # Pick n neighbors with lowest median distance
    if len(primer_dict["locus_feat_arrs_mc_l"]) != 0:
        print(
            f"Number of unique neighbors found through Monte Carlo: {len(neighbor_dist_dict)}.\nPicking top {n} neighbors."
        )

    medians = {idx: np.median(dists) for idx, dists in neighbor_dist_dict.items()}
    sorted_neighbors = sorted(medians.items(), key=lambda item: item[1])
    top_n_neighbors = sorted_neighbors[:n]

    ann_indexes = [idx for idx, _ in top_n_neighbors]
    ann_dists = [dist for _, dist in top_n_neighbors]

    for i in ann_indexes:
        if idx_arr[i] == primer_dict["lc_ztf_id"]:
            # drop input transient from ann_indexes and ann_dists
            idx_to_del = ann_indexes.index(i)
            del ann_indexes[idx_to_del]
            del ann_dists[idx_to_del]
            print(
                "First neighbor is input transient, so it will be excluded. Final neighbor count will be one less than expected."
            )
            break

    ann_alerce_links = [
        f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes
    ]
    ann_end_time = time.time()

    # Find optimal number of neighbors
    if suggest_neighbor_num:
        number_of_neighbors_found = len(ann_dists)
        neighbor_numbers_for_plot = list(range(1, number_of_neighbors_found + 1))

        knee = KneeLocator(
            neighbor_numbers_for_plot,
            ann_dists,
            curve="concave",
            direction="increasing",
        )
        optimal_n = knee.knee

        if optimal_n is None:
            print(
                "Couldn't identify optimal number of neighbors. Try a larger neighbor pool."
            )
        else:
            print(
                f"Suggested number of neighbors is {optimal_n}, chosen by comparing {n} neighbors."
            )

        plt.figure(figsize=(10, 4))
        plt.plot(
            neighbor_numbers_for_plot,
            ann_dists,
            marker="o",
            label="Distances",
        )
        if optimal_n:
            plt.axvline(
                optimal_n,
                color="red",
                linestyle="--",
                label=f"Elbow at {optimal_n}",
            )
        plt.xlabel("Neighbor Number")
        plt.ylabel("Distance")
        plt.title("Distance for Closest Neighbors")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_figures:
            os.makedirs(path_to_figure_directory, exist_ok=True)
            os.makedirs(
                path_to_figure_directory + "/neighbor_dist_plots/", exist_ok=True
            )
            plt.savefig(
                path_to_figure_directory
                + f"/neighbor_dist_plots/{plot_label}_n={n}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Saved neighbor distances plot to {path_to_figure_directory}/neighbor_dist_plots/n={n}"
            )
        plt.show()

        print(
            "Stopping nearest neighbor search after suggesting neighbor number. Set run_NN=True and suggest_neighbor_num=False for full search.\n"
        )
        return

    # Filter neighbors for maximum distance, if provided
    if max_neighbor_dist is not None:
        filtered_neighbors = [
            (idx, dist)
            for idx, dist in zip(ann_indexes, ann_dists)
            if dist <= abs(max_neighbor_dist)
        ]
        ann_indexes, ann_dists = (
            zip(*filtered_neighbors) if filtered_neighbors else ([], [])
        )
        ann_indexes = list(ann_indexes)
        ann_dists = list(ann_dists)

        if len(ann_dists) == 0:
            raise ValueError(
                f"No neighbors found for distance threshold of {abs(max_neighbor_dist)}. Try a larger maximum distance."
            )
        else:
            print(
                f"Found {len(ann_dists)} neighbors for distance threshold of {abs(max_neighbor_dist)}."
            )

    # 4. Get TNS, spec. class of neighbors
    tns_ann_names, tns_ann_classes, tns_ann_zs, neighbor_ztfids = [], [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        neighbor_ztfids.append(idx_arr[i])

        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)

        tns_ann_name, tns_ann_cls, tns_ann_z = re_getTnsData(idx_arr[i])

        tns_ann_names.append(tns_ann_name)
        tns_ann_classes.append(tns_ann_cls)
        tns_ann_zs.append(tns_ann_z)

    # Print the nearest neighbors and organize them for storage
    if primer_dict["lc_ztf_id"]:
        print(f"\t\t\t\t\t\t ZTFID     IAU_NAME SPEC  Z")
    else:
        print(f"\t\t\t\t\tIAU  SPEC  Z")
    print(
        f"Input transient: {'https://alerce.online/object/'+primer_dict['lc_ztf_id'] if primer_dict['lc_ztf_id'] else 'Theorized Lightcurve,'} {primer_dict['lc_tns_name']} {primer_dict['lc_tns_cls']} {primer_dict['lc_tns_z']}\n"
    )
    if primer_dict["host_ztf_id"] is not None:
        print(f"\t\t\t\t\t\t\t\t\tZTFID     IAU_NAME SPEC  Z")
        print(
            f"Transient with host swapped into input: https://alerce.online/object/{primer_dict['host_ztf_id']} {primer_dict['host_tns_name']} {primer_dict['host_tns_cls']} {primer_dict['host_tns_z']}\n"
        )

    # Plot lightcurves
    re_plot_lightcurves(
        primer_dict=primer_dict,
        plot_label=plot_label,
        theorized_lightcurve_df=theorized_lightcurve_df,
        neighbor_ztfids=neighbor_ztfids,
        ann_locus_l=ann_locus_l,
        ann_dists=ann_dists,
        tns_ann_names=tns_ann_names,
        tns_ann_classes=tns_ann_classes,
        tns_ann_zs=tns_ann_zs,
        figure_path=path_to_figure_directory,
        save_figures=save_figures,
    )

    # Plot hosts
    print("\nGenerating hosts grid plot...")

    df_bank = pd.read_csv(path_to_dataset_bank, index_col="ztf_object_id")

    hosts_to_plot = neighbor_ztfids.copy()
    host_ra_l, host_dec_l = [], []

    for ztfid in hosts_to_plot:
        host_ra, host_dec = (
            df_bank.loc[ztfid].host_ra,
            df_bank.loc[ztfid].host_dec,
        )
        host_ra_l.append(host_ra), host_dec_l.append(host_dec)

    # Add input host for plotting
    if primer_dict["host_ztf_id"] is None:
        hosts_to_plot.insert(0, primer_dict["lc_ztf_id"])
        host_ra_l.insert(0, primer_dict["lc_galaxy_ra"])
        host_dec_l.insert(0, primer_dict["lc_galaxy_dec"])
    else:
        hosts_to_plot.insert(0, primer_dict["host_ztf_id"])
        host_ra_l.insert(0, primer_dict["host_galaxy_ra"])
        host_dec_l.insert(0, primer_dict["host_galaxy_dec"])

    host_ann_df = pd.DataFrame(
        zip(hosts_to_plot, host_ra_l, host_dec_l),
        columns=["ZTFID", "HOST_RA", "HOST_DEC"],
    )

    re_plot_hosts(
        ztfid_ref=(
            primer_dict["lc_ztf_id"]
            if primer_dict["host_ztf_id"] is None
            else primer_dict["host_ztf_id"]
        ),
        plot_label=plot_label,
        df=host_ann_df,
        figure_path=path_to_figure_directory,
        ann_num=n,
        save_pdf=save_figures,
        imsizepix=100,
        change_contrast=False,
        prefer_color=True,
    )

    # Store neighbors and return
    storage = []
    neighbor_num = 1
    for al, iau_name, spec_cls, z, dist in zip(
        ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs, ann_dists
    ):
        print(f"ANN={neighbor_num}: {al} {iau_name} {spec_cls}, {z}")
        neighbor_dict = {
            "input_ztf_id": primer_dict["lc_ztf_id"],
            "input_swapped_host_ztf_id": primer_dict["host_ztf_id"],
            "neighbor_num": neighbor_num,
            "ztf_link": al,
            "dist": dist,
            "iau_name": iau_name,
            "spec_cls": spec_cls,
            "z": z,
        }
        storage.append(neighbor_dict)
        neighbor_num += 1

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time: {round(ann_elapsed_time, 3)} s")
    print(f"total elapsed_time: {round(elapsed_time, 3)} s\n")

    return pd.DataFrame(storage)


def re_train_AD_model(
    lc_features,
    host_features,
    path_to_dataset_bank,
    path_to_models_directory="../models",
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=False,
):
    """Train or load an Isolation-Forest anomaly-detection model.

    Parameters
    ----------
    lc_features, host_features : list[str]
        Feature columns used by the model.
    path_to_dataset_bank : str | Path
    path_to_models_directory : str | Path
    n_estimators, contamination, max_samples : see *pyod.models.IForest*
    force_retrain : bool, default False
        Ignore cached model and retrain.

    Returns
    -------
    str
        Filesystem path to the saved ``.pkl`` pipeline.
    """
    feature_names = lc_features + host_features
    df_bank_path = path_to_dataset_bank
    model_dir = path_to_models_directory
    model_name = f"IForest_n{n_estimators}_c{contamination}_ms{max_samples}_lc{len(lc_features)}_host{len(host_features)}.pkl"

    os.makedirs(model_dir, exist_ok=True)

    print("Checking if AD model exists...")

    # If model already exists, don't retrain
    if os.path.exists(os.path.join(model_dir, model_name)) and not force_retrain:
        print("Model already exists →", os.path.join(model_dir, model_name))
        return os.path.join(model_dir, model_name)

    print("AD model does not exist. Training and saving new model.")

    # Train model
    df = pd.read_csv(df_bank_path, low_memory=False)
    X = df[feature_names].values

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                IForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    max_samples=max_samples,
                    behaviour="new",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X)

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, model_name), "wb") as f:
        pickle.dump(pipeline, f)

    print(
        "Isolation Forest model trained and saved →",
        os.path.join(model_dir, model_name),
    )

    return os.path.join(model_dir, model_name)


def re_anomaly_detection(
    transient_ztf_id,
    lc_features,
    host_features,
    path_to_timeseries_folder,
    path_to_sfd_data_folder,
    path_to_dataset_bank,
    host_ztf_id_to_swap_in=None,
    path_to_models_directory="../models",
    path_to_figure_directory="../figures",
    save_figures=True,
    n_estimators=500,
    contamination=0.02,
    max_samples=1024,
    force_retrain=False,
):
   """Run anomaly detection for a single transient (with optional host swap).

    Generates an AD probability plot and calls
    :func:`re_check_anom_and_plot`.

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
        Pass-through to :func:`re_train_AD_model`.

    Returns
    -------
    None
    """
    print("Running Anomaly Detection:\n")

    # Train the model (if necessary)
    path_to_trained_model = re_train_AD_model(
        lc_features,
        host_features,
        path_to_dataset_bank,
        path_to_models_directory=path_to_models_directory,
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        force_retrain=force_retrain,
    )

    # Load the model
    with open(path_to_trained_model, "rb") as f:
        clf = pickle.load(f)

    # Load the timeseries dataframe
    print("\nRebuilding timeseries dataframe(s) for AD...")
    timeseries_df = re_get_timeseries_df(
        ztf_id=transient_ztf_id,
        theorized_lightcurve_df=None,
        path_to_timeseries_folder=path_to_timeseries_folder,
        path_to_sfd_data_folder=path_to_sfd_data_folder,
        path_to_dataset_bank=path_to_dataset_bank,
        save_timeseries=False,
        building_for_AD=True,
    )

    if host_ztf_id_to_swap_in is not None:
        # Swap in the host galaxy
        swapped_host_timeseries_df = re_get_timeseries_df(
            ztf_id=host_ztf_id_to_swap_in,
            theorized_lightcurve_df=None,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            path_to_dataset_bank=path_to_dataset_bank,
            save_timeseries=False,
            building_for_AD=True,
            swapped_host=True,
        )

        host_values = swapped_host_timeseries_df[host_features].iloc[0]
        for col in host_features:
            timeseries_df[col] = host_values[col]

    timeseries_df_filt_feats = timeseries_df[lc_features + host_features]
    input_lightcurve_locus = antares_client.search.get_by_ztf_object_id(
        ztf_object_id=transient_ztf_id
    )

    tns_name, tns_cls, tns_z = re_getTnsData(transient_ztf_id)

    re_check_anom_and_plot(
        clf=clf,
        input_ztf_id=transient_ztf_id,
        swapped_host_ztf_id=host_ztf_id_to_swap_in,
        input_spec_cls=tns_cls,
        input_spec_z=tns_z,
        anom_thresh=50,
        timeseries_df_full=timeseries_df,
        timeseries_df_features_only=timeseries_df_filt_feats,
        ref_info=input_lightcurve_locus,
        savefig=save_figures,
        figure_path=path_to_figure_directory,
    )
    return


def re_LAISS(
    path_to_dataset_bank,
    path_to_timeseries_folder="../timeseries",
    save_timeseries=True,
    transient_ztf_id=None,  # transient on which to run laiss
    theorized_lightcurve_df=None,  # optional, if provided will be used as a lightcurve instead of the transient_ztf_id
    host_ztf_id_to_swap_in=None,  # will swap the host galaxy of the input transient/theorized lightcurve to this transient's host
    host_feature_names=[],  # Leave blank for lightcurve-only LAISS
    lc_feature_names=[],  # Leave blank for host-only LAISS
    path_to_sfd_data_folder="../data/sfddata-master",  # to correct extracted magnitudes for dust; not needed if transient_ztf_id in dataset bank
    use_pca=False,
    num_pca_components=15,  # Only matters if use_pca = True
    force_recreation_of_annoy_index=False,  # Rebuild indexed space for ANNOY even if it already exists
    path_to_index_directory="../annoy_indices",  # folder to store ANNOY indices
    neighbors=10,  # will return this number of neighbors unless filtered by max_neighbor_distance
    num_mc_simulations=0,  # set to 0 to turn off simulation. If not using pca, set to 20. Not reccomended for use with pca.
    suggest_neighbor_num=False,  # plot distances of neighbors to help choose optimal neighbor number. If true, will stop nearest nearest neighbors and return nearest_neighbors_df, primer_dict but nearest_neighbors_df will be None.
    max_neighbor_distance=None,  # optional, will return all neighbors below this distance (but no more than the 'neighbors' argument)
    search_k=5000,  # for ANNOY search
    weight_lc_feats_factor=1,  # Makes lightcurve features a larger contributor to distance. Setting to 1 does nothing.
    run_AD=True,  # run anomaly detection
    run_NN=True,  # Run nearest neighbors. Will get cut off if suggest_neighbor_num=True.
    path_to_models_directory="../models",
    path_to_figure_directory="../figures",
    n_estimators=500,  # AD model param
    contamination=0.02,  # AD model param
    max_samples=1024,  # AD model param
    force_AD_retrain=False,  # Retrains and saves AD model even if it already exists
    save_figures=True,  # Saves all figures while running LAISS
):
    """High-level convenience wrapper: build index → NN search → AD.

    Combines the *primer*, *nearest-neighbours*, and *anomaly-detection*
    pipelines with many toggles for experimentation.

    Parameters
    ----------
    transient_ztf_id : str | None
    theorized_lightcurve_df : pandas.DataFrame | None
    host_ztf_id_to_swap_in : str | None
    lc_feature_names, host_feature_names : list[str]
    neighbors : int
        Target neighbour count.
    suggest_neighbor_num : bool
        Show elbow plot instead of full NN run.
    run_NN, run_AD : bool
        Enable/disable each pipeline stage.
    *Other params*
        See lower-level helpers for details.

    Returns
    -------
    (pandas.DataFrame | None, dict | None)
        Neighbours table and primer dict when NN stage executed; otherwise
        *None*.
    """
    if run_NN or suggest_neighbor_num:
        # build ANNOY indexed sample from dataset bank
        index_stem_name_with_path = re_build_indexed_sample(
            dataset_bank_path=path_to_dataset_bank,
            lc_features=lc_feature_names,
            host_features=host_feature_names,
            use_pca=use_pca,
            n_components=num_pca_components,
            num_trees=1000,
            path_to_index_directory=path_to_index_directory,
            save=True,
            force_recreation_of_index=force_recreation_of_annoy_index,
            weight_lc_feats_factor=weight_lc_feats_factor,
        )

        # run primer
        primer_dict = re_LAISS_primer(
            lc_ztf_id=transient_ztf_id,
            theorized_lightcurve_df=theorized_lightcurve_df,
            host_ztf_id=host_ztf_id_to_swap_in,
            dataset_bank_path=path_to_dataset_bank,
            path_to_timeseries_folder=path_to_timeseries_folder,
            path_to_sfd_data_folder=path_to_sfd_data_folder,
            lc_features=lc_feature_names,
            host_features=host_feature_names,
            num_sims=num_mc_simulations,
            save_timeseries=save_timeseries,
        )

        nearest_neighbors_df = re_LAISS_nearest_neighbors(
            primer_dict=primer_dict,
            path_to_dataset_bank=path_to_dataset_bank,
            theorized_lightcurve_df=theorized_lightcurve_df,
            annoy_index_file_stem=index_stem_name_with_path,
            use_pca=use_pca,
            num_pca_components=num_pca_components,
            n=neighbors,
            suggest_neighbor_num=suggest_neighbor_num,
            max_neighbor_dist=max_neighbor_distance,
            search_k=search_k,
            weight_lc_feats_factor=weight_lc_feats_factor,
            save_figures=save_figures,
            path_to_figure_directory=path_to_figure_directory,
        )

    if run_AD:
        if theorized_lightcurve_df is not None:
            print("Cannot run anomaly detection on theorized lightcurve. Skipping.")
        else:
            re_anomaly_detection(
                transient_ztf_id=transient_ztf_id,
                host_ztf_id_to_swap_in=host_ztf_id_to_swap_in,
                lc_features=lc_feature_names,
                host_features=host_feature_names,
                path_to_timeseries_folder=path_to_timeseries_folder,
                path_to_sfd_data_folder=path_to_sfd_data_folder,
                path_to_dataset_bank=path_to_dataset_bank,
                path_to_models_directory=path_to_models_directory,
                path_to_figure_directory=path_to_figure_directory,
                save_figures=save_figures,
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples=max_samples,
                force_retrain=force_AD_retrain,
            )

    if run_NN or suggest_neighbor_num:
        return nearest_neighbors_df, primer_dict

    return


def re_corner_plot(
    neighbors_df,  # from reLAISS nearest neighbors
    primer_dict,  # from reLAISS nearest neighbors
    path_to_dataset_bank,
    remove_outliers_bool=True,
    path_to_figure_directory="../figures",
    save_plots=True,
):
    """Corner-plot visualisation of feature distributions vs. neighbours.

    Parameters
    ----------
    neighbors_df : pandas.DataFrame
        Output from :func:`re_LAISS_nearest_neighbors`.
    primer_dict : dict
        Output from :func:`re_LAISS_primer`.
    path_to_dataset_bank : str | Path
    remove_outliers_bool : bool, default True
        Apply robust MAD clipping before plotting.
    save_plots : bool, default True
        Write PNGs to ``corner_plots/``.

    Returns
    -------
    None
    """
    if primer_dict is None:
        raise ValueError(
            "primer_dict is None. Try running NN search with reLAISS again."
        )
    if neighbors_df is None:
        raise ValueError(
            "neighbors_df is None. Try running reLAISS NN search again using run_NN=True, suggest_neighbor_num=False to get correct object."
        )

    lc_feature_names = primer_dict["lc_feat_names"]
    host_feature_names = primer_dict["host_feat_names"]

    if save_plots:
        os.makedirs(path_to_figure_directory, exist_ok=True)
        os.makedirs(path_to_figure_directory + "/corner_plots", exist_ok=True)

    logging.getLogger().setLevel(logging.ERROR)

    re_laiss_features_dict = create_re_laiss_features_dict(
        lc_feature_names, host_feature_names
    )

    neighbor_ztfids = [link.split("/")[-1] for link in neighbors_df["ztf_link"]]

    dataset_bank_df = pd.read_csv(path_to_dataset_bank)[
        ["ztf_object_id"] + lc_feature_names + host_feature_names
    ]
    print("Total number of transients for corner plots:", dataset_bank_df.shape[0])

    for batch_name, features in re_laiss_features_dict.items():
        print(f"Creating corner plot for {batch_name}...")

        # REMOVING OUTLIERS #
        def remove_outliers(df, threshold=7):
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                col_data = df_clean[col]
                median_val = col_data.median()
                mad_val = robust.mad(
                    col_data
                )  # By default uses 0.6745 scale factor internally

                # If MAD is zero, it means the column has too little variation (or all same values).
                # In that case, skip it to avoid removing all rows.
                if mad_val == 0:
                    continue

                # Compute robust z-scores
                robust_z = 0.6745 * (col_data - median_val) / mad_val

                # Keep only points where the robust z-score is within the threshold
                df_clean = df_clean[abs(robust_z) <= threshold]

            return df_clean

        dataset_bank_df_batch_features = dataset_bank_df[["ztf_object_id"] + features]

        if remove_outliers_bool:
            dataset_bank_df_batch_features = remove_outliers(
                dataset_bank_df_batch_features
            )
            print(
                "Total number of transients for corner plot after outlier removal:",
                dataset_bank_df_batch_features.shape[0],
            )
        else:
            dataset_bank_df_batch_features = dataset_bank_df_batch_features.replace(
                [np.inf, -np.inf, -999], np.nan
            ).dropna()
            print(
                "Total number of transients for corner plot after NA, inf, and -999 removal:",
                dataset_bank_df_batch_features.shape[0],
            )
        # REMOVING OUTLIERS #
        neighbor_mask = dataset_bank_df_batch_features["ztf_object_id"].isin(
            neighbor_ztfids
        )
        features_df = dataset_bank_df_batch_features[features]

        # remove 'feature_' from column names
        features_df.columns = [
            col.replace("feature_", "", 1) if col.startswith("feature_") else col
            for col in features_df.columns
        ]

        neighbor_features = features_df[neighbor_mask]
        non_neighbor_features = features_df[~neighbor_mask]

        col_order = lc_feature_names + host_feature_names
        queried_transient_feat_df = pd.DataFrame(
            [primer_dict["locus_feat_arr"]], columns=col_order
        )
        queried_features_arr = queried_transient_feat_df[features].values[0]

        figure = corner.corner(
            non_neighbor_features,
            color="blue",
            labels=features_df.columns,
            plot_datapoints=True,
            alpha=0.3,
            plot_contours=False,
            truths=queried_features_arr,
            truth_color="green",
        )

        # Overlay neighbor features (red) with larger, visible markers
        axes = np.array(figure.axes).reshape(len(features), len(features))
        for i in range(len(features)):
            for j in range(i):  # Only the lower triangle of the plot
                ax = axes[i, j]
                ax.scatter(
                    neighbor_features.iloc[:, j],
                    neighbor_features.iloc[:, i],
                    color="red",
                    s=10,
                    marker="x",
                    linewidth=2,
                )

        if save_plots:
            plt.savefig(
                path_to_figure_directory + f"/corner_plots/{batch_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    if save_plots:
        print("Corner plots saved to" + path_to_figure_directory + f"/corner_plots")
    else:
        print("Finished creating all corner plots!")
    return
