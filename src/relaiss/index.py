
def build_indexed_sample(
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
