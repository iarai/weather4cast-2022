# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
#
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


from re import VERBOSE
import numpy as np
import pandas as pd
import yaml
import h5py
import pickle
from torch.utils.data import DataLoader
import time
import torch
import os

from datetime import datetime, timedelta

VERBOSE = False
# VERBOSE = True


# __________________________________________________CREATING/LOADING SAMPLE IDS____________________________________________________


def load_sample_ids(
    data_split, splits_df, len_seq_in, len_seq_predict, regions, generate_pkl, years, path=""
):
    """For loading the sample idxs of the dataset. If a pkl file is found, it will be loaded. If not, it will be generated.
    If you want to save a .pkl file, set generate_pkl to True in the yaml file.

    Args:
        data_split (string): The data split to load the sample idxs for.
        splits_df (DataFrame): The dataframe specifying what split each timepoint belongs to. Used to generate samples
        len_seq_in (int): The length of the input sequence.
        len_seq_predict (int): The length of the prediction sequence.
        regions (list): the regions to create samples for
        generate_pkl (bool): If True, a pkl file will be generated. If False, a pkl file will be loaded.
        path (str, optional): The path to load sample idxs from. Defaults to ''.

    Returns:
        dict: A dictionary of the sample idxs.
    """
    
    print("|YEARS]", years)
    
    # if generate_pkl:
    #     idxs_r = generate_and_cache_sequences(
    #         data_split, splits_df, len_seq_in, len_seq_predict, regions, years, path
    #     )
    # # elif ".pkl" in path:  # read pre-computed (fastest)
    # #     print("load_sample_ids using pkl cache:", path)
    # #     idxs_r = read_samples_ids(path, data_split)
    # else:
    idxs_r = generate_sample_ids(
        data_split, splits_df, len_seq_in, len_seq_predict, regions, years
    )
    return idxs_r


def generate_and_cache_sequences(
    split, splits_df, len_seq_in, len_seq_predict, regions, path_name, years
):
    """Generates and saves the the sample idxs for the dataset. This generates the sample idxs for all data splits.

    Args:
        split (string): The data split to return the sample idxs for.
        splits_df (DataFrame): The dataframe specifying what split each timepoint belongs to. Used to generate samples
        len_seq_in (_type_): The length of the input sequence.
        len_seq_predict (_type_): The length of the prediction sequence.
        regions (_type_): The regions to create samples for
        path_name (_type_): The path to save the sample idxs to.

    Returns:
        list: A list of the sample idxs.
    """

    samples = {"training": [], "validation": [], "test": []}

    for data_split in ["training", "validation", "test"]:
        idxs = generate_sample_ids(
            data_split, splits_df, len_seq_in, len_seq_predict, regions, years
        )
        samples[data_split] = idxs
        if VERBOSE:
            print(f"{len(idxs)} {data_split} samples")

    qq = samples.copy()
    with open(path_name, "wb") as f:
        pickle.dump(qq, f)
    return samples[split]


def read_samples_ids(path, data_split):
    """read in sample idxs if they are already generated

    Args:
        path (String): path to pkl file with sample idxs
        data_split (String): data split to load sample idxs for

    Returns:
        list: list of sample idxs
    """
    with open(path, "rb") as f:
        loaded_dict = pickle.load(f)
        return loaded_dict[data_split]


def generate_sample_ids(data_split, splits_df, len_seq_in, len_seq_predict, regions, years):
    """generates sample idxs for a given data split

    Args:
        data_split (String): data split to load sample idxs for
        splits_df (DataFrame): the dataframe specifying what split each timepoint belongs to. Used to generate samples
        len_seq_in (int): the length of the input sequence.
        len_seq_predict (int): the length of the prediction sequence.
        regions (list): the regions to create samples for

    Returns:
        list: list of sample idxs for a specific data split
    """
    if data_split == "training":
        idxs_r = get_training_idxs(
            splits_df, len_seq_in, len_seq_predict, data_split, regions[0], years
        )
        for r in range(1, len(regions)):  # append rest of regions to df
            idxs = get_training_idxs(
                splits_df, len_seq_in, len_seq_predict, data_split, regions[r], years
            )
            idxs_r = idxs_r + idxs

    elif data_split == "validation":  # Validation/Testing/Heldout Data
        idxs_r = get_validation_idxs(
            splits_df, len_seq_in, len_seq_predict, data_split, regions[0], years
        )
        for r in range(1, len(regions)):  # append rest of regions to df
            idxs = get_validation_idxs(
                splits_df, len_seq_in, len_seq_predict, data_split, regions[r], years
            )
            idxs_r = idxs_r + idxs
    else:  # testing
        idxs_r = get_test_heldout_idxs(splits_df, len_seq_in, data_split, regions[0], years)
        for r in range(1, len(regions)):  # append rest of regions to df
            idxs = get_test_heldout_idxs(splits_df, len_seq_in, data_split, regions[r], years)
            idxs_r = idxs_r + idxs
    return idxs_r


def get_training_idxs(df, len_seq_in, len_seq_predict, data_split, region, years):
    """get sample idxs for training data split
    Args:
        df (DataFrame): _description_
        len_seq_in (int): length of input sequence
        len_seq_predict (int): length of prediction sequence
        data_split (String): data split to load sample idxs for
        region (String): region to create samples for
    Returns:
        list: list of sample idxs for training data split
    """
    idxs = []

    df = df[df['split'] == data_split]
    df = df[df['all_vars'] == 1]
    dfs= pd.DataFrame(columns = df.columns);
    
    print("years //// ", years)
    for year in years:
        dfs=pd.concat([dfs,df[df['date'].dt.year==int(year)]]);
    if VERBOSE:
        print(f"Keeping {dfs.shape[0]} {data_split} rows of {df.shape[0]} for {region} in {years}");
    df=dfs;

    # non testing
    for i in range(df.shape[0] - len_seq_in - len_seq_predict + 1):
        # date check

        s_id = df.iloc[i]["date_time_str"]
        e_id = df.iloc[i + len_seq_in + len_seq_predict - 1]["date_time_str"]

        dd1, mm1, yy1 = get_day_month_year(s_id)
        h1, m1, s1 = get_hours_minutes_seconds(s_id)
        start_dt = datetime(yy1, mm1, dd1, h1, m1, s1)

        dd2, mm2, yy2 = get_day_month_year(e_id)
        h2, m2, s2 = get_hours_minutes_seconds(e_id)
        end_dt = datetime(yy2, mm2, dd2, h2, m2, s2)

        if start_dt + timedelta(hours=8, minutes=45, seconds=0) == end_dt:
            # print(get_future_time(df.iloc[i]['time'], (len_seq_predict * 15)))
            in_seq = [i + j for j in range(len_seq_in)]
            out_seq = [i + len_seq_in + j for j in range(len_seq_predict)]
            idxs.append([in_seq, out_seq, region])

    return idxs


def get_test_heldout_idxs(df, len_seq_in, data_split, region, years):
    """get sample idxs for test data split

    Args:
        df (DataFrame): _description_
        len_seq_in (int): length of input sequence
        len_seq_predict (int): length of prediction sequence
        data_split (String): data split to load sample idxs for
        region (String): region to create samples for

    Returns:
        list: list of sample idxs for test data split
    """
    idxs = []

    split_type = f"{data_split}_in"
    df = df[df["split_type"] == split_type]
    df = df[df["all_vars"] == 1]
    dfs= pd.DataFrame(columns = df.columns);
    for year in years:
        dfs=pd.concat([dfs,df[df['date'].dt.year==int(year)]]);
    if VERBOSE:
        print(f"Keeping {dfs.shape[0]} {data_split} rows of {df.shape[0]} for {region} in {years}");
    df=dfs;


    for start_index in range(0, df.shape[0], len_seq_in):
        in_seq = [start_index + i for i in range(len_seq_in)]
        test_seq = [in_seq, [], region]
        idxs.append(test_seq)
    return idxs


def get_validation_idxs(df, len_seq_in, len_seq_out, data_split, region, years):
    """get sample idxs for validation data split

    Args:
        df (DataFrame): timepoint dataframe
        len_seq_in (int): length of input sequence
        len_seq_predict (int): length of prediction sequence
        data_split (String): data split to load sample idxs for
        region (String): region to create samples for

    Returns:
        list: list of sample idxs for validation data split
    """
    idxs = []
    split_type = f"{data_split}_in"
    df = df[df["split"] == data_split]
    df = df[df["all_vars"] == 1]
    dfs= pd.DataFrame(columns = df.columns);
    for year in years:
        dfs=pd.concat([dfs,df[df['date'].dt.year==int(year)]]);
    if VERBOSE:
        print(f"Keeping {dfs.shape[0]} {data_split} rows of {df.shape[0]} for {region} in {years}");
    df=dfs;
    
    for i in range(df.shape[0]):
        if df.iloc[i]["split_type"] == split_type:
            input_output_seq = get_io_validation_times(
                df, i, len_seq_in, len_seq_out, region
            )
            if input_output_seq:
                idxs.append(input_output_seq)
    return idxs


def get_io_validation_times(df, start_index, len_seq_in, len_seq_predict, region):
    """get input and output times for validation data split. Checks for valid sequences

    Args:
        df (int): timepoint dataframe
        start_index (int): startng index of sequence
        len_seq_in (int): length of input sequence
        len_seq_predict (int): length of prediction sequence
        region (String): region to create sequences for

    Returns:
        list: list of input, output times and region for validation data split
    """
    split_type_in = f"validation_in"
    end_index = start_index + len_seq_in + len_seq_predict - 1
    if end_index < len(df):
        seq = df[start_index:end_index]
        in_seq = seq[:len_seq_in]
        out_seq = seq[len_seq_in:]
        # check all input is of the same type
        if len(in_seq[in_seq["split_type"] == split_type_in]) != len(in_seq):
            return []
        # convert input/output sequence to lists and return
        else:
            in_seq = [start_index + i for i in range(len_seq_in)]
            out_seq = [start_index + len_seq_in + i for i in range(len_seq_predict)]
            return [in_seq, out_seq, region]
    else:
        return []


# ______________________________________LOADING DATA_______________________________________


def load_dataset(root, data_split, regions, years, product):
    """load dataset from root folder and return as list
    Args:
        root (String): root folder to load dataset from
        data_split (String): data split to load dataset for
        regions (list): list of regions to load dataset for
        product (String): product to load dataset for e.g. satellite or radar
    Returns:
        list : full dataset
    """
    dataset = {}
    if data_split == "validation":
        data_split = "val"
    elif data_split == "training":
        data_split = "train"
        
    if VERBOSE:
        print(f"Data split {data_split}")
        print(f"Regions {regions}")
        print(f"Years {years}")
        print(f"Product {product}")
    
    for region in regions:
        if VERBOSE: print(f"Region: {region}")
        yearly_records = {}
        for year in years:
            if VERBOSE: print(f"Year: {year}")
            if product == "RATE":
                file = f"{region}.{data_split}.rates.crop.h5"
                path = f"{root}/{year}/OPERA/{file}"
                f = h5py.File(path, "r")
                ds = f["rates.crop"]
                yearly_records[year] = ds
                # f.close()
            else:
                file = f"{region}.{data_split}.reflbt0.ns.h5"
                path = f"{root}/{year}/HRIT/{file}"
                f = h5py.File(path, "r")
                ds = f["REFL-BT"]
                yearly_records[year]  = ds
                # f.close()
                # close file memory leak?
                # when done reading, close files?
                
        if VERBOSE: print(f"Done reading {region}... skipping concatenation")
        dataset[region] = yearly_records
        
    if VERBOSE: print(f"{data_split} Data read ... returning to dataset.)")
    if VERBOSE: print(f"Read {len(dataset)} regions, with each containing {len(dataset[list(dataset.keys())[0]])} years.")
    return dataset 


def get_sequence(
    seq,
    root,
    data_split,
    region,
    product,
    bands,
    preprocess=None,
    swap_time_ch=False,
    ds=None,
):
    """get data and metadata for a given sequence.

    Args:
        seq (list): list containing sequence of idxs to be retrieved
        root (String): root for data
        data_split (String): data split to load sequence for
        region (String): region to load sequence for
        product (String): product to load sequence for e.g. satellite or radar
        bands (list): list of bands to load sequence for
        preprocess (list, optional): preprocessing settings. Defaults to None.
        swap_time_ch (bool, optional): swap time and channel axis if set to True. Defaults to False.
        ds (numpy array, optional): full dataset to read from. Defaults to None.
    Returns:
        prod_seq: sequence of data and metadata
        mask_seq: corresponding sequence of masks
    """

    prod_seq = []
    mask_seq = []
    for s in seq:
        prods, masks = get_file(
            s, root, data_split, region, product, bands, preprocess, ds
        )
        prod_seq.append(prods)
        mask_seq.append(masks)
    # return format - time x channels x width x height
    mask_seq = np.asarray(mask_seq)

    # Swapping Axes to give shape  channels x time x width x height
    if swap_time_ch:
        prod_seq = np.swapaxes(prod_seq, 0, 1)
        mask_seq = np.swapaxes(mask_seq, 0, 1)
    return np.array(prod_seq), mask_seq


def get_file(
    sample_id, root, data_split, region, product, bands, preprocess=None, ds=None
):
    """Read/Preprocess data and metadata for single timepoint.

    Args:
        sample_id (int): idx to be retrieved
        root (String): root for data
        data_split (String): data split to load data point for
        region (String): region to load data point for
        product (String): product to load data point for e.g. satellite or radar
        bands (list): satellite bands to load data point for
        preprocess (list, optional): preprocessing settings. Defaults to None.
        ds (numpy array, optional): full dataset to read from. Defaults to None.

    Returns:
        x(numpy array): data for single timepoint
        masks(numpy array): masks for single timepoint
    """

    if VERBOSE:
        print("\n", "_____________READING FILE_________________________")
        file_t = time.time()
    # Get file containing all neede channels for a given time - (1xCxWxH)
    prods = []
    masks = []

    x = read_file(sample_id, ds, region)
    x = np.float32(x)
    if VERBOSE:
        print(time.time() - file_t, "reading file time")
    if product == "RATE":
        for b in ["rainfall_rate-500X500"]:
            if preprocess[product][b]["mask"]:
                mask = create_mask(x, preprocess[product][b]["mask"])
                masks.append(mask)
                masks = masks[0]
            if preprocess is not None:
                x = preprocess_OPERA(x, preprocess[product][b])
            if VERBOSE:
                print(time.time() - file_t, "OPERA preprocess time")
    else:
        for j, b in enumerate(bands):
            if preprocess is not None:
                x[j] = preprocess_HRIT(x[j], preprocess[b])
            if VERBOSE:
                print(time.time() - file_t, "HRIT preprocess time")

    # return format - channels x width x height
    if VERBOSE:
        print(time.time() - file_t, "Total File Read Time")
    return x, masks


def read_file(sample_id, ds, region):
    previous_year_samples = 0
    for year in ds[region]:
        # print(sample_id, previous_year_samples)
        i = sample_id - previous_year_samples;
        if i < len(ds[region][year]):
            return ds[region][year][i]
        previous_year_samples += len(ds[region][year])
    
    raise Exception(f"Sample {sample_id} not found for {region}")

# ___________________PREPROCESSING FUNCTIONS_______________________________________


def crop_numpy(x, crop):
    """crop numpy array

    Args:
        x (numpy array): array to be cropped
        crop (int): crop size

    Returns:
        data(numpy array): cropped array
    """
    return x[:, :, crop:-crop, crop:-crop]


def preprocess_fn(x, preprocess, verbose=False):
    """Funciton to precprocess data

        Assumption: the mask has been previously created (otherwise won't be recovered)
        # 1. map values
        # 2. clip values out of range
    Args:
        x (numpy array): data to be preprocessed
        preprocess (list): preprocessing settings
        verbose (bool, optional): verbose preprocessing. Defaults to False.

    Returns:
        data(numpy array): preprocessed data
    """

    if verbose:
        print(0, np.unique(x))
    # 1 Map values
    for q, v in preprocess["map"]:

        if isinstance(q, str):
            if q == "nan":
                x[np.isnan(x)] = v
            elif q == "inf":
                x[np.isinf(x)] = v
            elif "greaterthan" in q:
                greater_v = float(q.partition("greaterthan")[2])
                x[x > greater_v] = v
            elif "lessthan" in q:
                less_v = float(q.partition("lessthan")[2])
                x[x < less_v] = v
        else:
            x[x == q] = v
    if verbose:
        print(1, np.unique(x, return_counts=True))

    # 2 Clip values out of range
    m, M = preprocess["range"]
    x[x < m] = m
    x[x > M] = M
    if verbose:
        print(2, np.unique(x, return_counts=True))

    return x, M


def preprocess_HRIT(x, preprocess, verbose=False):
    """Preprocess HRIT data - standardize data, map values, clip values out of range

        Assumption: the mask has been previously created (otherwise won't be recovered)
    Args:
        x (numpy array): data to be preprocessed
        preprocess (list): preprocessing settings
        verbose (bool, optional): verbose preprocessing. Defaults to False.

    Returns:
        x(list): preprocessed data
    """
    # 1, 2
    if verbose:
        print("HRIT file:")
    x, M = preprocess_fn(x, preprocess, verbose=verbose)

    # 3 - mean_std - standardisation
    if preprocess["standardise"]:
        # x = x/M
        mean, stdev = preprocess["mean_std"]
        x = x - mean
        x = x / stdev
    if verbose:
        print(3, np.uniquze(x, return_counts=True))
    return x


def preprocess_OPERA(x, preprocess, verbose=False):
    """Precprocess OPERA data - standardize data, map values, clip values out of range
        Assumption: the mask has been previously created (otherwise won't be recovered)
        # 1. map values
        # 2. clip values out of range

    Args:
        x (numpy array): data to be preprocessed
        preprocess (list): preprocessing settings
        verbose (bool, optional): verbose preprocessing. Defaults to False.

    Returns:
        x(numpy array): preprocessed data
    """
    # 1, 2
    if verbose:
        print("OPERA file:")
    x, M = preprocess_fn(x, preprocess, verbose=verbose)

    # 3 - mean_std - standardisation
    if preprocess["standardise"]:
        mean, stdev = preprocess["mean_std"]
        x = x - mean
        x = x / stdev
    if preprocess["bin"]:
        bins = np.arange(0, 128, 0.2)
        x = np.digitize(x, bins)
    if verbose:
        print(3, np.unique(x, return_counts=True))
    return x


# __________________________________GENERAL HELPER FUNCTIONS__________________________________


def standardise_time_strings(time):
    if len(time) < 6:
        time = time.rjust(6, "0")
    else:
        return time
    return time


def get_hours_minutes_seconds(date_time_str):
    h = date_time_str[9:11]
    m = date_time_str[11:13]
    s = date_time_str[13:15]
    return int(h), int(m), int(s)


def get_day_month_year(date_time_str):
    yy = date_time_str[0:4]
    mm = date_time_str[4:6]
    dd = date_time_str[6:8]
    return int(dd), int(mm), int(yy)


def time_2_channels(w, height, width):
    """collapse time dimension into channels - (B, C, T, H, W) -> (B, C*T, H, W)

    Args:
        w (numpy array): data
        height (int): image height in pixels
        width (int): image width in pixels

    Returns:
        w(numpy array): data with time dimension collapsed into channels
    """
    w = np.reshape(w, (-1, height, width))
    return w


def channels_2_time(w, seq_time_bins, n_channels, height, width):
    """Recover time dimension from channels - (B, C*T, H, W) -> (B, T, C, H, W)

    Args:
        w (numpy array): data
        seq_time_bins (_type_): number of time bins
        n_channels (int): number of channels
        height (int): image height in pixels
        width (int):  image width in pixels

    Returns:
        _type_: _description_
    """
    w = np.reshape(w, (seq_time_bins, n_channels, height, width))
    return w


def load_timestamps(
    path, types={"time": "str", "date_str": "str", "split_type": "str"}
):
    """load timestamps from a from csv file

    Args:
        path (String): path to the csv file
        types (dict, optional): types to cast columns of dataframe to. Defaults to {'time': 'str', 'date_str': 'str', 'split_type': 'str'}.

    Returns:
        Dataframe: dataframe with timestamps
    """

    df = pd.read_csv(path, index_col=False, dtype=types)

    df.sort_values(by=["date_str", "time"], inplace=True)

    # to datetime type
    df["date"] = pd.to_datetime(df["date"])

    # convert times to strings
    df["time"] = df["time"].astype(str)
    df["time"] = df["time"].apply(standardise_time_strings)

    df["date_str"] = df["date_str"].astype(str)
    # create date_time string
    df["date_time_str"] = df["date_str"] + "T" + df["time"]

    return df


def load_config(config_path):
    """Load confgiuration file

    Args:
        config_path (String): path to configuration file

    Returns:
        dict: configuration file
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def create_mask(data, mask_values, verbose=False):
    """Create a mask from a data array and a list of values to mask out
        - " 1's indicate (any kind of) missing value
    Args:
        data (numpy array): data to be masked
        mask_values (list): list of values to mask out
        verbose (bool, optional): verbose preprocessing. Defaults to False.

    Returns:
        mask(numpy array): array of mask corresponding to the data
    """

    mask = np.full(data.shape, False, dtype=bool)
    for m in mask_values:
        if isinstance(m, str):
            if m == "nan":
                filter = np.isnan(data)
            elif m == "inf":
                filter = np.isinf(data)
            elif "max" in m:
                max_v = int(m.partition("max")[2])
                filter = data > max_v
            elif "range" in m:
                range = m.partition("range")
                l = float(range[0])
                h = float(range[2])
                filter = (data >= l) & (data < h)
        else:
            filter = data == m
        mask = np.logical_or(mask, filter)

    if verbose:
        n_missing = np.unique(filter, return_counts=True)
        print("-----> total masked values:", n_missing, "\n")
    return mask


def get_mean_std(dataset, batch_size=10):
    """Get the mean and stdev of the different image bands
    Args:
        times (list): timepoints to get mean from
        batch_size (int, optional): batch size for dataloader. Defaults to 1000.
    Returns:
        mean (list): list of band means
        stdev (list): list of band stdevs
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    nimages = 0
    mean = 0
    std = 0
    count = 0
    for batch, _, _ in loader:
        print(count)
        count += 1
        # Rearrange batch to be the shape of [B, C, T * W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)

        # Compute mean of T * W * H
        # Sum over samples
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
    # Final step - divide by number of images to get average of each band
    mean /= nimages
    std /= nimages
    print(np.array(mean))
    return mean, std


def get_mean_std_opera(dataset, batch_size=10):
    """Get the mean and stdev of the different image bands
    Args:
        times (list): timepoints to get mean from
        batch_size (int, optional): batch size for dataloader. Defaults to 1000.
    Returns:
        mean (list): list of band means
        stdev (list): list of band stdevs
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    nimages = 0
    mean = 0.0
    std = 0.0
    count = 0
    for _, _, metadata in loader:
        print(count)
        count += 1
        batch = metadata["OPERA_input"]["data"]
        # Rearrange batch to be the shape of [B, C, T * W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)

        # Compute mean of T * W * H
        # Sum over samples
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    # Final step - divide by number of images to get average of each band
    mean /= nimages
    std /= nimages

    return mean, std


# ------------------------------------- GPU Functions------------------------------------------


def get_cuda_memory_usage(gpus):
    """Get the GPU memory usage

    Args:
        gpus (list): list of GPUs
    """
    for gpu in gpus:
        r = torch.cuda.memory_reserved(gpu)
        a = torch.cuda.memory_allocated(gpu)
        f = r - a  # free inside reserved
        print("GPU", gpu, "CUDA memory reserved:", r, "allocated:", a, "free:", f)


# --------------------------------------- Crop files--------------------------------------------


def crop_xarray(product, x_start, y_start, size):
    """Crop an xarray dataset
        - crop a squared region size
        - provide upper-left corner with (x_start, y_start)
    Args:
        product (xarray dataset): dataset to crop
        x_start (int): x start coordinate
        y_start (int): y start coordinate
        size (int): size of the crop
    Returns:
        xarray dataset: cropped dataset
    """
    return product.isel(
        x=slice(x_start, x_start + size), y=slice(y_start, y_start + size)
    )


def prepare_crop(location, opera_k=1):
    """prepare the crop for both context and region of interest (roi)"""

    position = np.asarray(location["up_left"])
    context, roi = location["context"], location["roi"]

    # scale if opera
    if opera_k != 1:
        context, roi, position = [opera_k * s for s in [context, roi, position]]
    context_size = context + roi + context

    context_shape = {
        "x_start": position[0],
        "y_start": position[1],
        "size": context_size,
    }
    position = position + context
    roi_shape = {"x_start": position[0], "y_start": position[1], "size": roi}
    return context_shape, roi_shape

    # --------------------------------------- Save predictions --------------------------------------------


def tensor_to_submission_file(predictions, predict_params):
    """saves prediction tesnor to submission .h5 file

    Args:
        predictions (numpy array): data cube of predictions
        predict_params (dict): dictionary of parameters for prediction
    """
    
    path = os.path.join(predict_params["submission_out_dir"],
                        str(predict_params["year_to_predict"]))
    if not os.path.exists(path):
        os.makedirs(path)
    
    submission_file_name = predict_params["region_to_predict"] + ".pred.h5"
    submission_path = os.path.join(path, submission_file_name)
    h5f = h5py.File(submission_path, "w")
    h5f.create_dataset("submission", data=predictions.squeeze())
    h5f.close()
