from datetime import datetime
import os
import sys
import json
import numpy as np
from typing import Union, Callable
from pandas import DataFrame

def MODULE_DIR():
    # Get the directory path of the currently executing script or module
    # This may be the virtual environment directory if the script is executed from there
    # We'll use the location of the current script as a fallback if __file__ doesn't give the expected result
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # If running from within a virtual environment, __file__ might point to the site-packages directory
    # In such cases, fallback to the location of the current script
    if 'site-packages' in package_dir:
        package_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    return package_dir

# def MODULE_DIR():
#     return 'absolute/path/to/analog-rb/directory/'

PROJECTORS_DIR = lambda d, n: f"{MODULE_DIR()}/projectors/{d}modes_{n}particles/"
PROJECTORS_FILE = (
    lambda d, n, irrep3: f"{PROJECTORS_DIR(d, n)}projector_{'-'.join(map(str, irrep3))}.txt"
)

NOW = lambda: datetime.now().strftime("%y-%m-%d-%H%M%S")

ARBDIR = (
    lambda d, n, int: f"{MODULE_DIR()}/arbdata/{d}d_{n}n_{'int' if int else 'nonint'}/"
)
ARBDIR_NEW_SIMULATION = lambda d, n, int: f"{ARBDIR(d, n, int)}sim-{NOW()}/"

FPDIR = lambda d, n, int: f"{MODULE_DIR()}/fpdata/{d}d_{n}n_{'int' if int else 'nonint'}/"
FPDIR_NEW_SIMULATION = lambda d, n, int: f"{FPDIR(d, n, int)}data-{NOW()}/"

METADATA_FILENAME = "metadata.json"


def create_dir_arb(d, n, interacting):
    arb_dir = ARBDIR_NEW_SIMULATION(d, n, interacting)
    if not os.path.isdir(arb_dir):
        os.makedirs(arb_dir)
    return arb_dir


def create_dir_fp(d, n, interacting):
    fp_dir = FPDIR_NEW_SIMULATION(d, n, interacting)
    if not os.path.isdir(fp_dir):
        os.makedirs(fp_dir)
    return fp_dir


def save_meta_data_arb(
    path, d, n, interacting, naverage, gatefactory, rho_init, error_channel, **kwargs
):
    # Serializing json
    all_dict = {
        "d": d,
        "n": n,
        "interacting": interacting,
        "naverage": naverage,
        "gatefactory": gatefactory.params_to_save,
        "errorchannel": error_channel.params_to_save,
        "rho_init_real": np.real(rho_init).tolist(),
        "rho_init_imag": np.imag(rho_init).tolist()
        if np.sum(np.imag(rho_init)) > 1e-8
        else None,
        **kwargs,
    }
    json_object = json.dumps(all_dict, indent=2)

    # Writing to sample.json
    with open(f"{path}{METADATA_FILENAME}", "a") as outfile:
        outfile.write(json_object)

    print("meta data saved to ", f"{path}{METADATA_FILENAME}")


def append_meta_data(path, **kwargs):
    with open(f"{path}{METADATA_FILENAME}", "r") as f:
        existing_data = json.load(f)
    existing_data.update(kwargs)
    with open(f"{path}{METADATA_FILENAME}", "w") as f:
        json.dump(existing_data, f, indent=2)


def save_meta_data_fp(path, d, n, interacting, naverage, gatefactory,**kwargs):
    all_dict = {
        "d": d,
        "n": n,
        "interacting": interacting,
        "naverage": naverage,
        "gatefactory": gatefactory.params_to_save,
        **kwargs
    }
    json_object = json.dumps(all_dict, indent=2)

    # Writing to sample.json
    with open(f"{path}{METADATA_FILENAME}", "w") as outfile:
        outfile.write(json_object)


def load_meta_data(path):
    with open(f"{path}{METADATA_FILENAME}", "r") as f:
        existing_data = json.load(f)
    return dict(existing_data)


def save_datarow(path, datarow):
    for key in datarow.keys():
        filename = f"{path}{key}.txt"
        with open(
            filename, "ab"
        ) as file:  # Use 'ab' mode to ensure binary mode and proper newline handling
            data = datarow[key]
            np.savetxt(file, np.array([data]).reshape(1, -1), newline="\n")


def load_data(path, dont_load=[]):
    import warnings

    warnings.filterwarnings("error")
    # Filter out only the files (not directories) ending with ".txt"
    ending = ".txt"
    txt_filenames = [
        file[: -len(ending)]
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith(ending)
    ]
    txt_filenames = [name for name in txt_filenames if name not in dont_load]
    data, lastline, k = [], False, 0
    while not lastline:
        mydict = {}
        try:
            for name in txt_filenames:
                mydict[name] = np.loadtxt(
                    f"{path}{name}.txt", skiprows=k, max_rows=1, dtype=np.complex128
                )
            data.append(mydict)
            k += 1
        except UserWarning:
            lastline = True
    warnings.resetwarnings()
    return data


def load_datarow(path: str, txt_filenames: list, kline: int):
    datarow = {}
    for name in txt_filenames:
        datarow[name] = np.loadtxt(
            f"{path}{name}.txt", skiprows=kline, max_rows=1, dtype=np.complex128
        )
    return datarow


def extract_from_data(
    data: Union[list[dict], DataFrame],
    output_key: str,
    groupby_key: str = "",
    agg_type: Union[str, Callable] = "",
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return wanted values from list of dictionaries via a dataframe and its properties.

    If ``groupby_key`` given, aggregate the dataframe, extract the data by which the frame was
    grouped, what was calculated given the ``agg_type`` parameter. Two arrays are returned then,
    the group values and the grouped (aggregated) data. If no ``agg_type`` given use a linear
    function. If ``groupby_key`` not given, only return the extracted data from given key.

    Args:
        output_key (str): Key name of the wanted output.
        groupby_key (str): If given, group with that key name.
        agg_type (str): If given, calcuted aggregation function on groups.

    Returns:
        Either one or two np.ndarrays. If no grouping wanted, just the data. If grouping
        wanted, the values after which where grouped and the grouped data.
    """
    if isinstance(data, list):
        data = DataFrame(data)
    # Check what parameters where given.
    if not groupby_key and not agg_type:
        # No grouping and no aggreagtion is wanted. Just return the wanted output key.
        return np.array(data[output_key].to_numpy())
    if not groupby_key and agg_type:
        # No grouping wanted, just an aggregational task on all the data.
        return data[output_key].apply(agg_type)
    if groupby_key and not agg_type:
        df = data.get([output_key, groupby_key])
        # Sort by the groupby key for making reshaping consistent.
        df.sort_values(by=groupby_key)
        # Grouping is wanted but no aggregation, use a linear function.
        grouped_df = df.groupby(groupby_key, group_keys=True).apply(lambda x: x)
        return grouped_df[groupby_key].to_numpy(), grouped_df[output_key].to_numpy()
    df = data.get([output_key, groupby_key])
    grouped_df = df.groupby(groupby_key, group_keys=True).agg(agg_type)
    return grouped_df.index.to_numpy(), grouped_df[output_key].values.tolist()
