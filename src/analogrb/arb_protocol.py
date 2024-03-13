import numpy as np
import pandas as pd
import os

from scipy.special import binom
from analogrb.projector import AllProjectors
from analogrb.bootstrap import fit_with_bootstrap
from copy import deepcopy


from analogrb.save_load import (
    create_dir_arb,
    save_meta_data_arb,
    save_datarow,
    load_data,
    load_meta_data,
    load_datarow,
    extract_from_data,
)


def calculate_normalization(d: int, n: int, interacting: bool, rho_init: np.ndarray):
    rho_vec = rho_init.reshape(1, -1)
    dF = int(binom(n + d - 1, n))
    states = np.eye(dF).reshape(dF, 1, dF)
    mv = np.einsum("ijk,ilm->ijklm", states, states).reshape(dF, -1)
    m = 0
    for v in mv:
        m += np.outer(v, v)
    ps = AllProjectors(d, n, interacting)
    ns = [
        np.trace(p.projector @ m)
        / np.trace(p.projector)
        * (rho_vec @ p.projector @ rho_vec.reshape(-1, 1))
        for p in ps
    ]
    return np.array(ns).flatten()


def filter_function(projector, U, rho_init):
    projected_rho = projector @ (U @ rho_init @ U.conj().T).reshape(-1, 1)
    return np.diag(projected_rho.reshape(rho_init.shape))


def filter_helper(all_projectors, U, rho_init, outcomes, ns):
    filtered_dict = {}
    for count, p in enumerate(all_projectors):
        filtered_dict[f"q_{p.name}"] = (
            np.real(
                filter_function(p.projector, U.reshape(rho_init.shape), rho_init)
                @ outcomes
            )
            / ns[count]
        )
    return filtered_dict


def filterf(datarow, rho_init, all_projectors):
    for p in all_projectors:
        datarow[f"q_{p.name}"] = np.real(
            filter_function(p.projector, datarow["U"].reshape(rho_init.shape), rho_init)
            @ datarow["outcomes"]
        )
    return datarow


def polish(datarow):
    datarow["m"] = int(np.real(datarow["m"]))
    datarow["outcomes"] = np.real(datarow["outcomes"])
    datarow["U"] = datarow["U"].reshape(int(np.sqrt(len(datarow["U"]))), -1)
    return datarow


def acquire(length, gatefactory, rho_init, error_channel):
    # Reset rho to be the initial state for every new sequence length.
    rho = deepcopy(rho_init)
    # Reset the storage for the applied gates.
    gates_product = np.eye(len(rho), dtype=np.complex128)
    # Apply a gate from gatefactory length many times.
    for _ in range(length):
        gate = next(gatefactory)
        if isinstance(gate, tuple):
            gates_product = gate[0] @ gates_product
            rho = gate[1] @ rho @ gate[1].conj().T
        else:
            gates_product = gate @ gates_product
            rho = gate @ rho @ gate.conj().T
        rho = error_channel.apply(rho)
    return {
        "m": length,
        "outcomes": np.real(np.diag(rho)),
        "U": gates_product,
    }


def simulate_data_acquisition(
    d, n, interacting_gates, ms, gatefactory, rho_init, error_channel, naverage, save
):
    if save:
        path = create_dir_arb(d, n, interacting_gates)
        save_meta_data_arb(
            path,
            d,
            n,
            interacting_gates,
            naverage,
            gatefactory,
            rho_init,
            error_channel,
        )

    data = []
    for k in range(naverage):
        for m in ms:
            datarow_dict = acquire(m, gatefactory, rho_init, error_channel)
            save_datarow(path, datarow_dict) if save else data.append(datarow_dict)
        print(f"{k + 1} out of {naverage} repetitions done")
    return path if save else data


def post_processing(
    path=None, data=None, d=None, n=None, rho_init=None, interacting=None
):
    if path is not None:
        metadata = load_meta_data(path)
        rho_init = np.array(metadata["rho_init_real"]) + (
            np.array(metadata["rho_init_imag"])
            if metadata["rho_init_imag"] is not None
            else 0
        )
        interacting = metadata["interacting"] if interacting is None else interacting
        d = metadata["d"]
        n = metadata["n"]
    if data is None and path is not None:
        data = load_data(path)

    all_projectors = AllProjectors(d, n, interacting)
    for datarow in data:
        filterf(datarow, rho_init, all_projectors)
        polish(datarow)
    ns = calculate_normalization(d, n, interacting, rho_init)
    df = pd.DataFrame(data)
    for count, p in enumerate(all_projectors):
        df[f"q_{p.name}"] /= ns[count]
    return df.to_dict("records")


def postprocess(path):
    import warnings

    warnings.filterwarnings("error")

    metadata = load_meta_data(path)
    rho_init = np.array(metadata["rho_init_real"]) + (
        np.array(metadata["rho_init_imag"])
        if metadata["rho_init_imag"] is not None
        else 0
    )
    interacting = metadata["interacting"]
    d = metadata["d"]
    n = metadata["n"]
    all_projectors = AllProjectors(d, n, interacting)
    ns = calculate_normalization(d, n, interacting, rho_init)

    ending = ".txt"
    txt_filenames = [
        file[: -len(ending)]
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.endswith(ending)
    ]
    data, lastline, k = [], False, 0
    while not lastline:
        try:
            datarow = load_datarow(path, txt_filenames, k)
            mydict = filter_helper(
                all_projectors, datarow["U"], rho_init, datarow["outcomes"], ns
            )
            save_datarow(path, mydict)
            mydict["m"] = int(np.real(datarow["m"]))
            data.append(mydict)
            k += 1
        except UserWarning:
            lastline = True
    warnings.resetwarnings()
    return data


def aggregate(data_path, start_fitting, nbootstraps, confidence, nshots="inf"):
    df = pd.DataFrame(load_data(data_path, dont_load=["U", "outcomes"]))
    df["m"] = df["m"].astype(int)
    ms = np.array(df["m"], dtype=int)
    msmax = np.max(ms)
    foundit = False
    k = len(ms)
    while not foundit:
        if ms[k - 1] != msmax:
            k -= 1
        else:
            foundit = True
    df = df.iloc[:k]
    # Extract the names of the irreps for the different decay curves for each irrep.
    if nshots != "inf":
        names = np.sort(
            [
                key
                for key in df.keys()
                if key.startswith("q_") and key.endswith(f"nshots{nshots}")
            ]
        )
    else:
        names = np.sort(
            [key for key in df.keys() if key.startswith("q_") and "nshots" not in key]
        )
    # Initialize a new data frame to store the results.
    df_result = pd.DataFrame(
        columns=[
            "m",
            "avg",
            "yerr",
            "popt",
            "pcov",
            "niter",
            "nbootstraps",
            "confidence",
        ],
        index=names,
    )
    # Go through all the irrep data.
    for name in names:
        df[name] = df[name].astype(float)
        ms, qs = extract_from_data(df, name, "m", list)
        df_result.loc[name] = (
            *fit_with_bootstrap(ms, qs, nbootstraps, confidence),
            [len(q) for q in qs],
            nbootstraps,
            confidence,
        )

    df_result_capped = pd.DataFrame(
        columns=[
            "m",
            "avg",
            "yerr",
            "popt",
            "pcov",
            "niter",
            "mthresh",
            "nbootstraps",
            "confidence",
        ],
        index=names,
    )
    # Go through all the irrep data.
    for name in names:
        df[name] = df[name].astype(float)
        ms, qs = extract_from_data(df, name, "m", list)
        fitting_index = np.argwhere(ms >= start_fitting)[0, 0]
        ms_capped = ms[fitting_index:]
        qs_capped = np.array(qs)[fitting_index:]
        df_result_capped.loc[name] = (
            *fit_with_bootstrap(ms_capped, qs_capped, nbootstraps, confidence),
            [len(q) for q in qs],
            start_fitting,
            nbootstraps,
            confidence,
        )
    return df_result, df_result_capped, names
