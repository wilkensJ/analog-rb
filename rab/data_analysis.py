from scipy.optimize import curve_fit
from typing import Union, Tuple, Optional, Callable
import numpy as np
import pandas as pd

from  matplotlib.axes import Axes
import matplotlib.pyplot as plt

ROUND = 10

def fit(func, x, ymean, **curvefit_dict):
    try:
        ypopt, ypcov = curve_fit(func, x, ymean, **curvefit_dict)
    except RuntimeError:
        ypopt, ypcov = None, None
    return ypopt, ypcov



def extract_from_data(
    data: Union[list[dict], pd.DataFrame],
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
        data = pd.DataFrame(data)
    # Check what parameters where given.
    if not groupby_key and not agg_type:
        # No grouping and no aggreagtion is wanted. Just return the wanted output key.
        return np.array(data[output_key].to_list())
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

def randomly_sample_from(data:np.ndarray, seed:int = None) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    sample_size = len(data[0])
    random_inds = random_generator.integers(0, sample_size, size=sample_size)
    return np.array(data)[:, random_inds]

def fit_randomly_sampled(x:Union[list, np.ndarray], data:list, func:callable, curvefit_dict:dict = {}) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(randomly_sample_from(data), axis = 1)
    popt, pcov = fit(func, x, means, **curvefit_dict)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
    return means, popt, perr

def data_uncertainties(data:np.ndarray, confidence: int, data_median: Optional[np.ndarray] = None):
    percentiles = [
        (100 - confidence) / 2,
        (100 + confidence) / 2,
    ]
    if data_median is None:
        data_median = np.median(data, axis=1)
    percentile_inteval = np.percentile(data, percentiles, axis=1)
    uncertainties = np.abs(np.vstack([data_median, data_median]) - percentile_inteval)
    return uncertainties

def single_bootstrap(x, data, func, curvefit_dict):
    return {k: v for k, v in zip(['mean', 'popt', 'perr'], fit_randomly_sampled(x, data, func, curvefit_dict))}

def bootstrap(x:np.ndarray, data:np.ndarray, func:callable, nbootstraps:int, curvefit_dict:dict = {}):
    bootstrapped_data = []
    for _ in range(nbootstraps):
        data_dict = single_bootstrap(x, data, func, curvefit_dict)
        if data_dict['popt'] is not None: bootstrapped_data.append(data_dict)
    return bootstrapped_data

def ndarrays_tolist_recursive(obj):
    if isinstance(obj, dict):
        return {key: ndarrays_tolist_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return np.round(obj, ROUND).tolist()
    else:
        return obj

def analyze(x, data, func, bootstrapped_data, confidence, curvefit_dict:dict = {}):
    ymean = np.mean(data, axis = 1)
    ypopt, ypcov = fit(func, x, ymean, **curvefit_dict)
    all_means = extract_from_data(bootstrapped_data, 'mean')
    mean_errup, mean_errlow = data_uncertainties(all_means.T, confidence, ymean)
    all_popts = extract_from_data(bootstrapped_data, 'popt')
    popt_errup, popt_errlow = data_uncertainties(all_popts.T, confidence, ypopt)
    result_dictionary = {
        'x': x,
        'y' : {'value': ymean, 'err_up': mean_errup, 'err_low': mean_errlow},
        'popt': {'value': ypopt, 'err_up': popt_errup, 'err_low': popt_errlow},
        'pcov': ypcov,
        'nbootstraps': len(bootstrapped_data),
        'confidence': confidence, 
        'curvefit_dict':  curvefit_dict
        }
    return ndarrays_tolist_recursive(result_dictionary)


def plot_result(resultdict:dict, func:callable, axis:Optional[Axes] = None):
    if axis is None: _, axis = plt.subplots(1, 1)
    x = resultdict['x']
    xlinspaced = np.linspace(np.min(x), np.max(x), 100)
    axis.errorbar(
        x, 
        resultdict['y']['value'], 
        yerr=[resultdict['y']['err_up'], resultdict['y']['err_low']],
        fmt='.', 
        capsize=2
    )
    # if 
    popt = np.array(resultdict['popt']['value']).astype(np.complex128)
    popt_err_up = np.array(resultdict['popt']['err_up']).astype(np.complex128)
    popt_err_low = np.array(resultdict['popt']['err_low']).astype(np.complex128)

    main_function = np.real(func(xlinspaced, *popt))
    error_tube_up = np.real(func(xlinspaced, *(popt + popt_err_up)))
    error_tube_low = np.real(func(xlinspaced, *(popt - popt_err_low)))

    swap = np.all(error_tube_up >= main_function) == False

    axis.plot(xlinspaced, main_function, '-')
    axis.fill_between(
        xlinspaced, 
        error_tube_up if not swap else error_tube_low,
        error_tube_low if not swap else error_tube_up, 
        alpha = 0.3
    )
    return axis