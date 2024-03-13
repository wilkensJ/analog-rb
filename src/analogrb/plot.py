import matplotlib.pyplot as plt
import numpy as np
from analogrb.bootstrap import fit_with_bootstrap
from analogrb.arb_protocol import EXP_FUNC
from analogrb.save_load import extract_from_data
import pandas as pd

COLORS=['#AF58BA','#32CB7B', '#FFC61E','#ED5DA3']
MARKERS = ['x', '.','4', 'p', 'D']
AVG_MARKER = 'x'


def plot_arb(data, nbootstraps, confidence, start_fitting = 0):
    df = pd.DataFrame(data)
    names = [key for key in df.keys() if key.startswith("q_")]

    fig = plt.figure(figsize=np.array([3.5, 3.5* 3 / 5]))
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=3)
    plt.tick_params(axis='both', direction='in')
    plt.rcParams.update({"font.size": 9})

    for k, name in enumerate(names):
        ms, qs = extract_from_data(df, name, 'm', list)
        fitting_index = np.argwhere(ms >= start_fitting)[0,0]
        ms_capped = ms[fitting_index:]
        qs_capped = qs[fitting_index:]
        color = COLORS[k]
        plt.scatter(*extract_from_data(df, name, 'm'), alpha=0.1, color=color)
        try:
            averages, errorbars, popt, pcov = fit_with_bootstrap(ms_capped, qs_capped, nbootstraps, confidence)
            plt.scatter(*extract_from_data(df, name, 'm'), alpha=0.3)
            plt.errorbar(ms_capped, averages, yerr=errorbars, fmt=AVG_MARKER, capsize=2, color=color, label=f"irrep {name[2:]}")
            ms_finespaced = np.linspace(ms[fitting_index], ms[-1], len(ms)*10)
            plt.plot(ms_finespaced, EXP_FUNC(ms_finespaced, *popt), '--', color = color)
            print(f"irrep {name[2:]} popt: A, p = {popt}")
        except RuntimeError:
            plt.plot([], [], "--", color = color, label=f"irrep {name} fit error")
    plt.ylabel('q')
    plt.xlabel('m')
    plt.legend()
    return fig

def plot_arb_only_data(x, data, nbootstraps, confidence, all_projectors, filename=None):
    nboxplots = len(data[0])
    fig = plt.figure(figsize=np.array([3.5, 3.5* 3 / 5]))
    ax = fig.add_subplot(111)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=3)
    plt.tick_params(axis='both', direction='in')
    plt.rcParams.update({"font.size": 9})
    ms_finespaced = np.linspace(x[0], x[-1], len(x)*10)
    for k, p in enumerate(all_projectors):
        y_scatter = data[k]
        color = COLORS[k]
        plt.scatter(np.tile(x, nboxplots), y_scatter, alpha=0.1, color=color)
        averages, error_bars, popt, pcov = fit_with_bootstrap(x, y_scatter.T, nbootstraps, confidence)
        plt.errorbar(x, averages, yerr=error_bars, fmt=AVG_MARKER, capsize=2, color=color)
        plt.plot(ms_finespaced, EXP_FUNC(ms_finespaced, *popt),'--', color = color)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    return fig

def plot_arb_intgates_nonintprojectors(x, data, nbootstraps, confidence, all_projectors, filename=None):
    nboxplots = len(data[0])
    fig = plt.figure(figsize=np.array([3.5, 3.5* 3 / 5]))
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=3)
    plt.tick_params(axis='both', direction='in')
    plt.rcParams.update({"font.size": 9})
    ms_finespaced = np.linspace(x[0], x[-1], len(x)*10)
    for k, p in enumerate(all_projectors):
        y_scatter = data[k]
        color = COLORS[k]
        plt.scatter(np.tile(x, nboxplots), y_scatter, alpha=0.1, color=color, label=f"irrep {p.irrep}")
        averages, error_bars, popt, pcov = fit_with_bootstrap(x, y_scatter.T, nbootstraps, confidence)
        plt.errorbar(x, averages, fmt=AVG_MARKER, capsize=2, color=color)
        # plt.plot(ms_finespaced, EXP_FUNC(ms_finespaced, *popt),'--', color = color)
        print(f"irrep {p.irrep} popt: A, p = {popt}")
    plt.ylabel('q')
    plt.xlabel('m')
    plt.legend()
    return fig
