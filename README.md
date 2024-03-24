# Randomized Analog Benchmarking (for particle preserving bosonic systems)

## Requirements

I recommend using [virtual environments](https://docs.python.org/3/tutorial/venv.html) and [pip](https://pypi.org/project/pip/).
The main instructions how to build a virtual environment and install this package and its requirements are listed below, given that Python 3 and pip are installed.

1. Open a terminal and navigate into this project's directory (.../analog-rb).
2. Create a virtual environment (type `python` or `python3`):
```console
python -m venv analogrbvenv
```
3. Activate this virtual environment in the terminal with 
```console
source analogrbvenv/bin/activate
```
4. Install this package by running the command
```console
pip install .
```

### compile C++ code
Only needed if non-interactive systems are benchmarked. If only the demo protocols are run, it is also not needed since the projectors are pre-saved.

The ClebschGordan.cpp code which calculates the needed Clebsch-Gordan coefficients for the projector was  taken from Ref. [Alex, 2021](https://doi.org/10.1063/1.3521562).
It has to be transformed into an executable script, how to do that depends on the operating system in use.
The C++ engine has to be installed first in order to make it executable, I suggest using `g++` and installing it via `conda` for Linux, something like `MinGW-W64` for windows and for mac using `homebrew`. 
Once this is installed the package `lapack` has to be installed.
Then run the following in the `analog-rb` directory in a terminal:
```console
g++ -o clebschgordan.out ClebschGordan.cpp -llapack
```
which should produce a file on the same level as the `README.md` called `clebschgordan.out`.

## Debugging
The data might be not stored in the right directory. Then you have to set the path by hand.
For that open file `src/analogrb/save_load` and comment out the function `MODULE_DIR()`.
Instead, comment in the function below that and enter the absolute path to the directory of your analog-rb:

![Bildschirmfoto 2024-03-24 um 16 38 50](https://github.com/wilkensJ/analog-rb/assets/70592365/aec6a863-5768-4f2b-a6b1-a0b47e3229bc)

Should then look like this (of course add your path here):

![Bildschirmfoto 2024-03-24 um 16 43 31](https://github.com/wilkensJ/analog-rb/assets/70592365/47ca8a35-ec19-4ee0-8fc6-6a5aeeda8c2b)

Then, after saving, again type `pip install .` in your terminal.

## Usage

## Useful Resources
