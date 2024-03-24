# Randomized Analog Benchmarking (for particle preserving bosonic systems)

## Requirements

I recomment using [virtual environments](https://docs.python.org/3/tutorial/venv.html) and [pip](https://pypi.org/project/pip/).
The main instruction how to build a virtual enviroment and install this package and its requirements are listed below, given that python 3 and pip are installed.

1. Open a terminal and navigate into this projects directory (.../analog-rb).
2. Create a virtual environment
```console
python3 -m venv analogrbvenv
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

## Usage

## Useful Resources
