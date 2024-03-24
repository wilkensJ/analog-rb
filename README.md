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
In order to run the non-interaction randomized analog benchmarking protocol, the projectors have to be build. 
The ClebschGordan.cpp code which calculates the needed Clebsch-Gordan coefficients for the projector was  taken from Ref. [Alex, 2021](https://doi.org/10.1063/1.3521562).
It has to be transformed into an executable script, how to do that depends on the OS system in use.
In general, store the compile ClebschGordan.cpp as 'clebschgordan.out'
The C++ engine has to be installed first in order to make it executable, I suggest using `g++` and installing it via `conda`. 
Once this is installed the package `lapack` has to be installed.
Then run
```console
g++ ClebschGordan.cpp -framework Accelerate -o clebschgordan.out
```

## Usage

## Useful Resources
