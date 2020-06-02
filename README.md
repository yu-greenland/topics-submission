# Topics in Computer Science Submission

The main system for the submission are ```adjust_cl.py```, ```adjust_i.py``` and the ```.pkl``` files.

## Installation

The ```adjust_xx.py``` programs do not need a virtual environment or any installations. All used libraries are standard.

If you want to use ```main.py``` and ```conduct_anaysis.py``` you will have to create a virtual environment and install all the libraries listed in ```requirements.txt```.

To create a virtual environment and download all the requirements go:

```bash
python3 -m venv venv
source ./setup.sh
```

Disclaimer: I have only tested this on my Mac

Note that this will not actually do anything because ```main.py``` requires a csv file to read data from. The original csv file is not included in this submission due to privacy and information sharing concerns.

## Usage

To run the interactive adjust program simply go:

```bash
python adjust_i.py
```

To view the options and how to run the command line version go:
```bash
python adjust_cl.py -h
```

Lastly, if you have a csv file with the raw data and want to produce density plots, go into virtual environment and go:
```bash
python main.py
```

## Authors

Greenland Yu, The University of Adelaide

Supervised by Mark Jenkinson and Stephan Lau
