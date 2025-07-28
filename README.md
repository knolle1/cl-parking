# Evaluating RL in Continual Learning

This project evaluates the behaviour of RL algorithms in continual learning settings.

## Dependencies

1. Create and activate virtual environment:
	- With conda:
		- `conda create -n cl-parking python=3.10`
		- `conda activate cl-parking`
	- Alternatively with venv:
		- `python -m venv .venv`
		- `source .venv/bin/activate`
2. Install pytorch (must be installed before other packages due to dependencies): `pip install torch==2.2.1`
2. Install dependencies: `pip install -r requirements.txt`
3. Install ImageMagick (for generating videos): `conda install conda-forge::imagemagick==7.1.1_28`