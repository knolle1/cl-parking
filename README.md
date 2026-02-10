# Evaluating RL in Continual Learning

This project evaluates the behaviour of RL algorithms in continual learning settings.

## Dependencies

1. Create and activate virtual environment:
	- With conda:
		- `conda create -n cl-parking python=3.10`
		- `conda activate cl-parking`
	- Alternatively with venv (requires python 3.10):
		- `python -m venv .venv`
		- `source .venv/bin/activate`
		
2. Install tools needed to build and install dependencies:
        - `pip install --upgrade pip`
        - `pip install packaging`
        - `pip install setuptools==69.5.1 wheel`
        
3. Install dependencies in the following order:
        - `pip install numpy==1.24.3`
        - `pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121`
        - `pip install causal-conv1d==1.5.0.post8 --no-build-isolation`
        - `pip install mamba-ssm==1.2.0.post1 --no-build-isolation`
        - `pip install -r requirements.txt`
