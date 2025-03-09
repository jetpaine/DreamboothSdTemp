# Dreambooth Stable Diffusion Guide

## Setup and Build
- Conda: `conda env create -f environment.yaml`
- Venv: `python -m venv dreambooth_joepenna && source dreambooth_joepenna/bin/activate && pip install -r requirements.txt`
- Setup: `pip install -e .`

## Run Commands
- Training: `python main.py --project_name "ProjectName" --training_model "/path/model.ckpt" --regularization_images "/path/reg" --training_images "/path/training" --class_word "person" --token "zwx" --save_every_x_steps 250`
- With config: `python main.py --config_file_path "/path/config.json"`
- Evaluation: `python scripts/evaluate_model.py`
- Image generation: `python scripts/stable_txt2img.py`

## Code Style Guidelines
- Python 3.7+ compatible code
- Use relative imports within the package
- Class names: CamelCase
- Functions/variables: snake_case
- Constants: UPPERCASE
- Prefer explicit type hints where possible
- Document complex functions with docstrings
- Error handling: use try/except with specific exceptions
- Config-driven approach for ML parameters

## Repository Structure
- configs/: Model and configuration files
- ldm/: Core diffusion model implementation
- dreambooth_helpers/: Training utilities
- scripts/: Utility scripts for running models