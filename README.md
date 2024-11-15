# yolo11-fish-detection

## Train and validation dataset
The dataset that is used for model training and validation is an open source dataset in Kaggle, 
the link is https://www.kaggle.com/datasets/larjeck/fish-detection-dataset

## Dependency management
This project use [uv](https://docs.astral.sh/uv/) as the dependency management tool.

1. Follow the uv document to install uv on your local machine
2. `uv lock`: Create a lockfile for the project's dependencies.
3. `uv sync`: Sync the project's dependencies with the environmen

## Streamlit app
This project include a simple Streamlit app for the fish detection demo
Use the following script to run Streamlit on your local machine
```bash
uv run streamlit run streamlit/app.py
```