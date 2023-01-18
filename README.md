# Income estimation model

This project trains a simple income estimation model from US census data.
For model details please see the [model scorecard](./model_card.md)

## Install

Install dependencies using pip or creating a conda environment

```bash
   # Create a new conda workspace
   conda create --name income-estimation
   # Activate the workspace
   conda activate income-estimation
   # Install requirements
   conda install --file requirements.txt
```

## Train

## REST API

There is a simple REST API available to make model predictions.
Start the api with

```bash
   uvicorn main:app
```

Now you can use the predict route to make predictions (see http://localhost:8000/docs for details).
There is also an example python script available in [api_predict.py](./api_predict.py)
