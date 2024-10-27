import sys
import yaml
import logging
import pandas as pd
from typing import Union
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from dotenv import load_dotenv
import os

from lab.src.data.data import Data, load_config as load_data_config
from lab.src.model.engine import ModelEngine

from lab.api.classes.data_types import InputData, OutputData, BatchInputData, \
    BatchOutputData


def load_api_config():
    config_path = (
        Path(__file__).resolve().parent / 'config_api.yaml'
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s',  # Changed format here
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Configure the specific logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load the configuration
API_CONFIG = load_api_config()
config_data = load_data_config()
preprocessing_pipeline = Data(config_data).get_preprocessing_pipeline()

app = FastAPI()

model = ModelEngine()
model.load_model(API_CONFIG['model_to_use'])


# Intenta cargar las variables de entorno, pero no falla si .env no existe
load_dotenv(verbose=True)

# Si API_TOKEN no está en .env, intenta obtenerlo directamente de las
# variables de entorno
API_KEY = os.getenv("API_TOKEN")

if not API_KEY:
    raise ValueError("API_TOKEN not found in environment variables")

print(f'API_KEY: {API_KEY}')
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        logger.warning("Unauthorized access")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )


@app.get("/")
async def welcome():
    logger.info("Solicitud de saludo recibida")
    msg = ("¡Hola! Bienvenido a la API de predicción de precios de Airbnb.")
    return {"message": msg}


@app.post(
    "/predict",
    response_model=Union[OutputData, BatchOutputData]
)
async def predict(
    input_data: Union[InputData, BatchInputData],
    api_key: APIKey = Depends(get_api_key)
):
    logger.info("Prediction request received")
    try:
        # Determine if it's an individual or batch prediction
        is_batch = isinstance(input_data, BatchInputData)

        # Convert input data to DataFrame
        if is_batch:
            input_df = pd.DataFrame([data.dict() for data in input_data.data])
        else:
            input_df = pd.DataFrame([input_data.dict()])

        # Preprocess the data
        preprocessed_df = preprocessing_pipeline.map_categorical_features(
            input_df
        )

        # Get the features needed for prediction
        features = model.get_features_names()
        X = preprocessed_df[features]

        # Make the prediction
        preds = model.predict(X).astype(int)

        # Map predictions to price categories
        predictions = [
            preprocessing_pipeline.get_category_name(pred) for pred in preds
        ]

        # Return the result
        if is_batch:
            results = [
                OutputData(id=data.id, price_category=pred)
                for data, pred in zip(input_data.data, predictions)
            ]
            logger.info(
                f"Batch prediction completed for {len(results)} predictions"
            )
            return BatchOutputData(results=results)
        else:
            logger.info("Individual prediction completed")
            return OutputData(id=input_data.id, price_category=predictions[0])

    except Exception as e:
        logger.exception(f"Exception during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )
