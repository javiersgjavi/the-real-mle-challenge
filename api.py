import os
import sys
import logging
from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey

from lab.api.classes.controller import APIController
from lab.api.classes.data_types import InputData, OutputData, BatchInputData, \
    BatchOutputData

from lab.api.utils import load_api_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load configurations
API_CONFIG = load_api_config()

# Load API token
load_dotenv(verbose=True)
API_KEY = os.getenv("API_TOKEN")
if not API_KEY:
    raise ValueError("API_TOKEN not found in environment variables")

app = FastAPI()
api_controller = APIController(API_CONFIG)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate the API key."""
    return await api_controller.get_api_key(api_key_header, API_KEY)


@app.get("/")
async def welcome():
    """Return a welcome message."""
    return await api_controller.welcome()


@app.post("/predict", response_model=Union[OutputData, BatchOutputData])
async def predict(
    input_data: Union[InputData, BatchInputData],
    api_key: APIKey = Depends(get_api_key)
):
    """
    Make predictions based on input data.

    This endpoint accepts either single input data or batch input data.
    It requires a valid API key for authentication.

    Args:
        input_data: Either a single InputData object or a BatchInputData
        object.
        api_key: API key for authentication.

    Returns:
        Prediction results as either OutputData or BatchOutputData.
    """
    return await api_controller.predict(input_data)
