import logging
import pandas as pd
from typing import Union, Dict, Any
from fastapi import HTTPException
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import ValidationError

from src.data.data import Data, load_config as load_data_config
from src.model.engine import ModelEngine
from app.classes.data_types import InputData, OutputData, BatchInputData, \
    BatchOutputData

logger = logging.getLogger(__name__)


class APIController:
    """
    Controller class for handling API requests and predictions.
    """

    def __init__(self, api_config: Dict[str, Any]):
        """
        Initialize the APIController.

        Args:
            api_config (Dict[str, Any]): Configuration dictionary for the API.
        """
        self.api_config = api_config
        config_data = load_data_config()
        self.preprocessing_pipeline = (
            Data(config_data).get_preprocessing_pipeline()
        )
        self.model = ModelEngine()
        self.model.load_model(api_config['model_to_use'])

    async def get_api_key(self, api_key_header: str, api_key: str):
        """
        Validate the API key.

        Args:
            api_key_header (str): API key from the request header.
            api_key (str): Expected API key.

        Returns:
            str: Validated API key.

        Raises:
            HTTPException: If the API key is invalid.
        """
        if api_key_header == api_key:
            return api_key_header
        else:
            logger.warning("Unauthorized access")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Could not validate credentials"
            )

    async def predict(
            self,
            input_data: Union[InputData, BatchInputData]
    ) -> Union[OutputData, BatchOutputData]:
        """
        Make predictions based on input data.

        Args:
            input_data (Union[dict, Dict[str, List[dict]]]): Input data for
            prediction.

        Returns:
            Union[OutputData, BatchOutputData]: Prediction results.

        Raises:
            HTTPException: If there's a validation error or internal server
            error.
        """
        logger.info("Prediction request received")
        try:
            is_batch = isinstance(input_data, BatchInputData)
            input_df = pd.DataFrame(
                [data.dict() for
                    data in (input_data.data if is_batch else [input_data])]
            )

            preprocessed_df = (
                self.preprocessing_pipeline.map_categorical_features(input_df)
            )
            features = self.model.get_features_names()
            X = preprocessed_df[features]

            preds = self.model.predict(X).astype(int)
            predictions = [
                self.preprocessing_pipeline.get_category_name(pred)
                for pred in preds
            ]

            if is_batch:
                results = [
                    OutputData(id=data.id, price_category=pred)
                    for data, pred in zip(input_data.data, predictions)
                ]

                n = len(results)
                logger.info(
                    f"Batch prediction completed for {n} predictions"
                )
                return BatchOutputData(results=results)
            else:
                logger.info("Individual prediction completed")
                return OutputData(
                    id=input_data.id,
                    price_category=predictions[0]
                )

        except ValidationError as ve:
            logger.error(f"Validation error: {str(ve)}")
            raise HTTPException(
                status_code=422,
                detail=f"Validation error: {str(ve)}"
            )
        except Exception as e:
            logger.exception(f"Exception during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def welcome(self):
        """
        Generate a welcome message for the API.

        Returns:
            Dict[str, str]: A dictionary containing the welcome message.
        """
        logger.info("Welcome request received")
        msg = "Hello! Welcome to the Airbnb price prediction API."
        return {"message": msg}
