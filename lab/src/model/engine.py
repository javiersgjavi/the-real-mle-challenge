import yaml
import pickle
import logging
import numpy as np

from pathlib import Path
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def load_config() -> Dict[str, Any]:
    """
    Load the configuration from the config_model.yaml file.

    Searches for the file in different locations and loads it if found.

    Returns:
        Dict[str, Any]: Dictionary with the loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    name_file = 'config_model.yaml'
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / name_file,
        Path(__file__).parent.parent.parent / name_file,
        Path(__file__).parent.parent / name_file,
        Path(__file__).parent / name_file,
        Path.cwd() / name_file
    ]

    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as file:
                return yaml.safe_load(file)

    logging.error(f"Could not find {name_file} file")
    raise FileNotFoundError(
        f"Could not find the {name_file} file. Searched in: \n" +
        "\n".join(str(p) for p in possible_paths)
    )


class ModelEngine:
    """
    Class for managing the machine learning model operations.
    """

    def __init__(self):
        """
        Initialize the ModelEngine class.
        """
        self.base_path = Path(__file__).parent.parent.parent.parent
        self.config = load_config()
        if self.config['model']['type'] == 'random_forest':
            self.model = RandomForestClassifier(
                **self.config['model']['params']
            )
        else:
            logging.error(
                f"Model type {self.config['model']['type']} not supported"
            )
            raise ValueError(
                f'Model type {self.config["model"]["type"]} not supported'
            )

        self.seed = self.config['seed']
        logging.info("ModelEngine initialized")

    def load_model(self) -> None:
        """
        Load a pre-trained model from a file.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        model_path = self.base_path / self.config['paths']['model']
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logging.error(f"Model file not found at {model_path}")
            raise

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model with the provided data.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        logging.info("Starting model training")
        self.model.fit(x, y)
        logging.info("Model training completed")

    def test(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Args:
            x (np.ndarray): Test features.
            y (np.ndarray): Test labels.

        Returns:
            Dict[str, float]: Dictionary containing accuracy and ROC AUC 
            scores.
        """
        logging.info("Starting model testing")
        y_pred = self.model.predict(x)
        y_proba = self.model.predict_proba(x)
        res = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba, multi_class='ovr'),
        }
        logging.info(f"Model testing completed. Results: {res}")
        return res

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x (np.ndarray): Array containing features for prediction.

        Returns:
            np.ndarray: Array of predictions.
        """
        logging.info("Making predictions")
        predictions = self.model.predict(x)
        logging.info("Predictions completed")
        return predictions

    def get_features_names(self) -> List[str]:
        return self.config['features']
 
    def get_target_name(self) -> str:
        return self.config['target']
