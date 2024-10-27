import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_config() -> Dict[str, Any]:
    """
    Load the configuration from the config_model.yaml file.

    Searches for the file and loads it if found.

    Returns:
        Dict[str, Any]: Dictionary with the loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """

    file_name = 'model.yaml'
    path = Path(__file__).parent.parent.parent.parent / 'config' / file_name

    if path.exists():
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    logging.error(f"Could not find {file_name} file")
    raise FileNotFoundError(
        f"Could not find the {file_name} file. Searched in: \n{path}"
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
        self.model_name = self.config['model']['type']
        self.train_config = self.config['train_config']

        if self.model_name == 'random_forest':
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

    def get_features_names(self) -> List[str]:
        return self.config['features']

    def get_target_name(self) -> str:
        return self.config['target']

    def load_model(self, name) -> None:
        """
        Load a pre-trained model from a file.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        model_path = self.base_path / self.config['paths']['model'] / name
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

    def retrain_and_save(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Retrain the model with new data, test it, and save the results.

        Args:
            data (pd.DataFrame): The dataset to use for retraining.

        Returns:
            Dict[str, float]: Dictionary containing accuracy and ROC AUC
            scores.
        """
        logging.info("Starting model retraining process")
        x = data[self.get_features_names()]
        y = data[self.get_target_name()]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.train_config['test_size'],
            random_state=self.train_config['random_state']
        )

        self.train(x_train, y_train)
        results = self.test(x_test, y_test)
        self.save_model(results)
        logging.info("Model retraining and saving completed")
        return results

    def save_model(self, results: Dict[str, float] = None) -> None:
        """
        Save the model and results to a file.

        Args:
            results (Dict[str, float], optional): Dictionary containing model
            performance metrics.
        """
        logging.info("Saving model")
        date = datetime.now().strftime('%Y%m%d_%H%M')
        if results is not None:
            results_str = '_'.join(
                [f'{k}_{v:.4f}' for k, v in results.items()]
            )
        else:
            results_str = ''

        name = f'{self.model_name}_{results_str}_{date}.pkl'
        model_path = self.base_path / self.config['paths']['model'] / name

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Model saved to {model_path}")
