import yaml
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from lab.src.data.data import Data, load_config


def load_test_config():
    config_path = Path(__file__).parent / 'test_config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


TEST_CONFIG = load_test_config()


class TestLoadConfig(unittest.TestCase):
    def test_load_config(self):
        # Try to load the configuration
        try:
            config = load_config()
        except FileNotFoundError as e:
            self.fail(f"Could not load the configuration: {str(e)}")

        # Verify that config is not None
        self.assertIsNotNone(config)

        # Verify that config is a dictionary
        self.assertIsInstance(config, dict)

        # Verify that config contains the expected keys
        self.assertIn('paths', config)
        self.assertIn('preprocessing', config)


class TestData(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.base_path = Path(__file__).parent.parent.parent
        self.data = Data(self.config)

    def test_load_raw(self):
        # Call the load_raw function
        result = self.data.load_raw()

        # Verify that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Verify that the result has the expected columns
        expected_columns = self.config['preprocessing']['columns_to_use']
        self.assertTrue(
            all(column in result.columns for column in expected_columns)
        )

        # Verify that the number of rows is correct
        self.assertEqual(len(result), TEST_CONFIG['expected_raw_row_count'])

    def test_load_clean(self):
        clean_path = self.base_path / self.config['paths']['clean']
        df_test = pd.read_csv(clean_path, index_col='Unnamed: 0').fillna(0)

        # Call the load_clean function
        result = self.data.load_clean().fillna(0)

        equals = (result == df_test).all(axis=1).all()
        self.assertTrue(equals)

        result = self.data.preprocessing_pipeline.map_categorical_features(
            result
        )

        self.assertEqual(result['room_type'].dtype, np.int64)
        self.assertEqual(result['neighbourhood'].dtype, np.int64)
