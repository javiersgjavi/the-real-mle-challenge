import yaml
import unittest
from pathlib import Path
from src.model.engine import ModelEngine
from src.data.data import Data, load_config
from sklearn.model_selection import train_test_split


def load_test_config():
    config_path = Path(__file__).parent / 'test_config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


TEST_CONFIG = load_test_config()


class TestModelEngine(unittest.TestCase):
    def setUp(self):
        self.model_engine = ModelEngine()

        # Load test data
        data_config = load_config()
        data = Data(data_config)
        self.df_clean = data.load_clean(map_categorical_features=True)
        self.df_clean = self.df_clean.dropna(axis=0)

        features_names = self.model_engine.get_features_names()
        target_name = self.model_engine.get_target_name()
        x = self.df_clean[features_names]
        y = self.df_clean[target_name]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=TEST_CONFIG['test_size'],
            random_state=TEST_CONFIG['random_state']
        )

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def check_results(self, res):
        self.assertIn('accuracy', res)
        self.assertIn('roc_auc', res)
        self.assertIsInstance(res['accuracy'], float)
        self.assertIsInstance(res['roc_auc'], float)
        self.assertTrue(0 <= res['accuracy'] <= 1)
        self.assertTrue(0 <= res['roc_auc'] <= 1)

        self.assertAlmostEqual(
            res['accuracy'],
            TEST_CONFIG['expected_accuracy'],
            places=2
        )

        self.assertAlmostEqual(
            res['roc_auc'],
            TEST_CONFIG['expected_roc_auc'],
            places=2
        )

    def test_train(self):
        # Train the model
        self.model_engine.train(self.x_train, self.y_train)

        # Test the model
        res = self.model_engine.test(self.x_test, self.y_test)

        # Verify the results
        self.check_results(res)

    def test_load_model(self):
        self.model_engine.load_model()
        self.assertIsNotNone(self.model_engine.model)

        res = self.model_engine.test(self.x_test, self.y_test)
        self.check_results(res)


if __name__ == '__main__':
    unittest.main()
