import yaml
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Load the configuration from the config_preprocessing.yaml file.

    Searches for the file and loads it if found.

    Returns:
        Dict[str, Any]: Dictionary with the loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    file_name = 'config_preprocessing.yaml'
    path = Path(__file__).parent.parent.parent.parent / file_name

    if path.exists():
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    logging.error(f"Could not find {file_name} file")
    raise FileNotFoundError(
        f"Could not find the {file_name} file. Searched in: \n{path}"
    )


class PreprocessingPipeline:
    """
    Class that implements the data preprocessing pipeline.
    """

    def __init__(self, config: Dict[str, Any]):

        self.config = config['preprocessing']

        self.columns_to_use = self.config['columns_to_use']
        self.columns_to_rename = self.config['columns_to_rename']

        self.target_regex = self.config['target_regex']
        self.target_dtype = self.config['target_dtype']

        self.min_price = self.config['min_price']

        self.bins_categories = [
            float(i) for i in self.config['bins_categories']
        ]

        self.amenities_to_drop = self.config['amenities_to_drop']

        categorical_mapping = self.config['categorical_mapping']

        self.mapping_room_type = categorical_mapping['room_type']
        self.mapping_neighbourhood = categorical_mapping['neighbourhood']

        logging.info("Preprocessing pipeline initialized")

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.columns_to_use]

    def num_bathroom_from_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the 'bathrooms_text' column to numeric values.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'bathrooms' column converted to
            numeric.
        """
        def _num_bathroom_from_text(text):
            try:
                if isinstance(text, str):
                    bath_num = text.split(" ")[0]
                    return float(bath_num)
                else:
                    return np.nan
            except ValueError:
                return np.nan
        df['bathrooms'] = df['bathrooms_text'].apply(_num_bathroom_from_text)
        df = df.drop(columns=['bathrooms_text'])
        return df

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.columns_to_rename)

    def drop_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=0)

    def clean_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert the 'price' column to the specified data type.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with cleaned and converted 'price' column.
        """
        df['price'] = df['price'].str.extract(self.target_regex, expand=False)
        df['price'] = df['price'].astype(self.target_dtype)
        return df

    def filter_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame by the specified minimum price.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame filtered by minimum price.
        """
        return df[df['price'] >= self.min_price]

    def categorize_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize prices into bins.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new 'category' column of categorized
            prices.
        """
        labels = list(range(len(self.bins_categories) - 1))
        df['category'] = pd.cut(
            df['price'],
            bins=self.bins_categories,
            labels=labels,
        )
        return df

    def preprocess_amenities(self, df: pd.DataFrame) -> pd.DataFrame:
        for amenity in self.amenities_to_drop:
            amenity_name = amenity.replace(' ', '_')
            check = df['amenities'].str.contains(amenity)
            df[amenity_name] = check.astype(int)

        df = df.drop(columns=['amenities'])
        return df

    def clean(
            self,
            df: pd.DataFrame,
            map_categorical_features: bool = False
    ) -> pd.DataFrame:

        logging.info("Starting data cleaning process")
        df = self.num_bathroom_from_text(df)
        df = self.select_columns(df)
        df = self.rename_columns(df)
        df = self.drop_nans(df)
        df = self.clean_target(df)
        df = self.filter_price(df)
        df = self.categorize_price(df)
        df = self.preprocess_amenities(df)

        if map_categorical_features:
            df = self.map_categorical_features(df)

        logging.info("Data cleaning process completed")
        return df

    def map_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map categorical features to numeric values.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with mapped categorical features.
        """
        df['room_type'] = df['room_type'].map(self.mapping_room_type)
        df['neighbourhood'] = df['neighbourhood'].map(
            self.mapping_neighbourhood
        )
        return df


class Data:
    """
    Class for loading and processing data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data class.

        Args:
            config (Dict[str, Any]): Configuration for data processing.
        """
        self.base_path = Path(__file__).parent.parent.parent.parent
        self.path_raw = self.base_path / config['paths']['raw']
        self.path_clean = self.base_path / config['paths']['clean']
        self.preprocessing_pipeline = PreprocessingPipeline(config)

    def load_raw(self) -> pd.DataFrame:
        return pd.read_csv(self.path_raw)

    def load_clean(
            self,
            map_categorical_features: bool = False
    ) -> pd.DataFrame:
        """
        Load and clean the data.

        Args:
            map_categorical_features (bool): Whether to map features.

        Returns:
            pd.DataFrame: DataFrame with clean and processed data.
        """
        df = pd.read_csv(self.path_raw)
        return self.preprocessing_pipeline.clean(df, map_categorical_features)
