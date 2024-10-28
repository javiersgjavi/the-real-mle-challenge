# SOLUTIONS.md

## Javier Solís García

### Challenge 1

For this part, I reviewed the notebook code and organized it into classes to make the workflow easier to manage. All code is implemented within `src/`.

I also implemented tests in the `tests/` folder, which can be run from the `test.sh` file in the project root. These tests, along with a coverage check and Flake8, are integrated into GitHub Actions to ensure automated testing with each commit, promoting a robust testing workflow in the repository. Additionally, Sphinx is used to automatically generate project documentation, accessible at:

[![Documentation Status](https://github.com/javiersgjavi/the-real-mle-challenge/actions/workflows/docs.yml/badge.svg)](https://javiersgjavi.github.io/the-real-mle-challenge/)

A `requirements.txt` file is also included, specifying the necessary dependencies for running the project.

The resulting project structure is as follows:
- `.github/workflows/`: folder containing workflows triggered with each commit.
- `config/`: folder for configuration files.
- `data/`: folder containing data files.
- `models/`: folder for stored models.
- `docs/`: folder for generating automatic documentation.
- `src/`: folder containing the main files for this part of the challenge.
- `tests/`: folder containing implemented test files.
- `test.sh`: Python script to run tests and measure coverage. It must be given execution permissions before running.
- `requirements.txt`

To test the repository, execute the following commands:

```bash
chmod +x test.sh
./test.sh
```

### Configuration Files

To manage class parameters and facilitate easy adjustments, I added configuration files in the `config/` folder:

1. **`preprocessing.yaml`**: Contains parameters and paths for data preprocessing, such as columns to use and rename, regular expressions for cleaning price, minimum price values, price categories with labels, and mappings for categorical variables.

2. **`model.yaml`**: Configures the model, its hyperparameters, and specific training parameters such as `test_size` and `random_state` to ensure reproducibility. The configured model is a Random Forest with class balancing and parallelization in `n_jobs`.

### Solution Structure

#### 1. `Data` Class

Handles the loading and storage of Airbnb data:
- **`load_raw()`**: Loads raw data from the CSV file specified in `preprocessing.yaml`.
- **`load_clean()`**: Executes the full preprocessing of the data using `PreprocessingPipeline`.

**Tests for `Data`**:  
Unit tests in `TestData` ensure that:
- The `load_raw()` function correctly loads data from the raw CSV file, validating that the columns match `columns_to_use` from the configuration and that the row count meets the expected value (`expected_raw_row_count` in `test_config.yaml`).
- The `load_clean()` function produces a DataFrame that matches the reference preprocessed file. This includes type verification for mapped columns (`room_type` and `neighbourhood`).


#### 2. `PreprocessingPipeline` Class

This class manages all preprocessing steps:
- **`select_columns()`** and **`rename_columns()`**: Selects and renames columns as specified in `preprocessing.yaml`.
- **`num_bathroom_from_text()`**: Converts textual values in the `bathrooms_text` column to numeric format.
- **`drop_nans()`**: Removes rows with null values.
- **`clean_target()`**, **`filter_price()`**, and **`categorize_price()`**: Normalize and filter the price column based on the defined range and categorize the price values into labeled bins.
- **`preprocess_amenities()`**: Converts `amenities` features into binary variables, making them model-ready.
- **`map_categorical_features()`**: Maps `room_type` and `neighbourhood` categories to numeric values as per the configuration in `preprocessing.yaml`.

**Tests for `PreprocessingPipeline`**:  
Tests for `PreprocessingPipeline` are included in `TestData` and verify:
- That each pipeline step produces the expected result, evaluating the accuracy of each operation, such as column renaming, text conversion, and price categorization.
- The correct assignment of integer values to `room_type` and `neighbourhood` categories.



#### 3. `ModelEngine` Class

The `ModelEngine` class handles the training, evaluation, and management of the prediction model configured in `model.yaml`:
- **`train()`** and **`test()`**: Conduct model training and evaluation according to the configured parameters.
- **`load_model()`**: Loads a previously trained model.
- **`retrain_and_save()`**: Allows for retraining with new data and saves the updated model.

**Tests for `ModelEngine`**:  
In `TestModelEngine`, tests verify:
- **`test_train()`**: Trains and evaluates the model, validating the `accuracy` and `roc_auc` results against expected values in `test_config.yaml`.
- **`test_load_model()`**: Confirms that the loaded model makes predictions and achieves the expected accuracy.
- **`test_retrain_and_save()`**: Evaluates the retraining process and saves the updated model.


#### Questions for the Team

- I noticed a minor issue in the price binning process, where prices equal to 10 are categorized as NaNs due to a parameter in the function used. Should we address this, or has it been accounted for already?

- Perhaps the configuration file structure could be optimized, and the classes could be set up to support multiple configurations, allowing greater flexibility. For example, enabling different types of models.

- Would it be beneficial to introduce hyperparameters similarly to argparse, as an additional option?

- Should we review the naming conventions used for new models that are trained?

- Are there any additional tests that would be interesting or valuable to perform?

- Is there interest in generating visualization charts for results from the notebooks as well?


### Challenge 2

I created a minimalistic API with essential prediction functionality, structuring the code in a way that facilitates testing and deployment.

At this point, the root project structure is as follows:

- `.github/workflows/`: folder containing workflows triggered with each commit.
- `config/`: folder for configuration files.
- `data/`: folder containing data files.
- `models/`: folder for stored models.
- `docs/`: folder for generating automatic documentation.
- `src/`: folder containing the main files for this part of the challenge.
- `tests/`: folder containing implemented test files.
- `api/`: API-related code.
- `test.sh`: Python script to run tests and measure coverage. It must be given execution permissions before running.
- `requirements.txt`
- `api.py`: main file to launch the developed API.

The primary components are as follows:

- **API Structure**: 
    - The API is built with **FastAPI** to handle asynchronous requests and simplify input validation.
    - The `APIController` class manages incoming requests, validates them, and generates predictions based on the input data.
    - The **Endpoints**:
      - `/`: Provides a welcome message.
      - `/predict`: Handles both individual and batch prediction requests.
    - Authentication is implemented through an API key stored in environment variables (`API_TOKEN`). Requests require this key for authorization, which stored in `.env`.
    - **API Documentation**: Accessible at [http://localhost:8000/redoc](http://localhost:8000/redoc) for a full overview of the available endpoints, request formats, and responses.

- **Modules**:
    - **`controller.py`**: Centralizes API logic, handling data preprocessing, prediction, and API key validation.
    - **`data_types.py`**: Defines data models using Pydantic, ensuring validation for all incoming requests.
    - **`utils.py`**: Loads the API configuration (`api.yaml`), handling errors if the configuration file is missing.
    - **`api.yaml`**: Specifies the model to use for predictions (`simple_classifier.pkl`), enabling flexibility in model selection.

- **Testing**:
    - **TestAPIPredict**: Contains various test cases to validate different API behaviors, including:
      - Authorized and unauthorized access.
      - Valid and invalid individual and batch prediction requests.
      - Boundary testing for numeric values, such as latitude, longitude, and boolean fields.
      - Validation for incorrect JSON formats and unexpected extra fields in input.
    - **Environment**: The tests utilize a `.env` file to store the `API_TOKEN`, ensuring tests that require authentication can run locally or in CI/CD pipelines. The `.env` file loading is managed by `dotenv`.

### Additional Improvements and Future Considerations

- **Configuration Flexibility**: We should consider extending `api.yaml` and model selection to support multiple model types.
- **Add Endpoints**: For loading new models, retraining with new data, etc.

### Challenge 3

The goal of this challenge is to dockerize the API, making it straightforward to deploy and run in production. To achieve this, I created a `docker/` folder containing the following files:

- `Dockerfile`:
    - Uses `python:3.12-slim` as the base image.
    - Sets `/app` as the working directory.
    - Sets `PYTHONUNBUFFERED=1` to ensure real-time logging output.

- `requirements-docker.txt`:
    - A minimized version of `requirements.txt`, optimized for production by only including the dependencies necessary to run the API.

- `docker-compose.yaml`:
    - Defines the `airbnb-price-prediction` service.
    - Mounts the project directory to the `/app` folder in the container, enabling easy development and updates.
    - Opens the required port (`8000`) to make the API accessible.
    - Runs the command `uvicorn api:app --host 0.0.0.0 --port 8000` to start the API within the container.

Additionally, I created a `launch.sh` script in the project root to streamline API startup, with the option to launch either directly or via Docker:

- `launch.sh`:
    - Parses the `--docker` argument to decide whether to launch with Docker or directly.
    - If `--docker` is specified, it builds the Docker image (`airbnb-price-prediction`), then uses `docker compose` to deploy the containerized API.
    - If launched without the `--docker` flag, it starts the API locally with `uvicorn`.

**CI/CD Integration**:
- In GitHub Actions, I added steps to build and deploy the Docker container, verifying that the containerized API is correctly built and deployed during CI/CD testing.

With these adjustments, the final repository structure is as follows:

- `.github/workflows/`: folder containing workflows triggered with each commit.
- `config/`: folder for configuration files.
- `data/`: folder containing data files.
- `docker/`: Docker configuration files.
- `models/`: folder for stored models.
- `docs/`: folder for generating automatic documentation.
- `src/`: folder containing the main files for this part of the challenge.
- `tests/`: folder containing implemented test files.
- `api/`: API-related code.
- `test.sh`: Python script to run tests and measure coverage. It must be given execution permissions before running.
- `launch.sh`: script to launch the application.
- `requirements.txt`
- `api.py`: main file to launch the developed API.

To launch the application, we can choose to use Docker for production or `uvicorn --reload` for development, as follows:

```bash
# For production with Docker
./launch.sh --docker
```
```bash
# For local development with uvicorn
./launch.sh
```

#### Potential Improvements
We could consider adding additional arguments to `launch.sh` to control deployment further, such as specifying the model name to load or ports to use.