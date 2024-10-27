import os
import unittest
import requests
from dotenv import load_dotenv

# Try to load environment variables, but don't fail if .env doesn't exist
load_dotenv(verbose=True)

# If API_TOKEN is not in .env, try to get it directly from environment
# variables
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    print(
        "Warning: API_TOKEN not found. Some tests may fail."
    )


class TestAPIPredict(unittest.TestCase):
    def setUp(self):
        self.url = "http://localhost:8000/predict"
        self.headers = {"X-API-Key": API_TOKEN} if API_TOKEN else {}
        self.data_individual = {
            "id": 1001,
            "accommodates": 4,
            "room_type": "Entire home/apt",
            "beds": 2,
            "bedrooms": 1,
            "bathrooms": 2.0,
            "neighbourhood": "Brooklyn",
            "tv": 1,
            "elevator": 1,
            "internet": 0,
            "latitude": 40.71383,
            "longitude": -73.9658
        }
        self.data_batch = {
            "data": [
                self.data_individual,
                {
                    "id": 1002,
                    "accommodates": 2,
                    "room_type": "Private room",
                    "beds": 1,
                    "bedrooms": 1,
                    "bathrooms": 1.0,
                    "neighbourhood": "Manhattan",
                    "tv": 0,
                    "elevator": 0,
                    "internet": 1,
                    "latitude": 40.75383,
                    "longitude": -73.9858
                }
            ]
        }

    @unittest.skipIf(not API_TOKEN, "API_TOKEN not available")
    def test_authorized_access(self):
        response = requests.post(
            self.url,
            json=self.data_individual,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)

    def test_individual_prediction(self):
        response = requests.post(
            self.url,
            json=self.data_individual,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)

        response = response.json()
        self.assertIn('id', response)
        self.assertIn('price_category', response)
        self.assertEqual(response['id'], self.data_individual['id'])
        self.assertIn(
            response['price_category'],
            ['Low', 'Medium', 'High', 'Very High']
        )

    def test_batch_prediction(self):
        response = requests.post(
            self.url,
            json=self.data_batch,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)

        response = response.json()
        self.assertIn('results', response)

        result0, result1 = response['results']
        self.assertEqual(result0['id'], self.data_batch['data'][0]['id'])
        self.assertEqual(result1['id'], self.data_batch['data'][1]['id'])
        self.assertIn(
            result0['price_category'],
            ['Low', 'Medium', 'High', 'Very High']
        )
        self.assertIn(
            result1['price_category'],
            ['Low', 'Medium', 'High', 'Very High']
        )

    def test_unauthorized_access(self):
        response = requests.post(
            self.url,
            json=self.data_individual
        )
        self.assertEqual(response.status_code, 403)

    def test_invalid_room_type(self):
        invalid_data = self.data_individual.copy()
        invalid_data['room_type'] = "Invalid Room Type"
        response = requests.post(
            self.url,
            json=invalid_data,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

    def test_invalid_neighbourhood(self):
        invalid_data = self.data_individual.copy()
        invalid_data['neighbourhood'] = "Invalid Neighbourhood"
        response = requests.post(
            self.url,
            json=invalid_data,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
