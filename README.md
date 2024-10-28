[![Documentation Status](https://github.com/javiersgjavi/the-real-mle-challenge/actions/workflows/docs.yml/badge.svg)](https://javiersgjavi.github.io/the-real-mle-challenge/) [![Python Tests](https://github.com/javiersgjavi/the-real-mle-challenge/actions/workflows/python-tests.yml/badge.svg)](https://github.com/javiersgjavi/the-real-mle-challenge/actions/workflows/python-tests.yml)


# The NY Estimator Problem
In this challenge, we will explore the use of Airbnb listing data to predict the price category for new listings. We want to represent a real-case scenario where the MLE is working hand-to-hand with the Data Scientists at Intelygenz.

In this case, the data scientists have handed us a set of notebooks (in `lab/analysis`) that describe the ML workflow for data preprocessing and modelling. They have also included the dataset used and the trained model.

We will use these notebooks as a baseline to create more optimized functions that can be used in an ML inference pipeline.
# The MLE Challenge
You have to fork this repository to complete the following challenges in your own `github` account. Feel free to solve the challenge however you want.

Once completed, add a `SOLUTIONS.md` file justifying your responses and don't forget to send back the solution.

If you have any doubts or questions, don't hesitate to open an issue to ask any question about any challenge.

## Challenge 1 - Refactor DEV code

The code included in `lab` has been developed by Data Scientists during the development stage of the project. Now it is time to take their solution into production, and for that we need to ensure the code is up to standard and optimised. The first challenge is to refactor the code in `lab/analysis` the best possible way to operate in production.

Not only optimisation is important at this stage, but also the code should be written and tested in a way that can be easily understood by other MLE and tested at different CI stages.

## Challenge 2 - Build an API

The next step is to build an API that make use of the trained model to define the price category for a new listing. Here is an example of an input/output payload for the API.

```json
input = {
    "id": 1001,
    "accommodates": 4,
    "room_type": "Entire home/apt",
    "beds": 2,
    "bedrooms": 1,
    "bathrooms": 2,
    "neighbourhood": "Brooklyn",
    "tv": 1,
    "elevator": 1,
    "internet": 0,
    "latitude": 40.71383,
    "longitude": -73.9658
}

output = {
    "id": 1001,
    "price_category": "High"
}
```

The key is to ensure the API is easy to use and easy to test. Feel free to architect the API in any way you like and use any framework you feel comfortable with. Just ensure it is easy to make calls to the API in a local setting.

## Challenge 3 - Dockerize your solution

Nowadays, we can't think of ML solutions in production without thinking about Docker and its benefits in terms of standardisation, scalability and performance. The objective here is to dockerize your API and ensure it is easy to deploy and run in production.
