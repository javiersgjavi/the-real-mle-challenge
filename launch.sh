#!/bin/bash

# Parse command line arguments
USE_DOCKER=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --docker) USE_DOCKER=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$USE_DOCKER" = true ] ; then
    echo "Launching API with Docker..."
    cd docker
    docker build . -t airbnb-price-prediction
    docker compose up -d
    echo "API launched with Docker successfully"
else
    echo "Launching API directly..."
    uvicorn api:app --reload --port 8000
    echo "API launched directly successfully"
fi
