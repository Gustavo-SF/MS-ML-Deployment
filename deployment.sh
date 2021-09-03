#!/bin/bash
#
# Deploys the machine learning workspace and code

# Create the model to be used
source venv/bin/activate
python source_code/create_model.py

az configure --defaults workspace=$WORKSPACE_NAME

# Create workspace and endpoint
az ml workspace create
az ml endpoint create --local -n $ENDPOINT_NAME -f create-material-match-endpoint.yml

