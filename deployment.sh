#!/bin/bash
#
# Deploys the machine learning workspace and code

# Create the model to be used
source venv/bin/activate
python source_code/create_model.py

az configure --defaults workspace=$WORKSPACE_NAME

# Create workspace and endpoint
az ml workspace create -o none
az ml endpoint create -n $ENDPOINT_NAME -f create-material-match-endpoint.yml

# Get URI and KEY
export SCORING_URI=$(az ml endpoint list --query [0].properties.scoringUri -o tsv)
export PRIMARY_ENDPOINT_KEY=$(az ml endpoint get-credentials -n $ENDPOINT_NAME --quer primaryKey -o tsv)