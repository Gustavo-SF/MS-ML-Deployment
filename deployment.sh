#!/bin/bash
#
# Deploys the machine learning workspace and code

export WORKSPACE_NAME="ppp_workspace"
export ENDPOINT_NAME="ppp-endpoint"

az configure --defaults workspace=$WORKSPACE_NAME

az ml workspace create
az ml endpoint create --local -n $ENDPOINT_NAME -f create-material-match-endpoint.yml

