#!/bin/bash

# Build and push the Docker image
docker build --no-cache -t quay.io/yourtechbud/orpheus-server .
docker push quay.io/yourtechbud/orpheus-server

