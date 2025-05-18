#!/bin/bash

# Build and push the Docker image
docker build --no-cache -t quay.io/yourtechbud/orpheus-server:0.1.1 .
docker push quay.io/yourtechbud/orpheus-server:0.1.1

