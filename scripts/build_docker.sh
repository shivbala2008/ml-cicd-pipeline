#!/bin/bash
set -e

echo "🐳 Building Docker image for ML model..."

# Configuration
IMAGE_NAME="ml-model-api"
TAG=${1:-latest}
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Build image
echo "Building image: ${FULL_IMAGE_NAME}"
docker build -t ${FULL_IMAGE_NAME} .

# Test the image
echo "🧪 Testing Docker image..."
docker run --rm -d --name ml-test -p 5001:5000 ${FULL_IMAGE_NAME}

# Wait for container to start
sleep 10

# Health check
echo "Checking health endpoint..."
if curl -f http://localhost:5001/health; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    docker logs ml-test
    docker stop ml-test
    exit 1
fi

# Stop test container
docker stop ml-test

echo "✅ Docker image built and tested successfully: ${FULL_IMAGE_NAME}"

# Optional: Push to registry
if [ "$2" = "push" ]; then
    echo "🚀 Pushing image to registry..."
    docker push ${FULL_IMAGE_NAME}
    echo "✅ Image pushed to registry"
fi