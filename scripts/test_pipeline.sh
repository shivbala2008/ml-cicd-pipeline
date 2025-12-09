#!/bin/bash
set -e

echo "🧪 Testing Complete ML CI/CD Pipeline Locally"
echo "=============================================="

# Function to print status
print_status() {
    echo "📋 $1"
}

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 successful"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Stage 1: Code Quality Checks
print_status "Stage 1: Code Quality Checks"
echo "Checking code formatting..."
black --check src/ tests/ || (echo "Run 'black src/ tests/' to fix formatting" && exit 1)
check_success "Code formatting"

echo "Linting code..."
flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
check_success "Code linting"

# Stage 2: Unit Tests
print_status "Stage 2: Unit Tests"
echo "Running unit tests..."
pytest tests/ -v --cov=src
check_success "Unit tests"

# Stage 3: Model Training
print_status "Stage 3: Model Training"
echo "Training ML model..."
python src/train.py
check_success "Model training"

# Validate model artifacts
if [ -f "models/model.joblib" ]; then
    echo "✅ Model file created"
else
    echo "❌ Model file not found"
    exit 1
fi

if [ -f "models/model_metrics.json" ]; then
    echo "✅ Model metrics file created"
    cat models/model_metrics.json | python -m json.tool
else
    echo "❌ Model metrics file not found"
    exit 1
fi

# Stage 4: Integration Tests
print_status "Stage 4: Integration Tests"
echo "Starting API server for integration testing..."

# Start API in background
python src/predict.py &
API_PID=$!
echo "API started with PID: $API_PID"

# Wait for server to start
echo "Waiting for API to start..."
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:5000/health > /dev/null
check_success "Health endpoint test"

# Test prediction endpoint
echo "Testing prediction endpoint..."
response=$(curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [15.0, 20.0, 100.0, 500.0, 0.1, 0.09, 0.03, 0.02, 0.18, 0.06]}')

echo "Prediction response: $response"

if echo "$response" | grep -q "prediction"; then
    echo "✅ Prediction endpoint test successful"
else
    echo "❌ Prediction endpoint test failed"
    kill $API_PID
    exit 1
fi

# Stop API server
echo "Stopping API server..."
kill $API_PID
echo "✅ API server stopped"

# Stage 5: Docker Build Test
print_status "Stage 5: Docker Build Test"
echo "Building Docker image..."
docker build -t ml-model-test:latest .
check_success "Docker build"

# Test Docker container
echo "Testing Docker container..."
docker run --rm -d --name ml-test-container -p 5001:5000 ml-model-test:latest

# Wait for container to start
sleep 10

# Test containerized API
echo "Testing containerized API..."
curl -f http://localhost:5001/health > /dev/null
check_success "Docker container health check"

# Stop test container
docker stop ml-test-container
echo "✅ Docker container test completed"

# Clean up Docker image
docker rmi ml-model-test:latest

print_status "Pipeline Test Summary"
echo "=============================="
echo "✅ All pipeline stages completed successfully!"
echo "🚀 Ready for production deployment"

echo ""
echo "📊 Model Performance Summary:"
python -c "
import json
with open('models/model_metrics.json', 'r') as f:
    metrics = json.load(f)
for metric, value in metrics.items():
    if metric != 'timestamp':
        print(f'  {metric}: {value:.4f}')
"

echo ""
echo "🔗 Next Steps:"
echo "  1. Commit and push code to trigger CI/CD pipeline"
echo "  2. Monitor pipeline execution in GitHub Actions"
echo "  3. Review deployment to staging environment"
echo "  4. Approve production deployment when ready"