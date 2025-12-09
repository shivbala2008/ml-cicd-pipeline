import subprocess
import os
import json
import time
import signal
import sys
import psutil
import requests
import re

# --- Configuration ---
# 1. FIX: Explicitly define the path to the Python interpreter within the VENV
# This ensures Windows uses the interpreter where all tools (black, pytest, etc.) are installed.
# Assumes the script is run from the project root (C:\Users\sbalak015\ml-cicd-pipeline)
PYTHON_INTERPRETER = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe')

API_PORT_LOCAL = 5000
API_PORT_DOCKER = 5001
PREDICTION_DATA = {"features": [15.0, 20.0, 100.0, 500.0, 0.1, 0.09, 0.03, 0.02, 0.18, 0.06]}
API_PROCESS = None
DOCKER_CONTAINER_NAME = "ml-test-container"
DOCKER_IMAGE_TAG = "ml-model-test:latest"

# --- Utility Functions ---

def print_status(message):
    """Prints a formatted status message."""
    print(f"\n📋 {message}")

def execute_command(command, success_message, fail_message, check_output=False):
    """Executes a shell command and checks its return code."""
    # Use the first element of the command list as the process name for logging
    cmd_log = ' '.join(command) 
    print(f"Executing: {cmd_log}")
    try:
        if check_output:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✅ {success_message} successful")
            return result.stdout
        else:
            # Check=True will raise CalledProcessError if the command fails (non-zero exit code)
            subprocess.run(command, check=True)
            print(f"✅ {success_message} successful")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ {fail_message} failed")
        print(f"Error details: {e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else 'No stderr output'}")
        sys.exit(1)

def kill_process(pid):
    """Gracefully terminates a process by PID, cross-platform."""
    try:
        # Use psutil to kill the process tree to ensure all sub-processes are terminated
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        parent.wait(timeout=5)
        print(f"✅ Process {pid} stopped")
    except psutil.NoSuchProcess:
        print(f"⚠️ Process {pid} was already terminated.")
    except Exception as e:
        print(f"⚠️ Could not kill process {pid}: {e}")

def run_pipeline():
    global API_PROCESS

    print("🧪 Testing Complete ML CI/CD Pipeline Locally")
    print("==============================================")

    # --- Stage 1: Code Quality Checks ---
    print_status("Stage 1: Code Quality Checks")
    
    # FIX: Use PYTHON_INTERPRETER to run black
    print("Checking code formatting...")
    black_command = [PYTHON_INTERPRETER, "-m", "black", "--check", "src/", "tests/"]
    try:
        execute_command(black_command, "Code formatting", "Code formatting")
    except SystemExit:
        # Custom exit logic for black --check failure
        print("Run 'python -m black src/ tests/' to fix formatting and commit the changes.")
        sys.exit(1)

    # FIX: Use PYTHON_INTERPRETER to run flake8
    print("Linting code...")
    flake8_command = [PYTHON_INTERPRETER, "-m", "flake8", "src/", "tests/", "--max-line-length=100", "--ignore=E203,W503"]
    execute_command(flake8_command, "Code linting", "Code linting")

    # --- Stage 2: Unit Tests ---
    print_status("Stage 2: Unit Tests")
    
    # FIX: Use PYTHON_INTERPRETER to run pytest
    print("Running unit tests...")
    pytest_command = [PYTHON_INTERPRETER, "-m", "pytest", "tests/", "-v", "--cov=src"]
    execute_command(pytest_command, "Unit tests", "Unit tests")

    # --- Stage 3: Model Training ---
    print_status("Stage 3: Model Training")
    
    # FIX: Use PYTHON_INTERPRETER to run src/train.py
    print("Training ML model...")
    train_command = [PYTHON_INTERPRETER, "src/train.py"]
    execute_command(train_command, "Model training", "Model training")

    # Validate model artifacts
    if not os.path.exists("models/model.joblib"):
        print("❌ Model file not found")
        sys.exit(1)
    print("✅ Model file created")

    if not os.path.exists("models/model_metrics.json"):
        print("❌ Model metrics file not found")
        sys.exit(1)
    print("✅ Model metrics file created")

    # FIX: Use PYTHON_INTERPRETER to format and print metrics
    print("Metrics content:")
    metrics_display_command = [
        PYTHON_INTERPRETER, "-c",
        "import json, sys; d=json.load(sys.stdin); print(json.dumps(d, indent=4))"
    ]
    with open("models/model_metrics.json", 'r') as f:
        metrics_data = json.load(f)
        print(json.dumps(metrics_data, indent=4))

    # --- Stage 4: Integration Tests ---
    print_status("Stage 4: Integration Tests")
    print("Starting API server for integration testing...")

    # FIX: Use PYTHON_INTERPRETER for API process
    try:
        API_PROCESS = subprocess.Popen([PYTHON_INTERPRETER, "src/predict.py"], start_new_session=True)
        print(f"API started with PID: {API_PROCESS.pid}")
        print("Waiting for API to start...")
        time.sleep(5) 
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        sys.exit(1)

    # Test health endpoint (using requests, as planned)
    health_url = f"http://localhost:{API_PORT_LOCAL}/health"
    try:
        requests.get(health_url, timeout=5).raise_for_status()
        print("✅ Health endpoint test successful")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint test failed: {e}")
        kill_process(API_PROCESS.pid)