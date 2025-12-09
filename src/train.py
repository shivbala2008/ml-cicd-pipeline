import json
import os
import sys
from datetime import datetime

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, config_path="config/model_config.yaml"):
        """Initialize model trainer with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.metrics = {}

    def load_config(self, config_path):
        """Load training configuration"""
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            return {
                "model": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                "training": {"test_size": 0.2, "random_state": 42},
                "quality_gates": {
                    "min_accuracy": 0.85,
                    "min_precision": 0.80,
                    "min_recall": 0.80,
                    "min_f1": 0.80,
                },
            }

    def load_data(self):
        """Load and prepare training data"""
        print("📊 Loading training data...")

        # Load breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data[:, :10], columns=data.feature_names[:10])
        y = pd.Series(data.target)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["training"]["test_size"],
            random_state=self.config["training"]["random_state"],
            stratify=y,
        )

        print(
            f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples"
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train the ML model"""
        print("🎯 Training model...")

        # Initialize model with config parameters
        self.model = RandomForestClassifier(
            n_estimators=self.config["model"]["n_estimators"],
            max_depth=self.config["model"]["max_depth"],
            random_state=self.config["model"]["random_state"],
        )

        # Train model
        self.model.fit(X_train, y_train)
        print("✅ Model training completed")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("📈 Evaluating model performance...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "timestamp": datetime.now().isoformat(),
        }

        print("📊 Model Performance:")
        for metric, value in self.metrics.items():
            if metric != "timestamp":
                print(f"  {metric}: {value: .4f}")

        return self.metrics

    def check_quality_gates(self):
        """Check if model passes quality gates"""
        print("🚪 Checking quality gates...")

        gates = self.config["quality_gates"]
        passed = True
        failed_gates = []

        for gate, threshold in gates.items():
            metric_name = gate.replace("min_", "")
            if metric_name in self.metrics:
                if self.metrics[metric_name] < threshold:
                    passed = False
                    failed_gates.append(
                        f"{metric_name}: {self.metrics[metric_name]:.4f} < {threshold}"
                    )
                    print(
                        f"❌ {metric_name}: {self.metrics[metric_name]:.4f} < {threshold}"
                    )
                else:
                    print(
                        f"✅ {metric_name}: {self.metrics[metric_name]:.4f} >= {threshold}"
                    )

        if not passed:
            raise ValueError(f"Quality gates failed: {', '.join(failed_gates)}")

        print("✅ All quality gates passed!")
        return passed

    def save_model(self, model_path="models/model.joblib"):
        """Save trained model"""
        print(f"💾 Saving model to {model_path}...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        joblib.dump(self.model, model_path)

        # Save metrics
        metrics_path = model_path.replace(".joblib", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print("✅ Model and metrics saved")
        return model_path

    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 Starting ML training pipeline...")

        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()

            # Train model
            self.train_model(X_train, y_train)

            # Evaluate model
            self.evaluate_model(X_test, y_test)

            # Check quality gates
            self.check_quality_gates()

            # Save model
            model_path = self.save_model()

            print("🎉 Training pipeline completed successfully!")
            return model_path, self.metrics

        except Exception as e:
            print(f"❌ Training pipeline failed: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    trainer = ModelTrainer()
    model_path, metrics = trainer.run_training_pipeline()
    print(f"Final model saved at: {model_path}")
