import mlflow

mlflow.set_tracking_uri("sqlite:///C:/Users/nagen/OneDrive/Desktop/Kaggle datasets/Market basket analysis/market-basket-mlflow/mlflow.db")
mlflow.set_experiment("TestExperiment")

with mlflow.start_run():
    mlflow.log_param("foo", "bar")
    mlflow.log_metric("test_metric", 123)
print(">>> Finished logging test run.")

