import mlflow
from mlflow_utils import create_mlflow_experiment

experiment_id = create_mlflow_experiment(
    experiment_name = "Nested Runs",
    artifact_location= "nested_runs_artifacts",
    tags={"purpose":"learning"}
)

with mlflow.start_run(run_name="parent") as parent:
    print("Run ID parent:", parent.info.run_id)
    mlflow.log_param("parent_param", "parent_value")

    with mlflow.start_run(run_name="child1", nested=True) as child1:
        print("Run ID child1", child1.info.run_id)
        mlflow.log_param("child1_param", "child1_value")

        with mlflow.start_run(run_name="child11", nested=True) as child11:
            print("Run ID child11", child11.info.run_id)
            mlflow.log_param("child11_param", "child11_value")

        with mlflow.start_run(run_name="child12", nested=True) as child12:
            print("Run ID child12", child12.info.run_id)
            mlflow.log_param("child12_param", "child12_value")

    with mlflow.start_run(run_name="child2", nested=True) as child2:
        print("Run ID child2", child2.info.run_id)
        mlflow.log_param("child2_param", "child2_value")