import mlflow

if __name__ == "__main__":
    # create a new mlflow experiment
    experiment_id = mlflow.create_experiment(
        name="testing_mlflow1",
        artifact_location="testing_mlflow_artifacts",
        tags={"env": "dev", "version": "1.0.0"},

    )

    # print the experiment id
    print(experiment_id)