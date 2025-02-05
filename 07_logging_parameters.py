import mlflow
from mlflow_utils import get_mlflow_experiment


if __name__ == "__main__":
    
    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    if experiment is not None:
        print("Name: {}".format(experiment.name))
    else:
        print("Experiment 'testing_mlflow1' not found.")

    with mlflow.start_run(run_name="logging_params", experiment_id=experiment.experiment_id) as run:

        # your machine learning code
        mlflow.log_param("learning_rate", 0.01)

        parameters = {
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 100,
            "loss_funciton": "mse",
            "optimizer": "adam"
        }

        mlflow.log_params(parameters)

        # print info
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print('status: {}'.format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))

