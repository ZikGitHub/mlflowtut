import mlflow

if __name__ == "__main__":

    with mlflow.start_run(run_name="mlflow_runs") as run:

        # your machine learning code
        mlflow.log_param("learning_rate", 0.01)
        print ("RUN ID: {}".format(run.info.run_id))
        print(run.info)