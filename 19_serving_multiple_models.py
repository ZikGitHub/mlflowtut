import mlflow
from mlflow_utils import create_mlflow_experiment
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
import numpy as np
import pandas as pd

class CustomModel(mlflow.pyfunc.PythonModel):
    def predict_model1(self, model_input):
        return 0 * model_input

    def predict_model2(self, model_input):
        return model_input

    def predict_model3(self, model_input):
        return 2 * model_input

    def predict(self, context, model_input):
        # Extract model_name from input data
        if "model_name" not in model_input.columns:
            raise Exception("Model name must be specified in input data")

        model_name = model_input["model_name"].iloc[0]  # Get first row value
        model_input_value = model_input["input"].iloc[0]  # Extract actual input

        if model_name == "model_1":
            return self.predict_model1(model_input_value)
        elif model_name == "model_2":
            return self.predict_model2(model_input_value)
        elif model_name == "model_3":
            return self.predict_model3(model_input_value)
        else:
            raise Exception("Model Not Found!")

if __name__ == "__main__":
    experiment_id = create_mlflow_experiment(
        experiment_name="Serving Multiple Models",
        artifact_location="serving_multiple_models",
        tags={"purpose": "learning"},
    )

    input_schema = Schema([
        ColSpec(type="integer", name="input"), 
        ColSpec(type="string", name="model_name")  # Include model_name in schema
    ])
    output_schema = Schema([ColSpec(type="integer", name="output")])

    model_signature = ModelSignature(
        inputs=input_schema, 
        outputs=output_schema
    )

    with mlflow.start_run(run_name="multiple_models", experiment_id=experiment_id) as run:
        mlflow.pyfunc.log_model(
            artifact_path="model", 
            python_model=CustomModel(), 
            signature=model_signature
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

        for n in range(3):
            print(f"PREDICTION FROM MODEL {n+1}")
            model_input = pd.DataFrame({
                "input": [np.int32(10)], 
                "model_name": [f"model_{n+1}"]  # Include model_name inside input
            })
            print(loaded_model.predict(model_input))  # Remove params argument
            print("\n")

        print(f"RUN_ID: {run.info.run_id}")
