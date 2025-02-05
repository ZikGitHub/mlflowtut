import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_images", experiment_id=experiment.experiment_id) as run:

        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)


        # precision-recall curve
        fig_pr = plt.figure()
        pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax= plt.gca())
        plt.title("Precision-Recall Curve")
        plt.legend()

        # ROC curve
        fig_roc = plt.figure()
        roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax= plt.gca())
        plt.title("ROC Curve")
        plt.legend()

        # confusion matrix
        fig_cm = plt.figure()
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax= plt.gca())
        plt.title("Confusion Matrix")
        plt.legend()

        # log images
        mlflow.log_figure(fig_pr, "metrics/precision_recall.png")
        mlflow.log_figure(fig_roc, "metrics/roc_curve.png")
        mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")

        # print info
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print('status: {}'.format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))


