import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Exports:
        Model and DictVectorizer to Mlflow
    """

    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    experiment_name = "module3"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
     
        dv, lr = data
     

        with open("dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)

        mlflow.log_artifact("dict_vectorizer.pkl")
        mlflow.sklearn.log_model(lr, artifact_path='artifact')