import pandas as pd
from sklearn.metrics import classification_report
from src.model import load_model
from src.preprocess import preprocess_data_with_smote_and_grouping

def predict_new_data(model_path, data_path):
    """
    Prédit les classes pour de nouvelles données.
    """
    # Charger le modèle
    model = load_model(model_path)

    # Charger les données pour la prédiction
    new_data = pd.read_csv(data_path)

    # Prétraiter les données (séparer en entraînement/test)
    _, X_test, _, y_test = preprocess_data_with_smote_and_grouping(new_data)

    # Vérifier les colonnes du modèle et des données
    print("Colonnes du modèle :", model.feature_name_)
    print("Colonnes des données :", X_test.columns.tolist())

    # Aligner les colonnes des données sur celles du modèle
    X_test = X_test[model.feature_name_]

    # Effectuer les prédictions
    predictions = model.predict(X_test)
    return predictions, y_test

if __name__ == "__main__":
    # Chemins des fichiers
    model_path = "lightgbm_grouped_model.pkl"
    data_path = "data/cleaned_winequality-red.csv"

    # Effectuer les prédictions
    predictions, true_labels = predict_new_data(model_path, data_path)
    print("Prédictions :", predictions)

    # Vérification des dimensions
    if len(predictions) != len(true_labels):
        raise ValueError(
            f"Les dimensions des prédictions ({len(predictions)}) et des labels réels ({len(true_labels)}) ne correspondent pas."
        )

    # Comparaison des prédictions et des labels réels
    print("\nComparaison des prédictions et des labels réels :")
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        print(f"Exemple {i + 1} : Prédiction = {pred}, Réel = {true}")

    # Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(true_labels, predictions, target_names=["faible", "moyenne", "haute"]))