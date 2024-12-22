from src.model import train_lightgbm_optimized, evaluate_lightgbm, save_model
from src.preprocess import preprocess_data_with_smote_and_grouping
import pandas as pd

if __name__ == "__main__":
    try:
        # Charger les données avec un séparateur correct
        data = pd.read_csv("data//raw/winequality-white.csv", sep=";", quotechar='"')

        # Vérifier les colonnes après nettoyage
        print("Colonnes nettoyées :", data.columns.tolist())

        # Diagnostiquer la présence de 'quality'
        if "quality" not in data.columns:
            raise KeyError("La colonne 'quality' est absente des données. Vérifiez le fichier CSV.")

        # Prétraiter les données
        X_train, X_test, y_train, y_test = preprocess_data_with_smote_and_grouping(data)

        # Diagnostics des données retournées
        print("Dimensions de X_train :", X_train.shape if X_train is not None else None)
        print("Dimensions de y_train :", len(y_train) if y_train is not None else None)
        print("Type de y_train :", type(y_train))

        # Entraîner le modèle
        model = train_lightgbm_optimized(X_train, y_train)

        # Sauvegarder le modèle
        save_model(model, filename="lightgbm_grouped_model.pkl")

        # Évaluer le modèle
        accuracy, report = evaluate_lightgbm(model, X_test, y_test)
        print("Précision globale :", accuracy)
        print("Rapport de classification :\n", report)

    except Exception as e:
        print("Erreur détectée :", str(e))