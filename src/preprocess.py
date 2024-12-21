from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

def preprocess_data_with_smote_and_grouping(data):
    """
    Prépare les données pour l'entraînement ou l'inférence :
    - Regroupe les classes si 'quality' est présent.
    - Prépare les données pour l'inférence si 'quality' est absent.
    """
    # Diagnostiquer la colonne 'quality'
    print("Colonnes disponibles dans les données :", data.columns.tolist())

    if "quality" in data.columns:
        # Regrouper les classes de qualité
        def group_quality(quality):
            if quality in [3, 4]:
                return "faible"   # Classe faible
            elif quality in [5, 6]:
                return "moyenne"  # Classe moyenne
            else:
                return "haute"    # Classe haute

        data["quality"] = data["quality"].apply(group_quality)

        # Vérifier si les regroupements ont fonctionné
        print("Distribution des classes regroupées :")
        print(data["quality"].value_counts())
    else:
        print("Aucune colonne 'quality' détectée. Passage en mode inférence.")

    # Encoder les colonnes catégoriques
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Si 'quality' est présent, préparer pour l'entraînement
    if "quality" in data.columns:
        X = data.drop("quality", axis=1)
        y = data["quality"]

        # Vérifier les dimensions de X et y
        print("Dimensions de X :", X.shape)
        print("Dimensions de y :", len(y))

        # Séparer les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Appliquer SMOTE pour équilibrer les classes
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Vérifier les dimensions après SMOTE
        print("Dimensions après SMOTE - X_train :", X_train_smote.shape)
        print("Dimensions après SMOTE - y_train :", y_train_smote.shape)

        return X_train_smote, X_test, y_train_smote, y_test

    # Si 'quality' est absent (cas d'inférence), retourner uniquement les caractéristiques
    return data, None, None, None