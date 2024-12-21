import pandas as pd

# Définir les colonnes attendues et leur correspondance pour le modèle
expected_columns = {
    'fixed acidity': 'fixed_acidity',
    'volatile acidity': 'volatile_acidity',
    'citric acid': 'citric_acid',
    'residual sugar': 'residual_sugar',
    'chlorides': 'chlorides',
    'free sulfur dioxide': 'free_sulfur_dioxide',
    'total sulfur dioxide': 'total_sulfur_dioxide',
    'density': 'density',
    'pH': 'pH',
    'sulphates': 'sulphates',
    'alcohol': 'alcohol',
    'quality': 'quality'
}

# Charger le fichier avec les paramètres corrects
data = pd.read_csv("data/winequality-red.csv", sep=";", quotechar='"')

# Afficher les colonnes présentes
print("Colonnes présentes :", data.columns.tolist())

# Vérifier si toutes les colonnes attendues sont présentes
missing_columns = [col for col in expected_columns.keys() if col not in data.columns]
if missing_columns:
    raise KeyError(f"Les colonnes suivantes sont manquantes dans le fichier : {missing_columns}")

# Renommer les colonnes pour correspondre aux noms utilisés par le modèle
data = data.rename(columns=expected_columns)

# Sauvegarder le fichier nettoyé
cleaned_file_path = "data/cleaned_winequality-red.csv"
data.to_csv(cleaned_file_path, index=False)
print(f"Fichier nettoyé et sauvegardé sous '{cleaned_file_path}'.")