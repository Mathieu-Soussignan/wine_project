import pandas as pd

def load_data():
    # Charger les fichiers CSV
    red_wine = pd.read_csv("data/winequality-red.csv", sep=";")
    white_wine = pd.read_csv("data/winequality-white.csv", sep=";")
    
    # Ajouter une colonne pour différencier les types de vin
    red_wine["wine_type"] = "red"
    white_wine["wine_type"] = "white"
    
    # Combiner les deux ensembles
    data = pd.concat([red_wine, white_wine], axis=0).reset_index(drop=True)
    return data