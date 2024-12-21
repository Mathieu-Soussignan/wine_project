from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib  # Pour sauvegarder et charger le modèle

def train_lightgbm_optimized(X_train, y_train):
    """
    Entraîne un modèle LightGBM avec des hyperparamètres optimisés.
    """
    # Vérification des données
    print("Vérification des données d'entraînement...")
    print("Type de X_train :", type(X_train))
    print("Dimensions de X_train :", X_train.shape)
    print("Type de y_train :", type(y_train))
    print("Dimensions de y_train :", len(y_train))
    print("Valeurs uniques dans y_train :", set(y_train))

    # Définir les hyperparamètres à tester
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # Initialiser LightGBM
    lgbm = LGBMClassifier(class_weight="balanced", random_state=42)

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # Lancer la recherche des hyperparamètres
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        print("Erreur pendant l'entraînement avec GridSearchCV :", str(e))
        raise

    # Retourner le meilleur modèle
    print("Meilleurs hyperparamètres :", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_lightgbm(model, X_test, y_test):
    """
    Évalue le modèle LightGBM sur les données de test.
    """
    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul de la précision et du rapport de classification
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()

    return accuracy, report

def save_model(model, filename="lightgbm_model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")

def load_model(filename="lightgbm_model.pkl"):
    """
    Charge un modèle sauvegardé depuis un fichier.
    """
    return joblib.load(filename)

def cross_validate_model(X_train, y_train):
    """
    Effectue une validation croisée pour évaluer les performances du modèle.
    """
    lgbm = LGBMClassifier(class_weight="balanced", random_state=42)
    scores = cross_val_score(lgbm, X_train, y_train, cv=5, scoring="f1_weighted")
    print("Validation croisée - Scores :", scores)
    print("Validation croisée - Score moyen :", scores.mean())