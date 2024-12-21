# Prédiction de la qualité du vin

## Présentation du projet

Ce projet a pour objectif de prédire la qualité d'un vin à partir de ses caractéristiques physico-chimiques. Le modèle prédictif est basé sur l'utilisation de **LightGBM**, un algorithme de boosting décisionnel performant et adapté aux problèmes de classification avec des classes déséquilibrées. Le projet inclut également un prétraitement des données pour s'assurer de la qualité des entrées et pour gérer les classes sous-représentées via des techniques comme **SMOTE**.

## Contenu du projet

### 1. Fichiers principaux

- **main.py** : Script principal pour l'entraînement du modèle.
- **predict.py** : Script permettant de prédire la qualité des vins sur de nouvelles données.
- **src/model.py** : Contient les fonctions pour entraîner, évaluer, sauvegarder et charger le modèle.
- **src/preprocess.py** : Inclut les étapes de prétraitement des données, notamment la gestion des classes et l'application de SMOTE.
- **data/clean_csv.py** : Script pour nettoyer les fichiers CSV afin de s'assurer que les colonnes sont correctement formatées.
- **data/** : Dossier contenant les fichiers de données (échantillons d'entraînement et de test).
- **src/models/** : Dossier pour stocker les modèles sauvegardés.
- **reports/** : Contient les rapports de classification et les fichiers de visualisation.

### 2. Données

Les données utilisées proviennent de l'ensemble de données « Wine Quality ». Ces données incluent des caractéristiques comme l'acidité, la teneur en alcool, le pH, etc. Les colonnes sont :

- `fixed_acidity`
- `volatile_acidity`
- `citric_acid`
- `residual_sugar`
- `chlorides`
- `free_sulfur_dioxide`
- `total_sulfur_dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (colonne cible)

### 3. Prétraitement des données

#### Regroupement des classes
Les classes de qualité sont regroupées en trois catégories :
- **Faible** : `quality` = 3 ou 4
- **Moyenne** : `quality` = 5 ou 6
- **Haute** : `quality` = 7 ou plus

#### Techniques appliquées
- **Encodage des colonnes** : Transformation des colonnes catégoriques en valeurs numériques.
- **SMOTE** : Rééchantillonnage pour gérer le déséquilibre des classes.

### 4. Modélisation

#### Algorithme choisi : LightGBM

LightGBM a été choisi pour sa rapidité et sa capacité à gérer les classes déséquilibrées. Une recherche d’hyperparamètres a été effectuée via **GridSearchCV** pour optimiser les performances du modèle.

#### Hyperparamètres optimisés
- Nombre d'arbres (`n_estimators`) : 200
- Profondeur maximale (`max_depth`) : 7
- Taux d'apprentissage (`learning_rate`) : 0.1
- Proportion d'échantillons pour chaque arbre (`subsample`) : 0.8
- Proportion des caractéristiques par arbre (`colsample_bytree`) : 1.0

#### Évaluation du modèle
- **Métriques principales** : Précision globale, précision par classe, rappel, et score F1.
- Le modèle a montré une précision globale de **83,3%** sur les données de test après optimisation.

### 5. Prédiction

Le fichier `predict.py` permet de prédire la qualité des vins sur de nouvelles données. Les étapes incluent :
- Chargement du modèle sauvegardé.
- Prétraitement des nouvelles données pour aligner les colonnes sur celles du modèle.
- Génération de prédictions et comparaison avec les labels réels (si disponibles).

### 6. Organisation du code

- **Clarté** : Le projet est structuré de manière à séparer les responsabilités (modélisation, prétraitement, prédiction).
- **Modularité** : Les fonctions sont réutilisables et peuvent être testées indépendamment.

### 7. Instructions d'installation

#### Prérequis
- Python 3.10+
- Pip
- Librairies Python : `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `imblearn`, `joblib`

#### Installation
1. Clonez le répertoire du projet :
   ```bash
   git clone <URL-du-dépôt>
   cd Wine
   ```
2. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Pour Linux/Mac
   .venv\Scripts\activate    # Pour Windows
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

#### Lancer le projet
- Pour entraîner le modèle :
  ```bash
  python main.py
  ```
- Pour effectuer des prédictions :
  ```bash
  python predict.py
  ```

### 8. Améliorations futures

- **Exploration d'autres modèles** : Random Forest, XGBoost.
- **Ajout de visualisations** : Analyse des importances des caractéristiques avec SHAP.
- **Optimisation avancée** : Utilisation de techniques comme BayesSearchCV pour optimiser les hyperparamètres.
- **Interface utilisateur** : Création d'une interface graphique pour faciliter les prédictions.