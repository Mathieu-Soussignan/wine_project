# Rapport de projet : Prédiction de la qualité des vins

## Introduction
L’objectif de ce projet était de construire un modèle prédictif capable d’évaluer la qualité des vins à partir de différentes caractéristiques chimiques. Le modèle doit fournir des prédictions fiables tout en gérant les déséquilibres de classes présents dans les données.

Ce rapport détaille les étapes suivies, les différents modèles testés, les choix finaux effectués, et les résultats obtenus.

---

## 1. Exploration des données
### 1.1 Description des données
Les données sont issues d'un jeu de données évaluant la qualité des vins rouges et blancs. Chaque ligne correspond à un échantillon de vin, avec les colonnes suivantes :
- **Caractéristiques chimiques** : acide fixe, acidité volatile, taux de chlorure, pH, etc.
- **Cible** : une note de qualité allant de 3 à 9, attribuée par des experts.

### 1.2 Préparation des données
- Les colonnes ont été nettoyées pour garantir une cohérence entre les noms présents dans les fichiers et les attentes du code.
- Les classes de qualité ont été regroupées pour simplifier le problème :
  - 3 et 4 : **faible**
  - 5 et 6 : **moyenne**
  - 7, 8 et 9 : **haute**

### 1.3 Analyse exploratoire
- Déséquilibre notable des classes : la majorité des échantillons appartient à la classe "moyenne".
- Présence de corrélations entre certaines variables chimiques et la qualité (ex. taux d’alcool).

---

## 2. Prétraitement des données
### 2.1 Techniques appliquées
1. **Encodage** : Les colonnes de type catégorique (ex. "type" pour les vins rouges et blancs) ont été encodées en valeurs numériques.
2. **Équilibrage des classes avec SMOTE** :
   - SMOTE (“Synthetic Minority Oversampling Technique”) a été utilisé pour créer des échantillons synthétiques dans les classes minoritaires.

### 2.2 Justification
- L’équilibrage était nécessaire pour éviter que le modèle favorise uniquement la classe majoritaire ("moyenne").
- Les regroupements ont réduit la complexité du problème tout en conservant une granularité suffisante.

---

## 3. Modélisation
### 3.1 Modèles testés
Plusieurs modèles ont été testés pour évaluer leurs performances sur ce problème :

#### a. **Random Forest**
- **Avantages** : robuste aux valeurs aberrantes, facile à interpréter.
- **Inconvénients** : temps d’exécution plus long avec des jeux de données volumineux.
- **Résultats** : précision globale de ~52 %, mais sous-performe sur les classes minoritaires.

#### b. **LightGBM**
- **Avantages** :
  - Conçu pour traiter des données avec des déséquilibres.
  - Performances élevées avec une exécution rapide.
- **Inconvénients** : peut surchauffer si les paramètres ne sont pas optimisés.
- **Résultats initiaux** :
  - Précision globale de ~65 %.
  - Sous-performances liées à un manque d’optimisation.

#### c. **Réseaux de neurones**
- **Avantages** : capable de capturer des relations complexes.
- **Inconvénients** : nécessite davantage de données et de temps d’entraînement.
- **Résultats** : performances inférieures (précision ~40 %), dû au faible volume de données et à l’absence de réglages avancés.

### 3.2 Modèle final choisi : **LightGBM**
**Justifications** :
1. Résultats supérieurs avec un temps d’exécution raisonnable.
2. Capacité à gérer les classes déséquilibrées grâce à la pondération des classes (“class_weight=balanced”).
3. Flexibilité pour l’optimisation des hyperparamètres.

---

## 4. Optimisation et évaluation
### 4.1 Optimisation des hyperparamètres
- **Méthode** : `GridSearchCV` pour rechercher les meilleurs paramètres parmi :
  - Nombre d’arbres (“ n_estimators ”) : [100, 150, 200]
  - Profondeur maximale (“ max_depth ”) : [3, 5, 7]
  - Taux d’apprentissage (“ learning_rate ”) : [0.05, 0.1]
  - Fraction d’échantillons utilisée (“ subsample ”) : [0.8, 1.0]
  - Fraction des features utilisées (“ colsample_bytree ”) : [0.8, 1.0]

- **Meilleurs paramètres** :
  - `n_estimators`: 200
  - `max_depth`: 7
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 1.0

### 4.2 Évaluation des performances
- **Précision globale** : 83 %.
- **Rapport de classification** :
  - Classe faible : précision de 52 % (progrès significatifs grâce à SMOTE).
  - Classe moyenne : 72 %.
  - Classe haute : 88 %.
- **Analyse** :
  - Les performances sont élevées pour la classe dominante (“ haute ”).
  - Les classes minoritaires sont mieux représentées grâce à SMOTE, mais les erreurs persistent.

---

## 5. Déploiement et prédictions sur de nouvelles données
### 5.1 Pipeline de prédiction
1. Chargement du modèle LightGBM optimisé.
2. Nettoyage des nouvelles données (alignement des colonnes).
3. Prédictions et génération de rapports.

### 5.2 Exemple de prédictions
- **Données testées** : un ensemble d’échantillons représentatifs des trois classes.
- **Résultats** :
  - Précision globale des prédictions : 80 %.
  - Rapport de classification :
    - Classe faible : sous-représentation dans les prédictions (dû à l’équilibre difficile).
    - Classe moyenne et haute : performances solides.

---

## 6. Conclusions et recommandations
### 6.1 Conclusions
1. Le modèle LightGBM s’est révélé le plus adapté, avec des performances globales élevées et un temps de traitement optimisé.
2. L’équilibrage avec SMOTE a amélioré la précision des classes minoritaires, bien que des améliorations soient encore possibles.

### 6.2 Recommandations
1. **Collecte de données** : Augmenter la proportion des classes minoritaires pour affiner le modèle.
2. **Exploration d’autres modèles** : Tester des techniques comme le XGBoost ou des réseaux de neurones avec plus de données.
3. **Visualisation** : Intégrer des visualisations SHAP.