# ⬡ DQGraph

**Graph Data Quality Platform** — application [Streamlit](https://streamlit.io) pour évaluer la qualité des données d'une base de graphes [Neo4j](https://neo4j.com) et mesurer l'impact de cette qualité sur des tâches de Graph Machine Learning.

## Présentation

DQGraph se connecte à une base Neo4j, calcule un ensemble de métriques de qualité de données (complétude, conformité, fraîcheur, unicité, exactitude) sur les nœuds/relations choisis, puis permet de dégrader artificiellement ces données pour observer l'impact sur les performances de modèles de Graph ML (classification de nœuds, régression de propriétés, prédiction de liens). Les résultats sont persistés par projet sur disque.

## Fonctionnalités

### Connexion & gestion de projets
- Connexion à une instance Neo4j (URI, utilisateur, mot de passe) avec listing des bases disponibles.
- Création / ouverture de **projets** : chaque projet conserve l'historique des métriques de qualité calculées et un cache des expériences de ML.

### Onglet « Data Quality Insights »
Calcule et affiche, par dimension, les métriques de qualité sur un label de nœud, un type de relation ou une propriété donnée :

| Dimension | Métriques |
|---|---|
| Completeness | Property Completeness, Edge Coverage, Average Type Completeness |
| Conformity | Property Conformity, Conditional Property Conformity, Average Type Conformity |
| Freshness | Average Type Age, Average Property Age, Update Frequency |
| Uniqueness | Exact Duplicate Rate, Partial Duplicate Rate, Uniqueness Score |
| Accuracy | Property Accuracy |

Les résultats s'accumulent dans le projet courant et sont regroupés par dimension dans des tableaux récapitulatifs.

### Onglet « Graph ML Impact »
1. Sélection d'une tâche de Graph ML : classification de nœuds, régression de propriété, ou prédiction de liens.
2. Choix d'une dimension de dégradation (Property Completeness, Edge Coverage, Property Accuracy) et de jusqu'à 3 algorithmes scikit-learn par tâche.
3. Extraction des données Neo4j vers pandas, sauvegarde/restauration des données d'origine, puis dégradation progressive (λ de 0.0 à 0.9, pas de 0.1).
4. Entraînement et évaluation locale (sans Neo4j GDS) à chaque niveau de dégradation.
5. Visualisation : courbe de la métrique de performance en fonction de λ, une courbe par algorithme. Les résultats sont mis en cache par configuration dans le dossier du projet.

### Modules additionnels (Neo4j GDS)
Le dossier `ml/` contient également des pipelines basés sur le plugin **Graph Data Science** de Neo4j (`community_detection.py`, `link_prediction.py`, `node_classification.py`, `node_regression.py`), ainsi que des composants UI associés (`ui/forms.py`, `ui/degraders_ui.py`, `ui/link_prediction_ui.py`). Ils ne sont pas branchés sur les deux onglets principaux de `app.py` mais peuvent être réutilisés/intégrés pour des analyses s'appuyant sur GDS.

## Architecture

```
app.py                     Point d'entrée : login → sélection de projet → onglets
analysers.py                Calcul des métriques de qualité (Freshness, Conformity, Completeness, Uniqueness, Accuracy)
degraders.py                 Dégradation réversible des données (Completeness, Edge Coverage, Accuracy)
project_manager.py          Persistance des projets (~/graphlens_projects)
pages/
  tab_quality.py             Onglet "Data Quality Insights"
  tab_graphml.py              Onglet "Graph ML Impact"
ui/
  schema.py                   Exploration dynamique du schéma Neo4j
  charts.py                   Jauges et graphiques Plotly
  forms.py, degraders_ui.py, link_prediction_ui.py   Composants UI additionnels (non branchés par défaut)
  styles.py                   CSS partagé
ml/
  ml_sklearn.py                Extraction Neo4j → pandas + entraînement scikit-learn (utilisé par tab_graphml)
  community_detection.py, link_prediction.py, node_classification.py, node_regression.py   Pipelines Neo4j GDS
```

## Prérequis

- Python 3.12+
- Une instance Neo4j accessible (Bolt), `bolt://localhost:7687` par défaut
- Les pipelines du dossier `ml/` basés sur GDS nécessitent le plugin **Neo4j Graph Data Science**

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install streamlit neo4j pandas numpy scipy scikit-learn plotly
```

## Lancement

```bash
source venv/bin/activate
streamlit run app.py
```

L'application s'ouvre sur la page de connexion : renseigner l'URI Neo4j, l'utilisateur et le mot de passe, puis créer ou ouvrir un projet.

## Stockage des projets

Chaque projet est un dossier sous `~/graphlens_projects/<nom_du_projet>/` contenant :
- `project.json` — métadonnées et historique des métriques de qualité calculées
- `ml_cache/` — un fichier JSON par expérience de Graph ML (clé = hash de la configuration)
