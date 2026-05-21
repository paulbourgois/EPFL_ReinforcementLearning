# Cartpole pipeline

Ce dossier contient le pipeline complet pour la partie Cartpole du projet.

## Ordre d'exécution

1. `find_expert_policy.py`
2. `generate_trajectory.py`
3. `visualize.py`

Exemple dans PowerShell, depuis la racine du projet:

```powershell
conda activate imitlearn
python .\Cartpole\find_expert_policy.py
python .\Cartpole\generate_trajectory.py
python .\Cartpole\visualize.py
```

## Rôle de chaque script

### `find_expert_policy.py`

- Entraîne un agent expert sur `CartPole-v1`.
- Utilise `PPO`, qui est plus stable et donne en général une meilleure politique sur CartPole.
- Sauvegarde le meilleur modèle dans `Cartpole/experts/best_model`.
- Sauvegarde aussi le modèle final dans `Cartpole/experts/ppo_cartpole_final`.
- Écrit les logs d'évaluation dans `Cartpole/logs/`.

### `generate_trajectory.py`

- Charge l'expert entraîné.
- Génère 50 trajectoires en faisant rouler la politique experte.
- Sauvegarde le dataset expert dans `Cartpole/datasets/cartpole_expert_50.h5`.
- Ce fichier contient les observations, actions, récompenses et indicateurs de fin d'épisode.

### `visualize.py`

- Charge le même expert et le dataset généré.
- Permet trois modes:
  - `TERMINAL`: affiche des statistiques sur les trajectoires.
  - `PLOT`: trace la distribution des retours.
  - `VIDEO`: enregistre une vidéo de l'expert.

## Ce que cela couvre dans l'énoncé

Cette partie du projet répond à la phase de **trajectory generation**:

- on entraîne d'abord une politique experte $\pi_1$;
- on génère ensuite un ensemble de trajectoires expert;
- on obtient un dataset réutilisable pour les expériences d'imitation learning.

Pour la suite de l'énoncé, il faut ensuite:

- lancer les algorithmes d'imitation learning demandés sur ce dataset;
- comparer les performances pour plusieurs tailles de dataset expert;
- répéter sur au moins deux environnements;
- reporter les résultats sur au moins 3 seeds.

## Fichiers produits

- `Cartpole/experts/best_model`
- `Cartpole/experts/ppo_cartpole_final`
- `Cartpole/logs/`
- `Cartpole/datasets/cartpole_expert_50.h5`
- `Cartpole/plots/expert_trajectories_cartpole.png`
- `Cartpole/videos/`

## Imitation learning (Cartpole)

Le dossier contient désormais un wrapper local pour lancer des sweeps BC spécifiques à Cartpole, enregistrer des vidéos et produire des `results.csv` exploitables.

- `Cartpole/sweep_bc_cartpole.py` — lance les entraînements BC pour plusieurs tailles de dataset et seeds, écrit `results.csv` et peut enregistrer des vidéos par expérience.
- `Cartpole/generate_video_from_bc.py` — script utilitaire et fonction `record_videos()` utilisée par le wrapper pour enregistrer N épisodes à partir d'un checkpoint BC.
- Les sorties par expérience sont écrites sous `Cartpole/il/k{K}/seed{S}/` avec `bc_policy.pt`, `metrics.json` et (si demandé) un sous-dossier `videos/` contenant les MP4.

Exemple (PowerShell):

```powershell
conda activate imitlearn
# Sweep Cartpole and record 3 videos per experiment
python .\Cartpole\sweep_bc_cartpole.py --env-id CartPole-v1 --dataset .\Cartpole\datasets\cartpole_expert_50.h5 --output-root .\Cartpole\il --sizes 1,5,10,25,50 --seeds 0,1,2 --record-videos --n-episodes 3
```

Remarque: enregistrer des vidéos pour toutes les expériences peut générer beaucoup de fichiers. Le notebook (voir ci‑dessous) n'intègrera qu'un sous-ensemble représentatif (p.ex. meilleur seed par taille) pour la visualisation.

## Notebook

Une ébauche de notebook `Cartpole/notebooks/Cartpole_pipeline.ipynb` sera ajoutée. Il contiendra les étapes principales du pipeline (génération d'expert, création de dataset, sweeps BC, plots de performance et intégration de vidéos représentatives). Le notebook utilisera les scripts existants et l'arborescence des résultats pour charger et afficher les vidéos et graphiques.

Le pipeline d'imitation learning réutilisable est décrit dans [ImitationLearning/README.md](../ImitationLearning/README.md).

## Remarque

Si `best_model` n'existe pas encore, il faut exécuter `find_expert_policy.py` avant les deux autres scripts.
