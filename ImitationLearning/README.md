# Imitation learning

Ce dossier contient une base réutilisable pour l'imitation learning à partir des datasets experts générés dans `Cartpole/` et `Pendulum/`.

## Ce que fait le script

Le fichier `bc.py` implémente un baseline de **behavior cloning**:

- il charge un dataset expert HDF5;
- il entraîne une politique supervisée sur les paires observation-action;
- il s'adapte automatiquement à un espace d'actions discret ou continu;
- il sauvegarde un checkpoint et un fichier de métriques;
- il évalue la politique sur l'environnement cible.

Cette brique sert de point de départ pour la suite du projet et permet déjà de comparer des performances selon:

- la taille du dataset expert;
- la seed d'entraînement;
- l'environnement utilisé.

Le fichier `sweep_bc.py` automatise ces comparaisons et écrit un `results.csv` par sweep.

## Dépendances attendues

Le script utilise:

- `gymnasium`
- `torch`
- `h5py`
- `numpy`
- `stable_baselines3` pour la génération des experts

## Exemple CartPole

```powershell
conda activate imitlearn
python .\ImitationLearning\bc.py train --env-id CartPole-v1 --dataset .\Cartpole\datasets\cartpole_expert_50.h5 --output-dir .\Cartpole\il\bc_seed0 --max-trajectories 50 --seed 0 --eval-episodes 20
python .\ImitationLearning\bc.py eval --checkpoint .\Cartpole\il\bc_seed0\bc_policy.pt --episodes 20
```

## Exemple Pendulum

```powershell
conda activate imitlearn
python .\ImitationLearning\bc.py train --env-id Pendulum-v1 --dataset .\Pendulum\datasets\pendulum_expert_50.h5 --output-dir .\Pendulum\il\bc_seed0 --max-trajectories 50 --seed 0 --eval-episodes 20
python .\ImitationLearning\bc.py eval --checkpoint .\Pendulum\il\bc_seed0\bc_policy.pt --episodes 20
```

## Sweep multi-seeds

```powershell
conda activate imitlearn
python .\ImitationLearning\sweep_bc.py --env-id CartPole-v1 --dataset .\Cartpole\datasets\cartpole_expert_50.h5 --output-root .\Cartpole\il\bc_sweep --sizes 1,5,10,25,50 --seeds 0,1,2 --eval-episodes 20
```

## Sweeps demandés par l'énoncé

Les résultats peuvent être obtenus en répétant le training pour plusieurs tailles de dataset et plusieurs seeds. Par exemple:

- tailles: 1, 5, 10, 25, 50 trajectoires;
- seeds: 0, 1, 2.

Les métriques sont écrites dans `metrics.json` dans chaque dossier d'expérience.

## Remarque
Ce baseline ne couvre pas encore les algorithmes IR plus avancés cités dans l'énoncé. Il fournit cependant une base propre pour brancher les implémentations suivantes sur les mêmes datasets expert et la même évaluation.

Pour les instructions spécifiques à `CartPole` (enregistrement de vidéos, wrapper local de sweep et notebook), consultez `Cartpole/README.md`.
