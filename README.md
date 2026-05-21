# Reinforcement Learning Project

Ce dépôt contient les scripts pour générer des experts, construire des datasets de trajectoires et lancer une base d'imitation learning pour Cartpole et Pendulum.

## Démarrage rapide

```powershell
conda activate imitlearn
python .\Cartpole\find_expert_policy.py
python .\Cartpole\generate_trajectory.py
python .\Cartpole\visualize.py
```

## Documentation utile

- [Cartpole/README.md](Cartpole/README.md)
- [ImitationLearning/README.md](ImitationLearning/README.md)

## Artefacts générés

- experts entraînés dans `Cartpole/experts/` et `Pendulum/experts/`
- datasets experts dans `Cartpole/datasets/` et `Pendulum/datasets/`
- résultats d'imitation learning dans `Cartpole/il/` ou `Pendulum/il/`
