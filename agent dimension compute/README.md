# Agent Dimension Compute

Agent LLM itératif qui résout le problème du kissing number en dimension 3.

## Description

Cet agent utilise Claude (Anthropic) pour générer itérativement des algorithmes d'optimisation qui trouvent des configurations optimales de sphères satisfaisant la contrainte du kissing number.

**Contrainte mathématique** : Pour un ensemble C ⊂ ℝ³ de points :
- 0 ∉ C (l'origine n'est pas dans C)
- min{||x-y|| : x≠y ∈ C} ≥ max{||x|| : x ∈ C}

**Objectif** : Trouver 12 sphères en dimension 3 (kissing number connu).

## Installation

```bash
# Installer les dépendances
pip install anthropic numpy python-dotenv scipy

# Ou utiliser le requirements.txt si créé
pip install -r requirements.txt
```

## Configuration

Le fichier `.env` contient votre clé API Anthropic :
```
ANTHROPIC_API_KEY=votre_clé_api_ici
```

⚠️ **Important** : Ne jamais commiter le fichier `.env` (déjà dans `.gitignore`)

## Utilisation

```bash
python kissing_number_agent.py
```

L'agent va :
1. Générer un premier algorithme d'optimisation
2. Exécuter le code généré
3. Valider la configuration selon la contrainte mathématique
4. Fournir un feedback détaillé
5. Itérer jusqu'à trouver une solution ou atteindre le maximum d'itérations (15 par défaut)

## Paramètres configurables

Dans la fonction `main()` :
- `dimension` : Dimension du problème (default: 3)
- `max_iterations` : Nombre maximum d'itérations (default: 15)
- `target_spheres` : Nombre cible de sphères (default: 12)

## Résultats

Les résultats sont sauvegardés dans `kissing_number_results.json` avec :
- Historique de toutes les tentatives
- Code généré à chaque itération
- Métriques de validation
- Feedback fourni à l'agent

## Architecture

```
KissingNumberAgent
├── validate_configuration() : Vérifie la contrainte mathématique
├── execute_code() : Exécute le code généré par l'agent
├── generate_feedback() : Crée le feedback pour guider l'agent
├── get_system_prompt() : Construit le prompt système
├── run() : Boucle d'optimisation itérative
└── save_results() : Sauvegarde JSON des résultats
```

## Approches suggérées à l'agent

- Placement sur structures géométriques régulières (polyèdres)
- Optimisation par forces (répulsion/attraction)
- Simulated annealing
- Algorithmes évolutionnaires
- Optimisation sous contraintes avec scipy.optimize

## Sécurité

Le code généré est exécuté dans un environnement contrôlé avec :
- Liste blanche de bibliothèques autorisées (numpy, scipy)
- Validation des types et formes des résultats
- Gestion des erreurs d'exécution

## License

MIT
