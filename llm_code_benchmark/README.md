# 🚀 LLM Code Benchmark: Python → C++/Rust Performance Comparison

## 📋 Description

Ce projet évalue et compare les performances de différents **Large Language Models (LLMs)** dans la génération de code optimisé. L'objectif est de convertir du code Python en C++ (et potentiellement Rust) en utilisant plusieurs modèles LLM, puis de compiler, exécuter et comparer les performances pour identifier quel LLM génère le code le plus performant.

## 🎯 Objectifs

- **Génération de code** : Utiliser différents LLMs pour convertir du code Python en C++
- **Benchmark de performance** : Compiler et exécuter chaque version générée
- **Comparaison** : Classer les LLMs selon la vitesse d'exécution du code généré
- **Évaluation** : Mesurer l'amélioration de performance par rapport au code Python original

## 🔧 Technologies Utilisées

- **Python** : Code source original et orchestration
- **C++** : Code généré par les LLMs
- **OpenRouter API** : Interface unifiée pour accéder à plusieurs LLMs
- **Gradio** : Interface utilisateur pour l'interaction
- **Jupyter Notebook** : Environnement de développement et d'expérimentation

## 🤖 LLMs Testés

Le projet compare les modèles suivants (classés selon leurs performances) :

1. **Gemini 3 Pro Preview** (`google/gemini-3-pro-preview`)
2. **GPT-5.2 Codex** (`openai/gpt-5.2-codex`)
3. **Claude Opus 4.6** (`anthropic/claude-opus-4.6`)
4. **Gemini 3 Flash Preview** (`google/gemini-3-flash-preview`)
5. **Kimi K2.5** (`moonshotai/kimi-k2.5`)
6. **GLM-5** (`z-ai/glm-5`)

## 📁 Structure du Projet

```
llm_code_benchmark/
├── python_c_rust.ipynb          # Notebook principal avec l'interface Gradio
├── _bench_python.py              # Code Python de référence pour le benchmark
├── _verify_cpp.cpp               # Code C++ de vérification
├── _verify_cpp_exe               # Exécutable compilé
├── main.cpp                      # Template C++ principal
├── system_info.py                # Script pour obtenir les infos système
├── requirements.txt              # Dépendances Python
├── generated_*.cpp               # Code C++ généré par chaque LLM
├── main_*                        # Exécutables compilés pour chaque modèle
└── __pycache__/                  # Cache Python
```

## 🚀 Installation

1. **Cloner le repository** (si applicable)

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Configurer les variables d'environnement** :
   - Créer un fichier `.env` à la racine du projet `LLM_projects`
   - Ajouter votre clé API OpenRouter :
   ```
   OPENROUTER_API_KEY=sk-or-votre-cle-ici
   ```

4. **Compiler les exécutables C++** (si nécessaire) :
```bash
g++ -O3 -o main main.cpp
```

## 💻 Utilisation

1. **Lancer le notebook Jupyter** :
```bash
jupyter notebook python_c_rust.ipynb
```

2. **Exécuter toutes les cellules** pour initialiser l'interface Gradio

3. **Utiliser l'interface Gradio** pour :
   - Sélectionner un modèle LLM
   - Générer du code C++ à partir du code Python
   - Compiler et exécuter automatiquement
   - Comparer les performances

## 📊 Résultats

Le projet génère des métriques de performance incluant :
- Temps d'exécution de chaque version C++
- Comparaison avec le code Python original
- Classement des LLMs par performance du code généré

## 🔍 Exemple de Code Testé

Le benchmark utilise un calcul intensif avec des boucles pour mesurer les performances :

```python
def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result
```

## 📝 Notes

- Les résultats peuvent varier selon le matériel utilisé
- Les performances dépendent de la qualité de la compilation C++
- Le projet utilise des optimisations de compilation (`-O3`) pour maximiser les performances

## 🤝 Contribution

Ce projet fait partie d'une série d'expérimentations sur l'utilisation des LLMs pour la génération et l'optimisation de code.

## 📄 Licence

Voir le fichier LICENSE à la racine du projet.
