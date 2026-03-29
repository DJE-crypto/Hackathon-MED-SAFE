# 🛡️ MED-SAFE : Audit de Robustesse Éthique des LLM Médicaux

**Projet de Hackathon - ECE Paris (B2)** *Évaluation automatisée de la sécurité des modèles de langage en contexte de santé via une stack d'observabilité ELK.*

## 📖 Présentation du Projet
Dans le cadre du Hackathon TreeTech 2026, nous avons conçu **MED-SAFE**, un framework d'audit pour tester la résistance des IA (Gemma, Llama, Mistral) face à des requêtes médicales malveillantes ou non éthiques. 

L'objectif est de quantifier l'alignement de ces modèles sur 150 scénarios critiques (fraude aux médicaments, violations de vie privée, erreurs de diagnostic intentionnelles) tout en surveillant les performances techniques (latence, longueur de réponse).

## 🛠️ Architecture Système
Le projet repose sur une pipeline d'ingestion de données automatisée :
1.  **Environnement :** Conteneurisation via Docker pour garantir la reproductibilité.
2.  **Moteur d'IA :** Modèles exécutés localement via l'API Ollama.
3.  **Automation :** Script Python (`send_to_elk.py`) pour le stress-test et le scoring automatique.
4.  **Observabilité :** Stack ELK (Logstash -> Elasticsearch -> Kibana) pour l'indexation et la visualisation en temps réel.

## 📁 Structure du Repository
- `data/` : Le benchmark de +150 prompts classés par catégories de risques.
- `scripts/` : Le script d'orchestration Python.
- `docker/` : Configuration `docker-compose.yml` pour la stack ELK.
- `results/` : Exportations CSV des tests bruts et scores.

## 🚀 Installation et Utilisation

### 1. Prérequis
- Docker & Docker Compose
- Python 3.9+
- Ollama (avec les modèles gemma2, llama3.2 et phi3 installés)

### 2. Lancement de la Stack d'Observabilité
```bash
docker-compose up -d
