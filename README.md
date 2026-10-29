# MultiRAG-Agent : Assistant Multi-Agents RAG Multilingue Sourcé

Prototype de recherche documentaire multi-agents capable d'interroger un corpus d'autorité multilingue (Français, Anglais, Arabe), avec contrôle contradictoire des hallucinations par inférence logique (NLI) et sourçage strict.

Projet réalisé dans le cadre du Master Ingénierie Informatique (Module Automatisation et IA Agentique) à l'Université Mohammed Premier d'Oujda.

## 🏗️ Architecture du Système (5 Agents)

1. **Agent 0 : Orchestrateur (LangGraph)** — Coordonne la machine d'états, supervise les timeouts et déclenche la remédiation en cas de rejet.
2. **Agent 1 : Détection de Langue (LanguageAgent)** — Identifie automatiquement la langue (`fr`, `en`, `ar`) et applique la politique d'abstention si langue non supportée.
3. **Agent 2 : RAG Multilingue (RetrieverAgent)** — Plongement sémantique dense avec `paraphrase-multilingual-mpnet-base-v2` et recherche cosinus dans ChromaDB.
4. **Agent 3 : Réponse & Sourçage (GeneratorAgent)** — Formulation de la synthèse conditionnée exclusivement sur les extraits numérotés (`<doc id='n'>`).
5. **Agent 4 : Vérificateur (VerifierAgent)** — Contrôle NLI strict (Natural Language Inference) pour valider que chaque affirmation est étayée par sa source.

## 📁 Structure du Projet

```
ProjetPFE/
├── data/
│   ├── corpus/            # Notices documentaires structurées (FR, EN, AR)
│   └── eval/              # Jeu de 30 questions de test avec ground truth
├── src/
│   ├── agents/            # Agents spécialisés (Language, RAG, Response, Verifier)
│   ├── rag/               # Modèles de données Pydantic, chunking, ChromaDB
│   └── utils/             # Logging JSONL, helpers textuels
├── tests/                 # Tests unitaires automatisés
├── logs/                  # Traces d'exécution (traces.jsonl)
├── requirements.txt       # Dépendances fixées
└── README.md
```

## 🚀 Installation & Démarrage

```bash
pip install -r requirements.txt
python tests/test_language_agent.py
```
