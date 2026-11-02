import os
import sys
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import MultiAgentOrchestrator, PipelineResult
from src.rag.vector_store import VectorStoreManager


# ──────────────────────────────────────────────
# Modèles de Données Pydantic pour l'API REST
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    """Schéma de la requête utilisateur envoyée à l'API."""
    query: str = Field(
        ...,
        description="Question posée par l'utilisateur en Français, Anglais ou Arabe",
        examples=["Qui était le roi Masinissa et quel royaume a-t-il unifié ?"]
    )


class CitationResponse(BaseModel):
    """Détail d'une source documentaire formelle citée."""
    source_num: Optional[int] = Field(None, description="Numéro de citation dans la réponse [n]")
    notice_id: str = Field(..., description="Identifiant unique de la notice d'origine")
    chunk_id: str = Field(..., description="Identifiant du passage vectorisé")
    titre: str = Field(..., description="Titre du document source")
    langue: str = Field(..., description="Langue de la source (fr, en, ar)")
    source_url: str = Field(..., description="Lien URL de vérification externe")


class StepTraceResponse(BaseModel):
    """Trace d'audit d'un agent de la chaîne."""
    agent: str
    action: str
    status: str
    duration_ms: float
    details: Dict[str, Any]


class AskResponse(BaseModel):
    """Schéma complet de la réponse certifiée retournée par le système."""
    query: str = Field(..., description="Question originale soumise")
    lang: str = Field(..., description="Langue détectée par Agent 1 (fr, en, ar)")
    status: str = Field(
        ...,
        description="Statut global : SUCCESS, AMBIGUOUS_SUGGESTION, REFUSAL_NO_CORPUS, REFUSAL_LANGUAGE, REFUSAL_VERIFICATION"
    )
    final_answer: str = Field(..., description="Réponse certifiée sourcée ou message d'orientation")
    fidelity_score: float = Field(..., description="Taux de fidélité factuelle certifié par Agent 4 (0.0 à 1.0)")
    citations: List[CitationResponse] = Field(default_factory=list, description="Sources documentaires authentiques utilisées")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions de désambiguïsation en cas de requête vague")
    total_latency_ms: float = Field(..., description="Latence totale de traitement en millisecondes")
    trace: List[StepTraceResponse] = Field(default_factory=list, description="Traçabilité détaillée de chaque agent")


class HealthResponse(BaseModel):
    """État de santé du système et des composants."""
    status: str
    system_name: str
    author: str
    indexed_documents_count: int
    vector_store: str
    llm_model: str


# ──────────────────────────────────────────────
# Initialisation de l'Application FastAPI
# ──────────────────────────────────────────────

app = FastAPI(
    title="MultiRAG-Agent API",
    description="""
### API REST du Système RAG Multi-Agents Trilingue (FR / EN / AR)
**Auteur : Amine GHAYOUT** — Projet de Master 2 & Candidature Recherche (Cnam / Efrei Paris)

Cette API expose l'architecture complète des **5 agents autonomes** :
1. **Agent 1 (LanguageAgent)** : Détection et validation de la langue (ISO-639-1).
2. **Agent 2 (RetrieverAgent)** : Recherche sémantique dense dans ChromaDB (MPNet 768d) avec seuil strict (0.40) et seuil de désambiguïsation (0.25).
3. **Agent 3 (ResponseAgent)** : Synthèse factuelle sourcée avec isolation de contexte (Gemini).
4. **Agent 4 (VerifierAgent)** : Audit NLI (Natural Language Inference) éradiquant les hallucinations.
5. **Agent 0 (MultiAgentOrchestrator)** : Coordination cyclique et traçabilité d'audit.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre la connexion depuis n'importe quelle interface (React, Vue, mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation paresseuse de l'Orchestrateur (Singleton)
_orchestrator: Optional[MultiAgentOrchestrator] = None

def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(
            db_path="data/chroma_db",
            collection_name="multirag_corpus",
            similarity_threshold=0.40,
            top_k=3
        )
    return _orchestrator


# ──────────────────────────────────────────────
# Routes de l'API
# ──────────────────────────────────────────────

@app.get("/", tags=["Général"])
def root():
    """Point d'entrée racine : informations sur le projet et liens de documentation."""
    return {
        "projet": "MultiRAG-Agent",
        "auteur": "Amine GHAYOUT",
        "master": "Automatisation et IA Agentique (M2I)",
        "destination": "Préparation Stage de Recherche Cnam / Efrei (Paris)",
        "documentation_interactive_swagger": "/docs",
        "statut": "Opérationnel",
        "langues_supportees": ["fr", "en", "ar"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Supervision"])
def health_check():
    """Vérifie l'état de santé opérationnel de la base ChromaDB et du pipeline d'agents."""
    try:
        orch = get_orchestrator()
        count = orch.retriever_agent.vector_store.count()
        return HealthResponse(
            status="healthy",
            system_name="MultiRAG-Agent Pipeline",
            author="Amine GHAYOUT",
            indexed_documents_count=count,
            vector_store="ChromaDB (Local Persistent)",
            llm_model="Gemini 3.5 Flash Lite"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'initialisation du système : {str(e)}"
        )


@app.post("/ask", response_model=AskResponse, tags=["Multi-Agents RAG"])
def ask_question(payload: AskRequest):
    """
    Exécute le pipeline complet des 5 agents sur la question soumise :
    - Détecte la langue (FR, EN, AR)
    - Effectue la recherche sémantique dans ChromaDB
    - Génère la réponse sourcée [n]
    - Vérifie la fidélité logique (NLI) anti-hallucination
    """
    query_clean = payload.query.strip()
    if not query_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La question ne peut pas être vide."
        )

    try:
        orch = get_orchestrator()
        result: PipelineResult = orch.process(query_clean)

        # Formater les citations
        citations_resp = []
        for c in result.citations:
            citations_resp.append(CitationResponse(
                source_num=c.get("source_num"),
                notice_id=c.get("notice_id", ""),
                chunk_id=c.get("chunk_id", ""),
                titre=c.get("titre", ""),
                langue=c.get("langue", "fr"),
                source_url=c.get("source_url", "")
            ))

        # Formater la trace d'audit
        trace_resp = []
        for t in result.trace:
            trace_resp.append(StepTraceResponse(
                agent=t.agent,
                action=t.action,
                status=t.status,
                duration_ms=t.duration_ms,
                details=t.details
            ))

        return AskResponse(
            query=result.query,
            lang=result.lang,
            status=result.status,
            final_answer=result.final_answer,
            fidelity_score=result.fidelity_score,
            citations=citations_resp,
            suggestions=result.suggestions,
            total_latency_ms=result.total_latency_ms,
            trace=trace_resp
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement de la requête : {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Démarrage de l'API MultiRAG-Agent sur http://localhost:8000 ...")
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
