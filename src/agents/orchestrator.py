import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.agents.language_agent import LanguageAgent, LanguageResult
from src.agents.retriever_agent import RetrieverAgent, RetrieverResult
from src.agents.response_agent import ResponseAgent, ResponseResult
from src.agents.verifier_agent import VerifierAgent, VerifierResult
from src.rag.models import DocumentChunk


# ──────────────────────────────────────────────
# Modèle de résultat global retourné par l'Orchestrateur
# ──────────────────────────────────────────────

class ExecutionStepTrace(BaseModel):
    """Enregistrement d'une étape du pipeline pour l'audit et l'évaluation."""
    agent: str = Field(..., description="Nom de l'agent intervenant")
    action: str = Field(..., description="Action exécutée")
    status: str = Field(..., description="'SUCCESS', 'WARNING', 'REFUSAL', 'ERROR'")
    duration_ms: float = Field(..., description="Durée de l'étape en millisecondes")
    details: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées spécifiques de l'étape")


class PipelineResult(BaseModel):
    """
    Résultat complet et transparent délivré à l'utilisateur final ou à l'API.
    """
    query: str = Field(..., description="Question originale soumise par l'utilisateur")
    final_answer: str = Field(..., description="Réponse finale certifiée et sourcée")
    lang: str = Field(..., description="Langue de traitement (fr, en, ar)")
    status: str = Field(
        ...,
        description="'SUCCESS', 'REFUSAL_LANGUAGE', 'REFUSAL_NO_CORPUS', 'REFUSAL_VERIFICATION'"
    )
    fidelity_score: float = Field(default=1.0, description="Score de fidélité certifié par l'Agent 4 (0 à 1)")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Liste des sources documentaires formelles")
    chunks_used: List[DocumentChunk] = Field(default_factory=list, description="Extraits documentaires retenus")
    total_latency_ms: float = Field(..., description="Latence totale de traitement de bout en bout")
    trace: List[ExecutionStepTrace] = Field(default_factory=list, description="Traçabilité complète des agents")


# ──────────────────────────────────────────────
# Agent 0 : Orchestrateur Multi-Agents
# ──────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Agent 0 : Chef d'orchestre du système MultiRAG-Agent.

    Coordonne la séquence d'exécution des 4 agents spécialisés :
      1. LanguageAgent  -> Détection et filtrage de la langue (FR, EN, AR)
      2. RetrieverAgent -> Extraction sémantique de passages (ChromaDB + MPNet)
      3. ResponseAgent  -> Génération factuelle sourcée avec isolation de contexte (Gemini)
      4. VerifierAgent  -> Audit NLI anti-hallucination (Validation, Retry ou Refus)
    """

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        collection_name: str = "multirag_corpus",
        similarity_threshold: float = 0.40,
        top_k: int = 3
    ):
        """Initialise tous les agents de la chaîne de traitement."""
        self.language_agent = LanguageAgent()
        self.retriever_agent = RetrieverAgent(
            db_path=db_path,
            collection_name=collection_name,
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )
        self.response_agent = ResponseAgent()
        self.verifier_agent = VerifierAgent()

    def process(self, query: str) -> PipelineResult:
        """
        Traite une requête de bout en bout à travers le graphe d'agents.
        """
        start_time = time.time()
        trace: List[ExecutionStepTrace] = []

        # ──────────────────────────────────────────────
        # Étape 1 : Agent 1 — Analyse Linguistique
        # ──────────────────────────────────────────────
        t0 = time.time()
        lang_res: LanguageResult = self.language_agent.run(query)
        dt = round((time.time() - t0) * 1000, 2)

        trace.append(ExecutionStepTrace(
            agent="Agent 1 (LanguageAgent)",
            action="Detection et validation linguistique",
            status="SUCCESS" if lang_res.is_supported else "REFUSAL",
            duration_ms=dt,
            details={"lang": lang_res.lang, "is_supported": lang_res.is_supported}
        ))

        if not lang_res.is_supported:
            total_dt = round((time.time() - start_time) * 1000, 2)
            return PipelineResult(
                query=query,
                final_answer=lang_res.message or "Langue non supportée.",
                lang=lang_res.lang,
                status="REFUSAL_LANGUAGE",
                fidelity_score=1.0,
                citations=[],
                chunks_used=[],
                total_latency_ms=total_dt,
                trace=trace
            )

        lang = lang_res.lang

        # ──────────────────────────────────────────────
        # Étape 2 : Agent 2 — Recherche Sémantique (RAG)
        # ──────────────────────────────────────────────
        t0 = time.time()
        retriever_res: RetrieverResult = self.retriever_agent.run(query, lang=lang)
        dt = round((time.time() - t0) * 1000, 2)

        trace.append(ExecutionStepTrace(
            agent="Agent 2 (RetrieverAgent)",
            action="Recherche semantique vectorielle dans ChromaDB",
            status="SUCCESS" if retriever_res.has_results else "REFUSAL",
            duration_ms=dt,
            details={"chunks_count": len(retriever_res.chunks), "message": retriever_res.message}
        ))

        if not retriever_res.has_results:
            total_dt = round((time.time() - start_time) * 1000, 2)
            return PipelineResult(
                query=query,
                final_answer=retriever_res.message,
                lang=lang,
                status="REFUSAL_NO_CORPUS",
                fidelity_score=1.0,
                citations=[],
                chunks_used=[],
                total_latency_ms=total_dt,
                trace=trace
            )

        chunks = retriever_res.chunks

        # ──────────────────────────────────────────────
        # Étape 3 : Agent 3 — Génération Factuelle Sourcée
        # ──────────────────────────────────────────────
        t0 = time.time()
        response_res: ResponseResult = self.response_agent.run(query, chunks=chunks, lang=lang)
        dt = round((time.time() - t0) * 1000, 2)

        trace.append(ExecutionStepTrace(
            agent="Agent 3 (ResponseAgent)",
            action="Generation de reponse factuelle avec citations formelles",
            status="SUCCESS" if response_res.has_answer else "WARNING",
            duration_ms=dt,
            details={"model": response_res.model_used, "has_answer": response_res.has_answer}
        ))

        if not response_res.has_answer:
            total_dt = round((time.time() - start_time) * 1000, 2)
            return PipelineResult(
                query=query,
                final_answer=response_res.answer,
                lang=lang,
                status="REFUSAL_NO_CORPUS",
                fidelity_score=1.0,
                citations=[],
                chunks_used=chunks,
                total_latency_ms=total_dt,
                trace=trace
            )

        candidate_answer = response_res.answer

        # ──────────────────────────────────────────────
        # Étape 4 : Agent 4 — Vérification NLI Anti-Hallucination
        # ──────────────────────────────────────────────
        t0 = time.time()
        verifier_res: VerifierResult = self.verifier_agent.run(
            query=query,
            response_text=candidate_answer,
            chunks=chunks
        )
        dt = round((time.time() - t0) * 1000, 2)

        trace.append(ExecutionStepTrace(
            agent="Agent 4 (VerifierAgent)",
            action="Audit NLI et calcul de la fidelite factuelle",
            status="SUCCESS" if verifier_res.is_faithful else "WARNING",
            duration_ms=dt,
            details={
                "fidelity_score": verifier_res.fidelity_score,
                "action": verifier_res.action_recommended,
                "hallucinations_count": len(verifier_res.hallucinations_detected)
            }
        ))

        # Décision de sécurité selon le verdict de l'Agent 4
        if verifier_res.action_recommended == "ACCEPT":
            final_answer = candidate_answer
            status = "SUCCESS"
        elif verifier_res.action_recommended == "RETRY":
            # Tentative de révision avec injection du feedback d'audit
            t0_retry = time.time()
            prompt_revision = (
                f"{query}\n\n[INSTRUCTION DU VERIFICATEUR : Supprime absolument ces affirmations non prouvees : "
                f"{', '.join(verifier_res.hallucinations_detected)}]"
            )
            retry_res = self.response_agent.run(prompt_revision, chunks=chunks, lang=lang)
            # Ré-audit
            re_verifier = self.verifier_agent.run(query, retry_res.answer, chunks=chunks)
            dt_retry = round((time.time() - t0_retry) * 1000, 2)

            trace.append(ExecutionStepTrace(
                agent="Agent 4 (VerifierAgent - Retry)",
                action="Re-audit apres revision",
                status="SUCCESS" if re_verifier.is_faithful else "REFUSAL",
                duration_ms=dt_retry,
                details={"new_fidelity": re_verifier.fidelity_score, "action": re_verifier.action_recommended}
            ))

            if re_verifier.is_faithful:
                final_answer = retry_res.answer
                verifier_res = re_verifier
                status = "SUCCESS"
            else:
                final_answer = "Information non validée par les sources documentaires (refus de sécurité anti-hallucination)."
                status = "REFUSAL_VERIFICATION"
        else:
            # Rejet critique immédiat (REFUSE)
            final_answer = "Information non validée par les sources documentaires (refus de sécurité anti-hallucination)."
            status = "REFUSAL_VERIFICATION"

        total_latency = round((time.time() - start_time) * 1000, 2)

        return PipelineResult(
            query=query,
            final_answer=final_answer,
            lang=lang,
            status=status,
            fidelity_score=verifier_res.fidelity_score,
            citations=response_res.citations if status == "SUCCESS" else [],
            chunks_used=chunks,
            total_latency_ms=total_latency,
            trace=trace
        )
