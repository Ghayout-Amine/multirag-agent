import os
import json
import re
from typing import List, Dict, Any, Optional
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.rag.models import DocumentChunk

# Charger les variables d'environnement (.env)
load_dotenv()


# ──────────────────────────────────────────────
# Modèle Pydantic pour le résultat de la vérification
# ──────────────────────────────────────────────

class VerificationClaim(BaseModel):
    """Détail d'une assertion vérifiée par rapport aux sources."""
    assertion: str = Field(..., description="L'affirmation extraite de la réponse")
    source_citee: Optional[str] = Field(None, description="Référence de la source citée (ex: [1])")
    is_supported: bool = Field(..., description="True si le passage source soutient formellement l'affirmation")
    explication: str = Field(..., description="Justification du verdict de vérification")


class VerifierResult(BaseModel):
    """
    Résultat produit par l'Agent 4 (VerifierAgent).
    Garantit l'éradication des hallucinations et calcule le taux de fidélité factuelle.
    """
    is_faithful: bool = Field(..., description="True si 100% des faits sont validés par les sources")
    fidelity_score: float = Field(..., description="Pourcentage d'affirmations validées (0.0 à 1.0)")
    claims_verified: List[VerificationClaim] = Field(default_factory=list, description="Liste des assertions analysées")
    hallucinations_detected: List[str] = Field(default_factory=list, description="Liste des affirmations non prouvées")
    action_recommended: str = Field(..., description="'ACCEPT' si fidèle, 'RETRY' pour révision, ou 'REFUSE' si infondé")
    feedback: str = Field(..., description="Explication globale pour l'orchestrateur")


# ──────────────────────────────────────────────
# Agent 4 : Verifier Agent (NLI / Anti-hallucination)
# ──────────────────────────────────────────────

class VerifierAgent:
    """
    Agent 4 : Contrôle de fidélité factuelle et élimination des hallucinations (NLI).

    Rôle :
    - Reçoit la réponse candidate formulée par Agent 3 (ResponseAgent).
    - Compare chaque affirmation avec le texte brut des passages sources (chunks).
    - Exécute un test d'inférence en langage naturel (Natural Language Inference).
    - Décide de valider la réponse ('ACCEPT'), demander une régénération ('RETRY'),
      ou forcer un refus sécurisé ('REFUSE').
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Clé GEMINI_API_KEY introuvable dans .env")
        self.model = model or self.DEFAULT_MODEL
        self.api_url = self.API_URL_TEMPLATE.format(model=self.model, api_key=self.api_key)

    def _build_verification_prompt(self, response_text: str, chunks: List[DocumentChunk]) -> str:
        """
        Construit un prompt spécialisé exigeant un JSON strict d'évaluation NLI.
        """
        # Assemblage des sources
        sources_text = ""
        for i, c in enumerate(chunks, 1):
            sources_text += f"[Source {i}] ({c.titre}) :\n{c.texte}\n\n"

        prompt = f"""Tu es un auditeur de sécurité factuelle (NLI - Natural Language Inference).
Ta mission est de vérifier si chaque affirmation contenue dans la réponse candidate est STRICTEMENT et DIRECTEMENT prouvée par les sources documentaires fournies.

SOURCES DISPONIBLES :
{sources_text}

RÉPONSE CANDIDATE À AUDITER :
"{response_text}"

INSTRUCTIONS D'ÉVALUATION :
1. Découpe la réponse en assertions factuelles atomiques.
2. Pour chaque assertion, vérifie si les sources ci-dessus contiennent explicitement la preuve.
3. Si une information est ajoutée, exagérée ou absente des sources, marque is_supported = false.
4. Réponds UNIQUEMENT sous la forme d'un objet JSON strict avec ce format :
{{
  "claims": [
    {{
      "assertion": "texte de l'affirmation",
      "source_citee": "[1]",
      "is_supported": true,
      "explication": "preuve trouvée dans Source 1"
    }}
  ],
  "global_action": "ACCEPT ou RETRY ou REFUSE",
  "summary": "résumé en une phrase"
}}"""

        return prompt

    def run(self, query: str, response_text: str, chunks: List[DocumentChunk]) -> VerifierResult:
        """
        Vérifie la fidélité de la réponse candidate.

        Args:
            query: Question originale
            response_text: Réponse candidate à auditer
            chunks: Passages sources authentiques

        Returns:
            VerifierResult avec verdict, score et recommandations
        """
        clean_response = response_text.strip()

        # ── Cas 1 : Réponse vide ou refus explicite ──────────────────
        is_declared_refusal = (
            not clean_response or
            "non trouvée dans le corpus" in clean_response.lower() or
            "non trouvee dans le corpus" in clean_response.lower() or
            "aucune source documentaire" in clean_response.lower() or
            "not found in" in clean_response.lower()
        )

        if is_declared_refusal:
            # Un refus explicite est parfaitement conforme : aucune affirmation douteuse
            return VerifierResult(
                is_faithful=True,
                fidelity_score=1.0,
                claims_verified=[],
                hallucinations_detected=[],
                action_recommended="ACCEPT",
                feedback="Refus légitime certifié : aucune assertion non prouvée formulée."
            )

        # ── Cas 2 : Absence totale de chunks sources mais réponse rédigée ──
        if not chunks:
            return VerifierResult(
                is_faithful=False,
                fidelity_score=0.0,
                claims_verified=[],
                hallucinations_detected=[clean_response],
                action_recommended="REFUSE",
                feedback="Alerte de sécurité : Réponse formulée sans aucun document source fourni."
            )

        # ── Cas 3 : Audit sémantique NLI via LLM ─────────────────────
        prompt = self._build_verification_prompt(clean_response, chunks)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,  # Déterminisme absolu pour l'audit
                "responseMimeType": "application/json"
            }
        }

        try:
            resp = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if resp.status_code != 200:
                raise RuntimeError(f"Erreur API Verifier ({resp.status_code}) : {resp.text}")

            data = resp.json()
            raw_json_str = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            parsed = json.loads(raw_json_str)

            claims_data = parsed.get("claims", [])
            claims_verified: List[VerificationClaim] = []
            hallucinations: List[str] = []

            for c in claims_data:
                is_supp = bool(c.get("is_supported", False))
                claim_obj = VerificationClaim(
                    assertion=c.get("assertion", ""),
                    source_citee=c.get("source_citee"),
                    is_supported=is_supp,
                    explication=c.get("explication", "")
                )
                claims_verified.append(claim_obj)
                if not is_supp:
                    hallucinations.append(claim_obj.assertion)

            # Calcul du score de fidélité
            total_claims = len(claims_verified)
            if total_claims > 0:
                supported_count = sum(1 for c in claims_verified if c.is_supported)
                fidelity_score = round(supported_count / total_claims, 4)
            else:
                fidelity_score = 1.0

            is_faithful = (len(hallucinations) == 0)

            # Détermination de l'action recommandée
            if is_faithful:
                action = "ACCEPT"
                feedback = f"Réponse 100% fidèle : les {total_claims} assertion(s) sont soutenues par les sources."
            elif fidelity_score >= 0.5:
                action = "RETRY"
                feedback = f"Attention : {len(hallucinations)} hallucination(s) détectée(s). Révision recommandée."
            else:
                action = "REFUSE"
                feedback = f"Rejet critique : fidélité insuffisante ({fidelity_score * 100}%). Réponse non sourcée."

            return VerifierResult(
                is_faithful=is_faithful,
                fidelity_score=fidelity_score,
                claims_verified=claims_verified,
                hallucinations_detected=hallucinations,
                action_recommended=action,
                feedback=feedback
            )

        except Exception as e:
            # Fallback en cas d'erreur de parsing ou réseau
            return VerifierResult(
                is_faithful=False,
                fidelity_score=0.0,
                claims_verified=[],
                hallucinations_detected=["Erreur technique durant la vérification"],
                action_recommended="RETRY",
                feedback=f"Erreur d'audit : {str(e)}"
            )
