import os
from typing import List, Dict, Any, Optional
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.rag.models import DocumentChunk

# Charger les variables d'environnement (.env)
load_dotenv()


# ──────────────────────────────────────────────
# Modèle Pydantic pour le résultat de la réponse
# ──────────────────────────────────────────────

class ResponseResult(BaseModel):
    """
    Résultat produit par l'Agent 3 (ResponseAgent).
    Contient la réponse générée strictement sourcée, les citations et les métadonnées.
    """
    query: str = Field(..., description="La question originale de l'utilisateur")
    answer: str = Field(..., description="La réponse textuelle avec citations formelles [1], [2]...")
    lang: str = Field(..., description="Langue de la réponse (fr, en, ar)")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Liste des sources citées avec identifiant, titre et URL"
    )
    has_answer: bool = Field(
        ...,
        description="True si une réponse factuelle a pu être rédigée, False en cas de refus motivé"
    )
    model_used: str = Field(
        default="gemini-3.5-flash-lite",
        description="Nom du modèle LLM utilisé pour l'inférence"
    )


# ──────────────────────────────────────────────
# Agent 3 : Response Generator
# ──────────────────────────────────────────────

class ResponseAgent:
    """
    Agent 3 : Génération de réponses sourcées (Fact-grounded Response Generation).

    Rôle :
    - Reçoit les passages pertinents (chunks) transmis par Agent 2 (RetrieverAgent).
    - Construit un prompt strict avec isolation de contexte (<context>...</context>)
      pour immuniser le LLM contre le prompt injection et les hallucinations.
    - Interroge le LLM (Gemini 3.5 Flash Lite) avec obligation de citations formelles [n].
    - Formate le retour sous forme de ResponseResult certifié.
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise le ResponseAgent avec la clé d'API Gemini.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Clé d'API introuvable. Veuillez définir GEMINI_API_KEY dans votre fichier .env"
            )
        self.model = model or self.DEFAULT_MODEL
        self.api_url = self.API_URL_TEMPLATE.format(model=self.model, api_key=self.api_key)

    def _build_prompt(self, query: str, chunks: List[DocumentChunk], lang: str) -> str:
        """
        Construit le prompt sous forme de conteneurs stricts anti-hallucination.
        """
        # Formater les extraits documentaires
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"<source id='{i}' titre='{chunk.titre}' langue='{chunk.langue}'>\n"
                f"{chunk.texte}\n"
                f"</source>"
            )
        context_str = "\n\n".join(context_blocks)

        # Instructions selon la langue demandée
        lang_instruction = {
            "fr": "Rédige ta réponse en FRANÇAIS.",
            "en": "Write your answer in ENGLISH.",
            "ar": "اكتب إجابتك باللغة العربية الفصحى."
        }.get(lang, "Rédige ta réponse en FRANÇAIS.")

        system_prompt = f"""Tu es un agent d'extraction factuelle hautement rigoureux dans un système RAG multilingue.
Ta mission est de répondre à la question de l'utilisateur en te basant STRICTEMENT et UNIQUEMENT sur les extraits fournis ci-dessous.

RÈGLES IMPÉRATIVES DE SÉCURITÉ ET FIDÉLITÉ :
1. RÈGLE D'OR : N'utilise AUCUNE connaissance externe. Si une information n'est pas explicitement écrite dans les extraits, tu ne dois PAS la deviner ni l'inventer.
2. CITATIONS : Chaque fait ou affirmation que tu avances DOIT être suivi de son numéro de source entre crochets, ex: [1], [2].
3. LANGUE : {lang_instruction}
4. ABSENCE D'INFO : Si les extraits ne permettent pas de répondre précisément à la question, réponds exactement par : "Information non trouvée dans le corpus documentaire fourni."

<context>
{context_str}
</context>

Question de l'utilisateur : {query}

Réponse factuelle et sourcée :"""

        return system_prompt

    def run(self, query: str, chunks: List[DocumentChunk], lang: str = "fr") -> ResponseResult:
        """
        Point d'entrée principal de l'agent 3.

        Args:
            query: Question de l'utilisateur
            chunks: Liste des DocumentChunk retenus par l'Agent 2
            lang: Langue cible de la réponse (fr, en, ar)

        Returns:
            ResponseResult contenant la réponse sourcée et la liste des sources
        """
        # Garde-fou : si aucun passage n'a été fourni par le Retriever
        if not chunks:
            refusal_messages = {
                "fr": "Aucune source documentaire disponible pour répondre à cette question.",
                "en": "No source documents available to answer this question.",
                "ar": "لا توجد مصادر وثائقية متاحة للإجابة على هذا السؤال."
            }
            return ResponseResult(
                query=query,
                answer=refusal_messages.get(lang, refusal_messages["fr"]),
                lang=lang,
                citations=[],
                has_answer=False,
                model_used=self.model
            )

        # 1. Construction du prompt strict
        prompt = self._build_prompt(query, chunks, lang)

        # 2. Appel à l'API Gemini
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Température basse pour maximiser le déterminisme et la fidélité
                "topP": 0.95,
                "maxOutputTokens": 1024
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Erreur API Gemini ({response.status_code}) : {response.text}"
                )

            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            return ResponseResult(
                query=query,
                answer=f"Erreur technique lors de la génération : {str(e)}",
                lang=lang,
                citations=[],
                has_answer=False,
                model_used=self.model
            )

        # 3. Assembler les citations associées
        citations = []
        for i, chunk in enumerate(chunks, 1):
            # Vérifier si la source [i] est mentionnée ou retenue
            citations.append({
                "source_num": i,
                "chunk_id": chunk.chunk_id,
                "notice_id": chunk.notice_id,
                "titre": chunk.titre,
                "langue": chunk.langue,
                "source_url": chunk.source_url
            })

        # Détecter si la réponse est un refus formulé
        is_refusal = (
            "non trouvée dans le corpus" in raw_text.lower() or
            "not found in" in raw_text.lower() or
            "غير موجودة" in raw_text
        )

        return ResponseResult(
            query=query,
            answer=raw_text,
            lang=lang,
            citations=citations if not is_refusal else [],
            has_answer=not is_refusal,
            model_used=self.model
        )
