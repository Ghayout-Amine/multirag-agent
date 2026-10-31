from pydantic import BaseModel, Field
from typing import List, Optional

from src.rag.models import DocumentChunk
from src.rag.vector_store import VectorStoreManager


# ──────────────────────────────────────────────
# Modèle de retour de l'Agent RAG Retriever
# ──────────────────────────────────────────────

class RetrieverResult(BaseModel):
    """
    Résultat retourné par le RetrieverAgent après recherche dans ChromaDB.
    Contient les passages pertinents (chunks) ou un message de refus motivé.
    """
    query: str = Field(..., description="La question originale de l'utilisateur")
    lang: str = Field(..., description="Langue de la question (fr, en, ar)")
    chunks: List[DocumentChunk] = Field(
        default_factory=list,
        description="Liste des passages pertinents récupérés du corpus"
    )
    has_results: bool = Field(
        ...,
        description="True si au moins un passage dépasse le seuil de similarité"
    )
    message: str = Field(
        ...,
        description="Message descriptif : succès avec nombre de chunks, ou refus motivé"
    )


# ──────────────────────────────────────────────
# Agent 2 : RAG Retriever
# ──────────────────────────────────────────────

class RetrieverAgent:
    """
    Agent 2 : Retrieval RAG (Recherche sémantique dans le corpus vectorisé).

    Rôle :
    - Reçoit la question et la langue depuis Agent 1 (LanguageAgent)
    - Interroge la base vectorielle ChromaDB via VectorStoreManager
    - Retourne les passages les plus pertinents, ou déclenche un refus si
      aucun passage ne dépasse le seuil de similarité minimal

    Ce refus est LÉGITIME : il protège contre les hallucinations en forçant
    le système à ne répondre que sur ce qui est réellement dans le corpus.
    """

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        collection_name: str = "multirag_corpus",
        similarity_threshold: float = 0.40,
        top_k: int = 3
    ):
        """
        Initialise le RetrieverAgent avec son VectorStoreManager.

        Args:
            db_path: Chemin vers la base ChromaDB persistante
            collection_name: Nom de la collection dans ChromaDB
            similarity_threshold: Seuil minimal de similarité cosinus (entre 0 et 1)
            top_k: Nombre maximum de passages à retourner
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # Connexion à la base vectorielle existante
        self.vector_store = VectorStoreManager(
            db_path=db_path,
            collection_name=collection_name
        )

    def run(self, query: str, lang: str) -> RetrieverResult:
        """
        Point d'entrée principal de l'agent.

        Args:
            query: La question de l'utilisateur (texte brut)
            lang: Code langue ISO détecté par Agent 1 (fr, en, ar)

        Returns:
            RetrieverResult avec les chunks pertinents ou un message de refus
        """
        query_clean = query.strip()

        # ── Garde-fou 1 : question vide ──────────────────────────────
        if not query_clean:
            return RetrieverResult(
                query=query,
                lang=lang,
                chunks=[],
                has_results=False,
                message="Aucune recherche effectuée : la question est vide."
            )

        # ── Recherche sémantique dans ChromaDB ───────────────────────
        chunks: List[DocumentChunk] = self.vector_store.search(
            query=query_clean,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )

        # ── Garde-fou 2 : aucun résultat pertinent ───────────────────
        if not chunks:
            return RetrieverResult(
                query=query_clean,
                lang=lang,
                chunks=[],
                has_results=False,
                message=(
                    "Aucun passage pertinent trouvé dans le corpus pour cette question. "
                    "Le système refuse de répondre par manque de base factuelle."
                )
            )

        # ── Succès : passages trouvés ─────────────────────────────────
        scores_info = ", ".join([f"{c.score_similarite:.4f}" for c in chunks])
        return RetrieverResult(
            query=query_clean,
            lang=lang,
            chunks=chunks,
            has_results=True,
            message=(
                f"{len(chunks)} passage(s) pertinent(s) récupéré(s) "
                f"(scores de similarité : {scores_info})"
            )
        )
