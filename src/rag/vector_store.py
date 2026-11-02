import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

from src.rag.models import DocumentNotice, DocumentChunk

class VectorStoreManager:
    """
    Gestionnaire de la base vectorielle ChromaDB.
    Permet l'indexation de notices documentaires (avec chunking) et la recherche sémantique multilingue.
    """

    def __init__(self, db_path: str = "data/chroma_db", collection_name: str = "multirag_corpus"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # S'assurer que le dossier parent existe
        os.makedirs(self.db_path, exist_ok=True)
        
        # 1. Initialiser le client ChromaDB avec persistance locale
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 2. Modèle d'embeddings multilingue Sentence-Transformers
        # Modèle de référence : paraphrase-multilingual-mpnet-base-v2 (ou sentence-transformers par défaut)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # 3. Créer ou récupérer la collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}  # Distance cosinus pour évaluer la similarité sémantique
        )

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Découpe un texte long en plusieurs passages (chunks) avec chevauchement.
        """
        words = text.split()
        if not words:
            return []
        
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += (chunk_size - overlap)
            
        return chunks

    def add_notice(self, notice: DocumentNotice, chunk_size: int = 300, overlap: int = 50) -> int:
        """
        Découpe une notice en fragments et les insère dans ChromaDB avec leurs métadonnées.
        """
        text_chunks = self.chunk_text(notice.texte, chunk_size=chunk_size, overlap=overlap)
        if not text_chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{notice.id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                "chunk_id": chunk_id,
                "notice_id": notice.id,
                "titre": notice.titre,
                "langue": notice.langue,
                "source_url": notice.source_url,
                "theme": notice.theme
            })

        # Insertion ou mise à jour dans ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return len(ids)

    def search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.40) -> List[DocumentChunk]:
        """
        Recherche les passages les plus pertinents pour une requête dans ChromaDB.
        Filtre les résultats selon un seuil minimal de similarité cosinus.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        matched_chunks: List[DocumentChunk] = []

        if not results or not results["documents"] or not results["documents"][0]:
            return matched_chunks

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

        for doc_text, meta, distance in zip(docs, metas, distances):
            # Pour la distance cosinus de ChromaDB : similarité = 1.0 - distance
            similarity = 1.0 - distance

            # Filtrage selon le seuil
            if similarity >= similarity_threshold:
                chunk = DocumentChunk(
                    chunk_id=meta["chunk_id"],
                    notice_id=meta["notice_id"],
                    titre=meta["titre"],
                    langue=meta["langue"],
                    source_url=meta["source_url"],
                    texte=doc_text,
                    score_similarite=round(similarity, 4)
                )
                matched_chunks.append(chunk)

        return matched_chunks

    def count(self) -> int:
        """Retourne le nombre total de passages actuellement indexés."""
        return self.collection.count()
