from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentNotice(BaseModel):
    """
    Modèle d'une notice documentaire propre dans le corpus.
    Chaque fiche est identifiée, sourcée et multilingue.
    """
    id: str = Field(..., description="Identifiant unique de la notice, ex: notice_fr_001")
    titre: str = Field(..., description="Titre du document ou sujet")
    langue: str = Field(..., description="Code langue ISO : fr, en, ar")
    source_url: str = Field(..., description="URL source (ex: lien Wikipedia ou archive)")
    theme: str = Field(default="Histoire et Patrimoine", description="Thème de classification")
    texte: str = Field(..., description="Contenu textuel complet nettoyé")

class DocumentChunk(BaseModel):
    """
    Modèle d'un passage (chunk) après découpage d'une notice.
    Ce fragment est vectorisé et indexé dans la base ChromaDB.
    """
    chunk_id: str = Field(..., description="Identifiant unique du chunk, ex: notice_fr_001_chunk_0")
    notice_id: str = Field(..., description="Lien de parenté vers la notice d'origine")
    titre: str = Field(..., description="Titre hérité de la notice")
    langue: str = Field(..., description="Langue héritée")
    source_url: str = Field(..., description="URL source pour garantir la citabilité")
    texte: str = Field(..., description="Texte exact du passage (300 tokens env.)")
    score_similarite: Optional[float] = Field(default=None, description="Score cosinus lors du retrieval RAG")
