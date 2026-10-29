from pydantic import BaseModel, Field
from typing import Optional
from langdetect import detect, DetectorFactory

# Fixer la graine aléatoire pour rendre la détection déterministe
DetectorFactory.seed = 0

class LanguageResult(BaseModel):
    lang: str = Field(..., description="Code langue détecté : fr, en, ar, etc.")
    is_supported: bool = Field(..., description="Indique si la langue fait partie des langues supportées")
    message: Optional[str] = Field(default=None, description="Message d'avertissement ou d'erreur")

class LanguageAgent:
    """
    Agent 1 : Détection automatique de la langue de la requête.
    Garantit la transmission du code langue aux agents d'orchestration et de réponse.
    """
    SUPPORTED_LANGUAGES = {"fr", "en", "ar"}

    def __init__(self):
        pass

    def run(self, query: str) -> LanguageResult:
        query_clean = query.strip()
        if not query_clean:
            return LanguageResult(
                lang="unknown",
                is_supported=False,
                message="La question soumise est vide."
            )
        
        try:
            detected_lang = detect(query_clean)
        except Exception as e:
            return LanguageResult(
                lang="unknown",
                is_supported=False,
                message=f"Erreur lors de l'analyse linguistique : {str(e)}"
            )

        if detected_lang in self.SUPPORTED_LANGUAGES:
            return LanguageResult(
                lang=detected_lang,
                is_supported=True,
                message=f"Langue supportée détectée : {detected_lang.upper()}"
            )
        else:
            return LanguageResult(
                lang=detected_lang,
                is_supported=False,
                message=f"Langue '{detected_lang}' non supportée. Veuillez poser votre question en Français, Anglais ou Arabe."
            )
