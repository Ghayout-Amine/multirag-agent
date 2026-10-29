import sys
import os

# Ajouter le dossier racine au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.language_agent import LanguageAgent

def test_language_detection():
    agent = LanguageAgent()

    # Test 1 : Français
    res_fr = agent.run("Quelle est la capitale de la Numidie ?")
    print(f"FR Test -> Lang: {res_fr.lang}, Supported: {res_fr.is_supported}")
    assert res_fr.lang == "fr" and res_fr.is_supported

    # Test 2 : Anglais
    res_en = agent.run("Who was the king Masinissa in ancient North Africa?")
    print(f"EN Test -> Lang: {res_en.lang}, Supported: {res_en.is_supported}")
    assert res_en.lang == "en" and res_en.is_supported

    # Test 3 : Arabe
    res_ar = agent.run("من هو الملك ماسينيسا وما هي معركة زامة؟")
    print(f"AR Test -> Lang: {res_ar.lang}, Supported: {res_ar.is_supported}")
    assert res_ar.lang == "ar" and res_ar.is_supported

    # Test 4 : Langue non supportée (ex: Espagnol)
    res_es = agent.run("¿Dónde está la biblioteca pública de la ciudad?")
    print(f"ES Test -> Lang: {res_es.lang}, Supported: {res_es.is_supported}")
    assert not res_es.is_supported

    print("\nTous les tests unitaires de l'Agent Langue ont reussi avec succes ! [OK]")

if __name__ == "__main__":
    test_language_detection()
