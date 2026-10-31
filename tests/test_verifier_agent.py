"""
Tests unitaires et d'intégration pour VerifierAgent (Agent 4).

Vérifie :
  1. Réponse 100% fidèle -> acceptée (is_faithful=True, action='ACCEPT')
  2. Réponse avec hallucination inventée -> détectée et bloquée (is_faithful=False)
  3. Refus explicite certifié -> accepté (action='ACCEPT')
"""

import sys
import os

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.verifier_agent import VerifierAgent, VerifierResult
from src.rag.models import DocumentChunk


def create_sample_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="notice_fr_001_chunk_0",
        notice_id="notice_fr_001",
        titre="Royaume de Numidie",
        langue="fr",
        source_url="https://fr.wikipedia.org/wiki/Numidie",
        texte=(
            "La Numidie est un ancien royaume berbère qui a existé en Afrique du Nord "
            "de 202 av. J.-C. à 46 av. J.-C. Il a été unifié sous le règne du roi Masinissa, "
            "qui a établi Cirta (actuelle Constantine) comme sa capitale."
        ),
        score_similarite=0.88
    )


def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST : {title}")
    print('=' * 60)


def print_result(result: VerifierResult):
    print(f"  -> is_faithful     : {result.is_faithful}")
    print(f"  -> fidelity_score  : {result.fidelity_score}")
    print(f"  -> action          : {result.action_recommended}")
    print(f"  -> feedback        : {result.feedback}")
    if result.hallucinations_detected:
        print(f"  -> hallucinations  : {result.hallucinations_detected}")


def test_reponse_fidele(agent: VerifierAgent):
    """Test 1 : Réponse parfaitement conforme aux sources -> doit être validée."""
    print_separator("Reponse fidele -> Validation ACCEPT")

    chunk = create_sample_chunk()
    response_text = "La capitale de la Numidie était Cirta [1], et le royaume a été unifié par Masinissa [1]."

    result = agent.run(
        query="Quelle est la capitale de la Numidie ?",
        response_text=response_text,
        chunks=[chunk]
    )
    print_result(result)

    assert isinstance(result, VerifierResult)
    assert result.is_faithful is True, "Une réponse fidèle doit avoir is_faithful = True"
    assert result.fidelity_score == 1.0, "Le score de fidélité doit être de 1.0 (100%)"
    assert result.action_recommended == "ACCEPT", "L'action recommandée doit être ACCEPT"
    assert len(result.hallucinations_detected) == 0, "Aucune hallucination ne doit être détectée"

    print("\n  [OK] TEST 1 PASSE")


def test_reponse_hallucinee(agent: VerifierAgent):
    """Test 2 : Réponse contenant un fait inventé -> doit être détectée et rejetée."""
    print_separator("Reponse avec hallucination -> Detection et Rejet")

    chunk = create_sample_chunk()
    # Hallucination délibérée : invention d'une tour Eiffel et d'un traité à Tokyo
    response_text = "Masinissa a construit la Tour Eiffel à Cirta et a signé un traité de paix à Tokyo en 1995."

    result = agent.run(
        query="Qu'a fait Masinissa ?",
        response_text=response_text,
        chunks=[chunk]
    )
    print_result(result)

    assert isinstance(result, VerifierResult)
    assert result.is_faithful is False, "L'hallucination doit rendre is_faithful = False"
    assert len(result.hallucinations_detected) > 0, "Au moins une hallucination doit être recensée"
    assert result.action_recommended in ["RETRY", "REFUSE"], "L'action doit être RETRY ou REFUSE"

    print("\n  [OK] TEST 2 PASSE")


def test_refus_legitime(agent: VerifierAgent):
    """Test 3 : Refus explicite certifié -> doit être immédiatement accepté."""
    print_separator("Refus explicite certifie -> Validation")

    chunk = create_sample_chunk()
    response_text = "Information non trouvée dans le corpus documentaire fourni."

    result = agent.run(
        query="Quelle est la recette du couscous ?",
        response_text=response_text,
        chunks=[chunk]
    )
    print_result(result)

    assert isinstance(result, VerifierResult)
    assert result.is_faithful is True, "Un refus légitime doit être validé comme conforme"
    assert result.action_recommended == "ACCEPT", "L'action recommandée pour un refus légitime est ACCEPT"

    print("\n  [OK] TEST 3 PASSE")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTS D'INTEGRATION - VerifierAgent (Agent 4)")
    print("=" * 60)

    agent = VerifierAgent()
    print(f"[OK] Agent Verifier initialise avec le modele : {agent.model}")

    tests = [
        test_reponse_fidele,
        test_reponse_hallucinee,
        test_refus_legitime
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn(agent)
            passed += 1
        except AssertionError as e:
            print(f"\n  [ECHEC] : {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTATS : {passed} passes / {failed} echoues / {len(tests)} total")
    print(f"{'=' * 60}\n")
