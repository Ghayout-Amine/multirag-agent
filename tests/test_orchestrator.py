"""
Tests d'intégration de bout en bout pour l'Orchestrateur (Agent 0).

Scénarios testés :
  1. Requête FR dans le corpus -> Pipeline complet SUCCESS avec citations
  2. Requête Arabe dans le corpus -> Traitement multilingue AR SUCCESS
  3. Requête hors langues supportées (ex: espagnol) -> REFUSAL_LANGUAGE
  4. Requête hors corpus (ex: recette de cuisine) -> REFUSAL_NO_CORPUS
"""

import sys
import os

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import MultiAgentOrchestrator, PipelineResult


def print_separator(title: str):
    print(f"\n{'=' * 65}")
    print(f"  TEST ORCHESTRATEUR : {title}")
    print('=' * 65)


def safe_print(text: str) -> str:
    """Evite les erreurs d'encodage cp1252 sous Windows terminal."""
    try:
        return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")


def print_pipeline_result(result: PipelineResult):
    print(safe_print(f"  -> Question       : {result.query}"))
    print(safe_print(f"  -> Langue         : {result.lang}"))
    print(safe_print(f"  -> Statut         : {result.status}"))
    print(safe_print(f"  -> Fidelite       : {result.fidelity_score * 100}%"))
    print(safe_print(f"  -> Latence        : {result.total_latency_ms} ms"))
    print(safe_print(f"  -> Nb citations   : {len(result.citations)}"))
    print(safe_print(f"  -> Reponse Finale :\n{result.final_answer}\n"))
    print("  -> Traces des agents :")
    for step in result.trace:
        print(safe_print(f"     [{step.agent}] ({step.duration_ms} ms) -> {step.status} | {step.action}"))


def test_pipeline_complet_fr(orchestrator: MultiAgentOrchestrator):
    """Test 1 : Requête française dans le corpus traitée de bout en bout."""
    print_separator("Requete FR dans le corpus (Pipeline complet)")

    result = orchestrator.process("Qui etait le roi Masinissa et quel royaume a-t-il unifie ?")
    print_pipeline_result(result)

    assert result.status == "SUCCESS", "Le statut global doit être SUCCESS"
    assert result.lang == "fr", "La langue détectée doit être fr"
    assert result.fidelity_score == 1.0, "La fidélité doit être de 100%"
    assert len(result.citations) > 0, "Des citations formelles doivent être incluses"
    assert "Masinissa" in result.final_answer or "Numidie" in result.final_answer
    assert len(result.trace) >= 4, "Toutes les étapes d'agents doivent figurer dans la trace"

    print("\n  [OK] TEST 1 PASSE")


def test_pipeline_arabe(orchestrator: MultiAgentOrchestrator):
    """Test 2 : Requête en langue arabe traitée de bout en bout."""
    print_separator("Requete Arabe dans le corpus")

    result = orchestrator.process("ما هي عاصمة مملكة نوميديا ومن هو الملك الذي وحدها؟")
    print_pipeline_result(result)

    assert result.status == "SUCCESS", "Le statut global doit être SUCCESS"
    assert result.lang == "ar", "La langue détectée doit être ar"
    assert len(result.citations) > 0, "Des citations formelles doivent être incluses"

    print("\n  [OK] TEST 2 PASSE")


def test_refus_langue_non_supportee(orchestrator: MultiAgentOrchestrator):
    """Test 3 : Langue non supportée (ex: espagnol) -> Refus précoce."""
    print_separator("Langue non supportee -> Refus precoce Agent 1")

    result = orchestrator.process("¿Cuál es la capital del antiguo reino de Numidia?")
    print_pipeline_result(result)

    assert result.status == "REFUSAL_LANGUAGE", "Le statut doit être REFUSAL_LANGUAGE"
    assert len(result.citations) == 0, "Aucune citation pour un refus de langue"
    assert len(result.trace) == 1, "Le pipeline doit s'arrêter dès l'Agent 1"

    print("\n  [OK] TEST 3 PASSE")


def test_refus_hors_corpus(orchestrator: MultiAgentOrchestrator):
    """Test 4 : Question hors corpus -> Refus légitime Agent 2."""
    print_separator("Question hors corpus -> Refus precoce Agent 2")

    result = orchestrator.process("Comment preparer une paella traditionnelle espagnole ?")
    print_pipeline_result(result)

    assert result.status == "REFUSAL_NO_CORPUS", "Le statut doit être REFUSAL_NO_CORPUS"
    assert len(result.citations) == 0, "Aucune citation pour une question hors corpus"
    assert len(result.trace) == 2, "Le pipeline doit s'arrêter à l'Agent 2 sans appeler le LLM"

    print("\n  [OK] TEST 4 PASSE")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TESTS D'INTEGRATION DE BOUT EN BOUT - ORCHESTRATEUR (Agent 0)")
    print("=" * 65)

    orchestrator = MultiAgentOrchestrator(
        db_path="data/chroma_db",
        collection_name="multirag_corpus",
        similarity_threshold=0.40,
        top_k=3
    )

    tests = [
        test_pipeline_complet_fr,
        test_pipeline_arabe,
        test_refus_langue_non_supportee,
        test_refus_hors_corpus
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn(orchestrator)
            passed += 1
        except AssertionError as e:
            print(f"\n  [ECHEC] : {e}")
            failed += 1

    print(f"\n{'=' * 65}")
    print(f"  RESULTATS GLOBAUX : {passed} passes / {failed} echoues / {len(tests)} total")
    print(f"{'=' * 65}\n")
