import os
import sys
import json
import time
from typing import List, Dict, Any

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.orchestrator import MultiAgentOrchestrator, PipelineResult


def run_benchmark(dataset_path: str = "data/eval/questions_test.json"):
    print("=" * 70)
    print("  BANC D'ESSAI SCIENTIFIQUE : ÉVALUATION DES 30 QUESTIONS (BENCHMARK)")
    print("  Assistant MultiRAG-Agent — Master 2 & Candidature Recherche Cnam/Efrei")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"❌ Fichier de questions introuvable : {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n[1] Chargement de {len(questions)} questions de test (FR: 10, EN: 10, AR: 10)...")
    in_corpus_count = sum(1 for q in questions if q["type"] == "in_corpus")
    out_corpus_count = sum(1 for q in questions if q["type"] == "out_of_corpus")
    print(f"    - Questions factuelles dans le corpus : {in_corpus_count}")
    print(f"    - Questions pièges hors-corpus        : {out_corpus_count}")

    print("\n[2] Initialisation du système Multi-Agents (ChromaDB + MPNet + Gemini)...")
    orchestrator = MultiAgentOrchestrator(
        db_path="data/chroma_db",
        collection_name="multirag_corpus",
        similarity_threshold=0.40,
        top_k=3
    )

    os.makedirs("logs", exist_ok=True)
    traces_log_path = "logs/traces.jsonl"
    report_path = "logs/benchmark_report.json"

    results = []
    latencies = []
    pertinence_matches = 0
    total_faithful_claims = 0
    total_claims_audited = 0
    hallucination_cases = 0
    justified_refusals = 0

    print("\n[3] Exécution des requêtes à travers les 5 agents...")
    print("-" * 70)

    # Ouvrir le fichier de log JSONL
    with open(traces_log_path, "w", encoding="utf-8") as trace_file:
        for idx, q_item in enumerate(questions, 1):
            q_id = q_item["id"]
            lang = q_item["lang"]
            q_type = q_item["type"]
            q_text = q_item["question"]

            print(f"[{idx:02d}/30] ({lang.upper()}) [{q_type:13}] {q_text[:45]}...", end=" ", flush=True)

            # Exécution par l'orchestrateur
            t0 = time.time()
            res: PipelineResult = orchestrator.process(q_text)
            elapsed_ms = res.total_latency_ms
            latencies.append(elapsed_ms)

            # Analyse des métriques
            is_pertinent = False
            is_justified_refusal = False
            has_hallucination = False

            if q_type == "in_corpus":
                # Vérifier si la réponse a été produite avec succès
                if res.status == "SUCCESS":
                    is_pertinent = True
                    pertinence_matches += 1

                # Vérifier la fidélité
                if res.fidelity_score < 1.0:
                    has_hallucination = True
                    hallucination_cases += 1

            elif q_type == "out_of_corpus":
                # Doit être un refus légitime
                if res.status in ["REFUSAL_NO_CORPUS", "REFUSAL_LANGUAGE", "REFUSAL_VERIFICATION", "AMBIGUOUS_SUGGESTION"]:
                    is_justified_refusal = True
                    justified_refusals += 1
                else:
                    # Alerte : le système a répondu sur une question hors corpus !
                    has_hallucination = True
                    hallucination_cases += 1

            # Journaliser dans traces.jsonl
            log_entry = {
                "id": q_id,
                "timestamp": time.time(),
                "lang": lang,
                "type": q_type,
                "question": q_text,
                "ground_truth": q_item["ground_truth"],
                "status": res.status,
                "fidelity_score": res.fidelity_score,
                "citations_count": len(res.citations),
                "latency_ms": elapsed_ms,
                "answer": res.final_answer,
                "agent_trace": [t.model_dump() for t in res.trace]
            }
            trace_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            trace_file.flush()

            status_display = "OK" if (is_pertinent or is_justified_refusal) else "WARN"
            print(f"-> {res.status:18} | {int(elapsed_ms):4}ms | [{status_display}]")

    # Calcul des métriques globales
    pertinence_rate = round((pertinence_matches / in_corpus_count) * 100, 2)
    refusal_rate = round((justified_refusals / out_corpus_count) * 100, 2)
    hallucination_rate = round((hallucination_cases / len(questions)) * 100, 2)
    fidelity_rate = round(100.0 - hallucination_rate, 2)
    avg_latency = round(sum(latencies) / len(latencies), 2)

    # Sauvegarder le rapport JSON
    summary_report = {
        "projet": "MultiRAG-Agent",
        "auteur": "Amine GHAYOUT",
        "date_evaluation": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "in_corpus_tested": in_corpus_count,
        "out_of_corpus_tested": out_corpus_count,
        "metriques": {
            "pertinence_documentaire": f"{pertinence_rate}%",
            "fidelite_sourcage": f"{fidelity_rate}%",
            "taux_hallucinations": f"{hallucination_rate}%",
            "refus_justifie_abstention": f"{refusal_rate}%",
            "latence_moyenne_ms": f"{avg_latency} ms"
        },
        "cibles_cahier_des_charges": {
            "pertinence_cible": ">= 85%",
            "fidelite_cible": ">= 95%",
            "hallucination_cible": "<= 5%",
            "refus_cible": "100%",
            "latence_cible": "< 4000 ms"
        }
    }

    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(summary_report, rf, ensure_ascii=False, indent=2)

    # Affichage du rapport final
    print("\n" + "=" * 70)
    print("  RAPPORT SCIENTIFIQUE D'ÉVALUATION FINALE (BENCHMARK)")
    print("=" * 70)
    print(f"  • Pertinence Documentaire (In-Corpus)  : {pertinence_rate}%  (Cible CDC: >= 85%)")
    print(f"  • Fidélité Factuelle aux Sources      : {fidelity_rate}%  (Cible CDC: >= 95%)")
    print(f"  • Taux d'Hallucinations Détectées     : {hallucination_rate}%   (Cible CDC: <= 5%)")
    print(f"  • Refus Justifié (Questions Pièges)   : {refusal_rate}% (Cible CDC: 100%)")
    print(f"  • Latence Moyenne de Traitement       : {avg_latency} ms (Cible CDC: < 4000 ms)")
    print("=" * 70)
    print(f"\n[OK] Rapport complet sauvegardé dans : {report_path}")
    print(f"[OK] Traces analytiques sauvegardées dans : {traces_log_path}\n")


if __name__ == "__main__":
    run_benchmark()
