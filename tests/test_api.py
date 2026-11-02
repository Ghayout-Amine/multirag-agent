"""
Tests automatisés des endpoints de l'API FastAPI (src/main.py).
Vérifie la conformité de l'API REST avec TestClient.
"""

import sys
import os
from fastapi.testclient import TestClient

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import app

client = TestClient(app)


def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST API : {title}")
    print('=' * 60)


def test_root_endpoint():
    """Test 1 : Point d'entrée racine GET /."""
    print_separator("GET / (Racine)")
    response = client.get("/")
    print("Status code :", response.status_code)
    print("Response JSON :", response.json())

    assert response.status_code == 200
    data = response.json()
    assert data["projet"] == "MultiRAG-Agent"
    assert data["auteur"] == "Amine GHAYOUT"
    assert "/docs" in data["documentation_interactive_swagger"]
    print("\n  [OK] TEST 1 PASSE")


def test_health_endpoint():
    """Test 2 : Supervision GET /health."""
    print_separator("GET /health (Supervision ChromaDB & Modèles)")
    response = client.get("/health")
    print("Status code :", response.status_code)
    print("Response JSON :", response.json())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["indexed_documents_count"] >= 60
    assert "Gemini" in data["llm_model"]
    print("\n  [OK] TEST 2 PASSE")


def test_ask_empty_query():
    """Test 3 : Validation d'erreur POST /ask avec question vide."""
    print_separator("POST /ask (Question vide -> Erreur 400)")
    response = client.post("/ask", json={"query": "   "})
    print("Status code :", response.status_code)
    print("Response JSON :", response.json())

    assert response.status_code == 400
    assert "vide" in response.json()["detail"].lower()
    print("\n  [OK] TEST 3 PASSE")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTS AUTOMATISES DES ENDPOINTS FASTAPI")
    print("=" * 60)

    test_root_endpoint()
    test_health_endpoint()
    test_ask_empty_query()

    print(f"\n{'=' * 60}")
    print("  TOUS LES TESTS FASTAPI SONT PASSES AVEC SUCCES (3/3) !")
    print(f"{'=' * 60}\n")
