import sys
import os
import streamlit as st

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import MultiAgentOrchestrator, PipelineResult


# ──────────────────────────────────────────────
# Configuration de la page Streamlit
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MultiRAG-Agent | Assistant Multilingue Sourcé",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour une interface moderne et académique
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .agent-pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pill-success { background-color: #D1FAE5; color: #065F46; }
    .pill-warning { background-color: #FEF3C7; color: #92400E; }
    .pill-danger  { background-color: #FEE2E2; color: #991B1B; }
    .source-card {
        border-left: 4px solid #3B82F6;
        background-color: #F9FAFB;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Mise en cache de l'Orchestrateur (Singleton)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du système Multi-Agents et de ChromaDB...")
def load_orchestrator():
    return MultiAgentOrchestrator(
        db_path="data/chroma_db",
        collection_name="multirag_corpus",
        similarity_threshold=0.40,
        top_k=3
    )

orchestrator = load_orchestrator()


# ──────────────────────────────────────────────
# Barre latérale (Sidebar) - Présentation & Paramètres
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.markdown("### **MultiRAG-Agent**")
    st.caption("Projet de Master 2 — Automatisation & IA Agentique\n**Amine GHAYOUT**")
    
    st.divider()
    
    st.markdown("#### 🤖 Architecture des 5 Agents")
    st.markdown("""
    - **Agent 0** : Orchestrateur central
    - **Agent 1** : Détection langue (FR, EN, AR)
    - **Agent 2** : Recherche sémantique (ChromaDB + MPNet)
    - **Agent 3** : Réponse factuelle sourcée (Gemini)
    - **Agent 4** : Vérificateur NLI (Zéro hallucination)
    """)
    
    st.divider()
    st.markdown("#### 💡 Exemples de questions")
    
    if st.button("🇫🇷 Qui était le roi Masinissa ?"):
        st.session_state["user_query"] = "Qui était le roi Masinissa et quel royaume a-t-il unifié ?"
        
    if st.button("🇬🇧 Who was King Masinissa?"):
        st.session_state["user_query"] = "Who was King Masinissa and what did he accomplish?"
        
    if st.button("🇲🇦 من هو الملك ماسينيسا؟"):
        st.session_state["user_query"] = "من هو الملك ماسينيسا وما هي إنجازاته؟"
        
    if st.button("🚫 Piège hors-corpus (Couscous)"):
        st.session_state["user_query"] = "Donne-moi la recette traditionnelle du couscous royal."


# ──────────────────────────────────────────────
# Zone Principale de l'Application
# ──────────────────────────────────────────────
st.markdown("<div class='main-header'>🛡️ MultiRAG-Agent</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Assistant RAG Multilingue avec Réponses Sourcées et Garantie Zéro Hallucination</div>",
    unsafe_allow_html=True
)

# Champ de saisie
default_q = st.session_state.get("user_query", "")
query = st.text_input(
    "Posez votre question en Français, Anglais ou Arabe :",
    value=default_q,
    placeholder="Ex: Qui était le roi Masinissa ? / Who was King Masinissa? / من هو الملك ماسينيسا؟",
    key="input_query"
)

col_btn, col_clear = st.columns([1, 6])
with col_btn:
    submit = st.button("🔍 Interroger", type="primary", use_container_width=True)

if submit and query.strip():
    with st.spinner("Exécution de la chaîne d'agents en cours..."):
        result: PipelineResult = orchestrator.process(query.strip())
        
    st.divider()
    
    # En-tête des résultats : Métriques et Statuts
    col_stat, col_fid, col_lat, col_lang = st.columns(4)
    
    with col_stat:
        if result.status == "SUCCESS":
            st.success("✅ Réponse Certifiée")
        elif result.status == "REFUSAL_LANGUAGE":
            st.info("🌐 Langue Non Supportée")
        elif result.status == "REFUSAL_NO_CORPUS":
            st.warning("📚 Hors Corpus Documentaire")
        else:
            st.error("🛡️ Refus Anti-Hallucination")
            
    with col_fid:
        st.metric("Fidélité Factuelle", f"{int(result.fidelity_score * 100)}%")
        
    with col_lat:
        st.metric("Latence Totale", f"{int(result.total_latency_ms)} ms")
        
    with col_lang:
        st.metric("Langue Détectée", result.lang.upper())
        
    st.markdown("### 📝 Réponse formulée")
    st.markdown(f"**{result.final_answer}**")
    
    # ── Section 1 : Sources documentaires citées ──
    if result.citations:
        st.markdown("### 📚 Sources Documentaires Utilisées")
        for cit in result.citations:
            st.markdown(f"""
            <div class='source-card'>
                <strong>Source [{cit.get('source_num', '1')}] : {cit.get('titre', 'Document')}</strong> 
                <span class='agent-pill pill-success'>{cit.get('langue', 'fr').upper()}</span><br>
                <small>Identifiant notice : <code>{cit.get('notice_id')}</code> | Chunk : <code>{cit.get('chunk_id')}</code></small><br>
                <a href="{cit.get('source_url', '#')}" target="_blank">🔗 Consulter la source originale</a>
            </div>
            """, unsafe_allow_html=True)
            
    # ── Section 2 : Traçabilité et Audit Multi-Agents ──
    with st.expander("🔍 Voir la trace d'exécution détaillée des 5 Agents (Audit)"):
        st.markdown("#### Chronologie de la décision :")
        for step in result.trace:
            st.markdown(f"""
            - **{step.agent}** : `{step.action}`  
              *Statut :* `{step.status}` | *Durée :* `{step.duration_ms} ms`  
              *Détails :* `{step.details}`
            """)
            
        if result.chunks_used:
            st.markdown("#### Extraits bruts transmis par ChromaDB :")
            for i, chunk in enumerate(result.chunks_used, 1):
                st.text_area(
                    f"Extrait [{i}] - {chunk.titre} (Score: {chunk.score_similarite})",
                    value=chunk.texte,
                    height=100,
                    disabled=True
                )
