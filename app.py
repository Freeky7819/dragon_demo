import streamlit as st
import os
import sys
from dragon_brain import DragonAgent, LLMBridge

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dragon Agent Demo", page_icon="🐲", layout="wide")

# --- SYSTEM CHECK ---
def check_system():
    status = {"models": True, "folders": True}
    if not os.path.exists("models"):
        os.makedirs("models")
        status["folders"] = False
    if not os.path.exists("samples"):
        os.makedirs("samples")
    
    # Check Weights (Soft check)
    if not os.path.exists("models/dragon_pro_1_16.pth"):
        status["models"] = False
        
    return status

sys_status = check_system()

# --- INITIALIZATION ---
if 'agent' not in st.session_state:
    # Check if models exist, otherwise warn but allow run
    if not sys_status["models"]:
        st.warning("⚠️ Model weights (.pth) missing in /models. Agent will use random init (Garbage output).")
    
    with st.spinner("Awakening Dragon..."):
        # Initialization with default values. We will control them via GUI.
        # Use semantic chunking for GUI (better UX), benchmarks can use "fixed"
        st.session_state.agent = DragonAgent(chunking_strategy="semantic")

agent = st.session_state.agent

# --- SIDEBAR ---
with st.sidebar:
    st.title("🐲 Dragon Control")
    
    # Provider Selection
    st.subheader("LLM Configuration")
    prov = st.selectbox("LLM Provider", ["ollama", "openai", "local_api", "anthropic"])
    
    # Try to keep the currently selected model
    current_model = getattr(agent.llm, 'model_name', None)
    default_model = current_model if current_model else ("llama3" if prov == "ollama" else "gpt-3.5-turbo")
    
    model_name = st.text_input("Model Name", value=default_model)
    api_key = ""
    # Check if API key is required (Ollama and local_api usually don't need it)
    if prov not in ["ollama", "local_api"]:
        api_key = st.text_input("API Key", type="password")

    # === NEW: Advanced RAG Settings ===
    st.divider()
    st.subheader("🧠 RAG Parameters (Advanced)")

    # Read current values from agent for controller initialization (use getattr for safety)
    current_sensitivity = float(getattr(agent, 'compress_sensitivity', 1.0))
    current_debias = float(getattr(agent, 'source_debias_lambda', 0.0))
    current_neg_thr = float(getattr(agent, 'neg_thr', 0.59))
    current_balanced = bool(getattr(agent, 'use_balanced_search', True))
    current_hybrid = bool(getattr(agent, 'hybrid_anchor', False))

    # 1. Query-Time Settings (Instant Effect)
    st.caption("Query-Time Settings (Instant Effect)")
    
    # Formatting for better display of decimal numbers in Streamlit
    formatted_debias = float(f"{current_debias:.4f}")
    source_debias_lambda = st.number_input("Source Debiasing Lambda", min_value=0.0, max_value=0.2, value=formatted_debias, step=0.005, help="Penalty coefficient for large sources (e.g. 0.015 for books). 0.0 = Disabled.")
    
    neg_thr = st.number_input("Text Threshold (neg_thr)", min_value=0.3, max_value=0.9, value=current_neg_thr, step=0.01, help="Minimum similarity score for text inclusion.")
    use_balanced_search = st.toggle("Balanced Search (Rerank)", value=current_balanced, help="Rerank results to ensure source diversity.")

    # 2. Ingestion-Time Settings (Requires Reprocessing)
    st.caption("Ingestion-Time Settings (Requires Reprocessing)")
    # Sensitivity explanation: Values around 1.0 are default.
    compress_sensitivity = st.number_input("Compression Sensitivity", min_value=0.1, max_value=5.0, value=current_sensitivity, step=0.1, help="Controls vector density. Default: 1.0")
    hybrid_anchor = st.toggle("Hybrid Anchor", value=current_hybrid, help="Store one global vector per text chunk. Improves stability, but increases memory usage for text.")
    
    # ===================================

    # Update Agent (Combined button for LLM and RAG)
    if st.button("Update Agent Settings"):
        try:
            # 1. Update LLM
            # Try to initialize LLMBridge to catch any errors (e.g. missing packages)
            new_llm = LLMBridge(provider=prov, model_name=model_name, api_key=api_key)
            agent.llm = new_llm
            
            # 2. Update RAG parameters
            agent.source_debias_lambda = source_debias_lambda
            agent.neg_thr = neg_thr
            agent.use_balanced_search = use_balanced_search
            agent.compress_sensitivity = compress_sensitivity
            agent.hybrid_anchor = hybrid_anchor
            
            st.success("✅ Agent settings updated successfully!")
            
            # 3. Critical warning if Ingestion settings have changed
            if (current_sensitivity != compress_sensitivity) or (current_hybrid != hybrid_anchor):
                st.warning("⚠️ WARNING: Data ingestion settings (Sensitivity/Anchor) have been changed. These apply only to NEW data. To apply changes to the existing knowledge base, you MUST clear memory ('Clear Memory') and reprocess files.")

        except ImportError as e:
            # Robust error handling if any packages are missing (openai, ollama, anthropic)
            st.error(f"❌ Error initializing LLM: {e}. Please ensure the required package is installed.")
        except Exception as e:
            st.error(f"❌ Error updating agent: {e}")

    st.divider()
    st.subheader("Controls")
    visual_mode = st.toggle("👁️ Visual Deep Search", help="Lowers thresholds for image retrieval.")
    
    # Button for clearing memory
    if st.button("🧹 Clear Memory"):
        agent.memory_vectors = []
        agent.memory_texts = []
        agent.memory_types = []
        agent.memory_modalities = []
        agent.processed_files = set()
        agent.chat_history = []
        agent.source_counts = {}
        st.rerun()

# --- MAIN UI ---
st.title("Dragon Agent Demo")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Knowledge Base", "📊 Dashboard"])

# TAB 1: CHAT
with tab1:
    # History
    for msg in agent.chat_history:
        role = msg["role"]
        avt = "🐲" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avt):
            st.markdown(msg["content"])
            if "rag_info" in msg and msg["rag_info"]:
                with st.expander("🧠 RAG Thoughts"):
                    for match in msg["rag_info"]:
                        # Handle both old format (sim, text, mtype) and new format (raw_sim, debias_sim, text, mtype)
                        if len(match) == 4:
                            raw_sim, debias_sim, text, mtype = match
                            score = debias_sim  # Use debiased similarity for display
                        elif len(match) == 3:
                            score, text, mtype = match
                        else:
                            continue  # Skip invalid formats
                        icon = "🖼️" if mtype == "image" else "📄"
                        # Improved score display
                        st.caption(f"{icon} [Score: {score:.2f}] {text[:100]}...")

    # Input
    c1, c2 = st.columns([4, 1])
    with c2:
        img_upload = st.file_uploader("Upload Image", type=['png', 'jpg'], key="chat_img")
    
    prompt = st.chat_input("Ask Dragon...")
    
    if prompt:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🐲"):
            with st.spinner("Dragon is thinking..."):
                img_pil = None
                if img_upload:
                    from PIL import Image
                    img_pil = Image.open(img_upload)
                    
                resp = agent.chat(prompt, image_input=img_pil, visual_mode=visual_mode)
                st.rerun()

# TAB 2: KNOWLEDGE
with tab2:
    st.header("Feed the Dragon")
    files = st.file_uploader("Upload Text/Images to Memory", accept_multiple_files=True)
    if files and st.button("Process Files"):
        bar = st.progress(0)
        for i, f in enumerate(files):
            stat = agent.process_file(f, f.name)
            st.write(f"{f.name}: {stat}")
            bar.progress((i + 1) / len(files))
        st.success("Processing Complete!")

# TAB 3: DASHBOARD
with tab3:
    c1, c2, c3 = st.columns(3)
    n_vec = len(agent.memory_vectors)
    c1.metric("Vectors Stored", n_vec)
    c2.metric("Documents", len(agent.processed_files))
    c3.metric("Vision Cortex", "ACTIVE" if agent.has_vision else "OFFLINE")
    
    if n_vec > 0:
        st.subheader("Memory Inspector")
        # CHANGE: Removed [:50], now showing full content!
        st.dataframe(
            [{"Type": t, "Content": txt} for t, txt in zip(agent.memory_types, agent.memory_texts)],
            use_container_width=True
        )

if not sys_status["models"]:
    st.error("❌ MISSING MODELS: Please place 'dragon_pro_1_16.pth' and 'dragon_vision_v1.pth' in the 'models' folder.")