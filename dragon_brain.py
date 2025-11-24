import os
import re
import io
import base64
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

# --- SAFE IMPORTS ---
try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import easyocr
except ImportError:
    easyocr = None  # If library is not available, set to None

try:
    import anthropic
except ImportError:
    anthropic = None

from dragon_core import Dragon

# ==========================================
# LLM BRIDGE (Communications)
# ==========================================
class LLMBridge:
    def __init__(self, provider="ollama", model_name="llama3", api_key=None, base_url=None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.client = None
        
        if self.provider == "ollama":
            if ollama is None:
                raise ImportError("Ollama provider selected but 'ollama' package is not installed.")
        elif self.provider in ["openai", "deepseek", "local_api", "lm_studio"]:
            if OpenAI is None:
                raise ImportError("OpenAI-style provider selected but 'openai' package is not installed.")
            if not api_key: api_key = "dummy"
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "anthropic":
            if anthropic is None:
                raise ImportError("Anthropic provider selected but 'anthropic' package is not installed.")
            self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages, system_prompt=""):
        full_hist = [{"role": "system", "content": system_prompt}] + messages
        try:
            if self.provider == "ollama":
                if ollama is None:
                    return {"content": "❌ Error: Ollama package is not installed.", "debug": {}}
                resp = ollama.chat(model=self.model_name, messages=full_hist)
                return {"content": resp['message']['content'], "debug": {"prompt": system_prompt}}
            elif self.provider in ["openai", "deepseek", "local_api", "lm_studio"]:
                if self.client is None:
                    return {"content": "❌ Error: OpenAI client not initialized.", "debug": {}}
                resp = self.client.chat.completions.create(model=self.model_name, messages=full_hist, temperature=0.7)
                return {"content": resp.choices[0].message.content, "debug": {"prompt": system_prompt}}
            elif self.provider == "anthropic":
                if self.client is None:
                    return {"content": "❌ Error: Anthropic client not initialized.", "debug": {}}
                resp = self.client.messages.create(model=self.model_name, max_tokens=1024, system=system_prompt, messages=messages)
                return {"content": resp.content[0].text, "debug": {"prompt": system_prompt}}
        except Exception as e:
            return {"content": f"❌ Error: {str(e)}", "debug": {}}

    def classify_intent(self, message):
        prompt = f"Analyze: '{message}'. Return one word: 'SAVE' (if keeping info) or 'SEARCH' (if asking/finding). Default to SEARCH."
        try:
            if self.provider == "ollama" and ollama is not None:
                return ollama.generate(model=self.model_name, prompt=prompt)['response'].strip().upper()
            return "SEARCH"
        except Exception as e:
            print(f"⚠️ Intent classification failed: {e}")
            return "SEARCH"

# ==========================================
# DRAGON AGENT (Logic & Memory)
# ==========================================
class DragonAgent:
    def __init__(self, llm_provider="ollama", llm_model="llama3", neg_thr=0.59, visual_neg_thr=None, source_debias_lambda=0.0, use_balanced_search=True, use_llava: bool = True, hybrid_anchor: bool = False, compress_sensitivity: float = 1.0, compress_min_vectors: int = 2, chunking_strategy="semantic"):
        print("🤖 Initializing Dragon Agent...")
        
        # --- SAFE OCR INITIALIZATION ---
        if easyocr:
            print("📖 Loading OCR Reader...")
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        else:
            print("⚠️ EasyOCR is not installed. Text reading from images will be disabled.")
            self.reader = None
        
        # Load Neural Networks
        self.text_compressor = Dragon(vision_mode=False)
        self.has_vision = False
        try:
            self.vision_compressor = Dragon(vision_mode=True)
            self.has_vision = True
        except Exception as e:
            print(f"⚠️ Vision Model failed to load: {e}")
            print("⚠️ Visual features disabled.")
        
        # Memory Stores
        self.memory_vectors = []
        self.memory_texts = []
        self.memory_types = []
        self.processed_files = set()
        
        # Modal index separation: track which indices belong to which modality
        # TEXT: real book text (not image descriptions)
        # IMAGE_DESC: image descriptions (OCR, LLaVA descriptions)
        # IMAGE_HEX: visual hex vectors (for cross-modal search)
        self.memory_modalities = []  # "text", "image_desc", "image_hex"
        
        # Source balancing: track chunk counts per source
        self.source_counts = {}
        
        # Source debias parameter (log penalty coefficient)
        self.source_debias_lambda = source_debias_lambda
        self.use_balanced_search = use_balanced_search  # Enable per-source top-k rerank by default
        
        # RAG thresholds (from benchmark calibration)
        self.neg_thr = float(neg_thr)
        self.visual_neg_thr = float(visual_neg_thr) if visual_neg_thr is not None else None
        
        # Vision backend control (backwards compatible flag)
        self.use_llava = bool(use_llava)
        
        # Compression tuning (default = old behavior)
        self.hybrid_anchor = bool(hybrid_anchor)
        self.compress_sensitivity = float(compress_sensitivity)
        self.compress_min_vectors = int(compress_min_vectors)
        
        # Chunking strategy: "fixed" (for benchmarks) or "semantic" (for GUI)
        self.chunking_strategy = chunking_strategy
        
        # LLM + chat
        self.llm = LLMBridge(provider=llm_provider, model_name=llm_model)
        self.chat_history = []
    
    def get_src_sizes(self):
        """Return source sizes dictionary for benchmark/debiasing."""
        return dict(self.source_counts)

    def _smart_chunking(self, text, chunk_size=600, overlap=100):
        if self.chunking_strategy == "fixed":
            # OLD logic (for benchmarks) - fixed-size chunks with overlap
            text = re.sub(r'\s+', ' ', text).strip()
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end].strip())
                start = end - overlap
            return chunks
        else:  # self.chunking_strategy == "semantic" (for GUI)
            # NEW smart logic - split by sentences, respect chunk_size
            text = re.sub(r'\s+', ' ', text).strip()
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_len = 0
            
            for sentence in sentences:
                s_len = len(sentence)
                if current_len + s_len > chunk_size and current_chunk:
                    full_chunk = " ".join(current_chunk)
                    chunks.append(full_chunk)
                    
                    # Overlap: take last sentences that fit in overlap window
                    overlap_len = 0
                    new_start_chunk = []
                    for prev_s in reversed(current_chunk):
                        if overlap_len + len(prev_s) < overlap:
                            new_start_chunk.insert(0, prev_s)
                            overlap_len += len(prev_s)
                        else:
                            break
                    
                    current_chunk = new_start_chunk
                    current_len = overlap_len
                
                current_chunk.append(sentence)
                current_len += s_len
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks

    def analyze_image(self, img, fast=False, skip_ocr=None, **kwargs):
        """
        Analyze image using OCR and/or LLaVA vision model.
        
        Args:
            img: PIL Image or numpy array
            fast: optional hint from benchmarks; currently ignored.
            skip_ocr: If True, skip OCR and use only LLaVA (for fair comparison).
                      If None, uses self._skip_ocr if set (for benchmark control).
            kwargs: swallow future flags safely.
        """
        # Use instance flag if skip_ocr not explicitly provided
        if skip_ocr is None:
            skip_ocr = getattr(self, '_skip_ocr', False)
        
        desc = ""
        # --- SAFE OCR USAGE ---
        if not skip_ocr and self.reader:
            try:
                ocr = self.reader.readtext(np.array(img), detail=0)
                if ocr: desc += f"DETECTED TEXT: {', '.join(ocr)}. "
            except Exception as e:
                print(f"OCR Error: {e}")
        
        # --- OLLAMA LLAVA VISION DESCRIPTION ---
        if self.use_llava and ollama is not None:
            try:
                # Convert image to base64 format for Ollama
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Attempt to use Ollama Llava for image description
                # Format: messages with 'images' list of base64 strings
                vision_response = ollama.chat(
                    model='llava',
                    messages=[{
                        'role': 'user',
                        'content': 'Describe this image in detail in English. What do you see in the image? What objects, colors, text, or people are visible?',
                        'images': [img_base64]
                    }]
                )
                
                if vision_response and 'message' in vision_response:
                    vision_desc = vision_response['message'].get('content', '')
                    if vision_desc:
                        desc += f"IMAGE CONTENT: {vision_desc}. "
            except Exception as e:
                # If Llava is not available or an error occurs, continue without it
                # (e.g. model is not installed or API error)
                print(f"⚠️ Llava vision description failed: {e}")
        
        return desc if desc else "Image content analyzed (Visual Hex Only)."

    def save_memory(self, content, source="User", c_type="text", short_len=64):
        if c_type == "text":
            chunks = self._smart_chunking(content)
            if not chunks: return 0
            
            # Short-text bypass:
            # For short, information-dense texts (e.g. BEIR/SciFact, captions),
            # token compression hurts recall. We store a single full embedding instead.
            # Long texts still use Dragon compression for 16x memory savings.
            total_saved = 0
            for chunk in chunks:
                # Check token length using the text compressor's encoder
                token_emb = self.text_compressor.nlp.encode([chunk], output_value='token_embeddings', convert_to_tensor=True)
                # token_emb is a tensor with shape [seq_len, d_model] for single text
                if isinstance(token_emb, list):
                    T = token_emb[0].shape[0] if len(token_emb) > 0 and len(token_emb[0].shape) > 0 else 0
                else:
                    T = token_emb.shape[0] if len(token_emb.shape) > 0 else 0
                
                if T <= short_len:
                    # BYPASS: store a single sentence embedding (SBERT / text_model CLS)
                    vec = self.text_compressor.nlp.encode([chunk], convert_to_numpy=True, normalize_embeddings=True)
                    vec = F.normalize(torch.tensor(vec), p=2, dim=1)
                    self.memory_vectors.append(vec.cpu().numpy())
                    self.memory_texts.append(f"[{source}] {chunk}")
                    self.memory_types.append("text")
                    # Determine modality: image descriptions are detected by content markers
                    is_image_desc = "DETECTED TEXT" in chunk or "IMAGE CONTENT" in chunk
                    self.memory_modalities.append("image_desc" if is_image_desc else "text")
                    # Track source counts for balancing
                    self.source_counts[source] = self.source_counts.get(source, 0) + 1
                    total_saved += 1
                else:
                    # Existing Dragon compress path for long texts
                    res = self.text_compressor.compress(
                        [chunk],
                        adaptive=True,
                        sensitivity=self.compress_sensitivity,
                        min_vectors=self.compress_min_vectors
                    )
                    for r in res:
                        vec = F.normalize(r['compressed_vectors'], p=2, dim=1)
                        self.memory_vectors.append(vec.cpu().numpy())
                        self.memory_texts.append(f"[{source}] {chunk}")
                        self.memory_types.append("text")
                        # Determine modality: image descriptions are detected by content markers
                        is_image_desc = "DETECTED TEXT" in chunk or "IMAGE CONTENT" in chunk
                        self.memory_modalities.append("image_desc" if is_image_desc else "text")
                        # Track source counts for balancing
                        self.source_counts[source] = self.source_counts.get(source, 0) + 1
                        total_saved += 1
                    
                    # Hybrid anchor: add one SBERT global vector per chunk (optional anchor)
                    if self.hybrid_anchor:
                        try:
                            # one global SBERT vector per chunk (stabilization anchor)
                            gv = self.text_compressor.nlp.encode(
                                [chunk],
                                convert_to_numpy=True,
                                normalize_embeddings=True
                            )[0]  # (384,)
                            gv = gv[None, :]  # (1,384)
                            gv = F.normalize(torch.tensor(gv), p=2, dim=1)
                            
                            self.memory_vectors.append(gv.cpu().numpy())
                            self.memory_texts.append(f"[{source}] {chunk} (GLOBAL_ANCHOR)")
                            self.memory_types.append("text")
                            self.memory_modalities.append("text")
                            self.source_counts[source] = self.source_counts.get(source, 0) + 1
                        except Exception as e:
                            print(f"⚠️ Hybrid anchor encode failed: {e}")
            return total_saved
            
        elif c_type == "image" and self.has_vision:
            # 1. Visual Hex Vector (this accepts PIL or path)
            res = self.vision_compressor.compress_image(content)
            vec_vis = F.normalize(res['compressed_vectors'].squeeze(0), p=2, dim=1)
            self.memory_vectors.append(vec_vis.cpu().numpy())
            self.memory_texts.append(f"[{source}] 🖼️ [Visual Hex Data]")
            self.memory_types.append("image")
            self.memory_modalities.append("image_hex")
            # Track source counts for balancing
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
            
            # 2. Ensure PIL for OCR/LLaVA
            try:
                if isinstance(content, str):
                    img_pil = Image.open(content).convert("RGB")
                else:
                    img_pil = content
            except Exception as e:
                print(f"⚠️ Could not open image for OCR/LLaVA: {e}")
                img_pil = None
            
            # 3. Semantic Description
            desc = self.analyze_image(img_pil) if img_pil is not None else ""
            if desc:
                # We keep the same source id for image descriptions so that
                # vision benchmarks and UI link desc -> original image cleanly.
                # Modality is inferred from content markers (DETECTED TEXT / IMAGE CONTENT).
                self.save_memory(desc, source=source, c_type="text")
            return 2
        return 0

    def search_memory(self, query, top_k=4, visual_boost=False, return_raw_sims=False):
        """
        Search memory with optional raw similarity scores (without debias).
        
        Args:
            query: Search query
            top_k: Number of results to return
            visual_boost: Enable visual search
            return_raw_sims: If True, return (raw_sim, debias_sim, text, type) tuples
        
        Returns:
            List of (sim, text, type) tuples, or (raw_sim, debias_sim, text, type) if return_raw_sims=True
        """
        if not self.memory_vectors: return []
        
        # Modal router: separate indices by modality
        # TEXT index: only real book text (not image descriptions)
        # IMAGE_DESC index: image descriptions (OCR, LLaVA)
        # IMAGE_HEX index: visual hex vectors (for cross-modal)
        
        # Determine search scope based on query type
        # For text queries: search in text + image_desc (same embedding space)
        # For visual queries: can search in image_hex (but only if explicitly visual)
        search_text = True
        # For text queries, image descriptions live in the SAME text embedding space,
        # so they must be searchable by default.
        search_image_desc = True
        # Don't search image_hex for text queries (different embedding space, no alignment)
        search_image_hex = False  # Disabled until alignment is implemented
        
        # Text Search Vector
        # Keep query compression sensitivity consistent with memory build
        q_res = self.text_compressor.compress(
            [query],
            adaptive=True,
            sensitivity=self.compress_sensitivity,
            min_vectors=self.compress_min_vectors
        )[0]
        q_vec = F.normalize(q_res['compressed_vectors'], p=2, dim=1)
        
        results = []
        for i, mem_vec in enumerate(self.memory_vectors):
            # Modal filtering: skip based on modality
            modality = self.memory_modalities[i] if i < len(self.memory_modalities) else "text"
            
            if modality == "text" and not search_text:
                continue
            if modality == "image_desc" and not search_image_desc:
                continue
            if modality == "image_hex" and not search_image_hex:
                continue
            
            # Legacy filter for image types
            if self.memory_types[i] == "image" and not visual_boost:
                continue
            
            m_tensor = torch.tensor(mem_vec).to(q_vec.device)
            # Ensure L2 normalization for cosine similarity (safety check)
            m_tensor = F.normalize(m_tensor, p=2, dim=1)
            
            # Smooth max pooling (logsumexp) to avoid one-chunk lottery while keeping recall
            # Using matmul on normalized vectors = cosine similarity
            sims = torch.matmul(q_vec, m_tensor.T)  # [q_tokens, mem_chunks]
            tau = 0.10
            tok_sim = torch.logsumexp(sims / tau, dim=1) * tau  # [q_tokens]
            tok_sim_mean = tok_sim.mean().item()
            
            # Option B: Penalize OTHER sources (image descriptions, etc.) for text-only queries
            mem_text = self.memory_texts[i]
            raw_sim = tok_sim_mean  # Store raw similarity before any penalties
            
            if not visual_boost and "(Desc)" in mem_text:
                tok_sim_mean = tok_sim_mean - 0.08  # Penalty for image descriptions in text-only search
            
            # --- Source debias (soft, capped) ---
            src = mem_text.split("]")[0][1:] if mem_text.startswith("[") else ""
            debias_pen = 0.0
            if self.source_debias_lambda > 0 and src:
                n_src = self.source_counts.get(src, 1)
                # log1p is stable; formula: score' = score - lambda * log(1 + N_src)
                debias_pen = self.source_debias_lambda * math.log1p(n_src)
                debias_pen = min(debias_pen, 0.12)  # CAP to prevent killing signals
            
            sim_adj = tok_sim_mean - debias_pen
            # For return_raw_sims, return unclamped sim_adj; clamp only for ranking
            if return_raw_sims:
                sim = sim_adj  # Don't clamp, return raw adjusted score
            else:
                sim = max(0.0, sim_adj)  # Clamp for ranking
            
            # DEBUG (temporary)
            # if len(results) < 3:
            #     print(f"[DBG] src={src} raw={raw_sim:.3f} pen={debias_pen:.3f} adj={adj_sim:.3f}")
            
            # Logic Boosts
            if visual_boost and self.memory_types[i] == "image": sim += 0.2
            elif visual_boost and "Desc" in self.memory_texts[i]: sim += 0.15
            
            if return_raw_sims:
                # Return: (raw_sim, sim_adj, text, mtype, src) - sim_adj is unclamped for debug
                results.append((raw_sim, sim_adj, self.memory_texts[i], self.memory_types[i], src))
            else:
                # Return: (sim, text, mtype, src) - sim is clamped for ranking
                results.append((sim, self.memory_texts[i], self.memory_types[i], src))
            
        # Per-source top-k rerank (anti-dominance of large sources)
        if self.use_balanced_search and len(results) > top_k:
            # Take top_k_raw (e.g., 24) for grouping - larger candidate pool
            top_k_raw = min(24, len(results))
            # Pool by raw_sim (index 0) - ensures small sources enter candidate pool
            if return_raw_sims:
                # Pool by raw_sim (index 0)
                raw_sorted = sorted(results, key=lambda x: x[0], reverse=True)[:top_k_raw]
            else:
                # If not return_raw_sims, results format is (sim, text, mtype, src)
                raw_sorted = sorted(results, key=lambda x: x[0], reverse=True)[:top_k_raw]
            
            # Group by source
            by_src = {}
            if return_raw_sims:
                for raw_sim, sim, text, mtype, src in raw_sorted:
                    if src not in by_src:
                        by_src[src] = []
                    by_src[src].append((raw_sim, sim, text, mtype, src))
            else:
                for sim, text, mtype, src in raw_sorted:
                    if src not in by_src:
                        by_src[src] = []
                    by_src[src].append((sim, text, mtype, src))
            
            # Take top-m from each source (e.g., m=3)
            m_per_source = 3
            candidates = []
            for src_items in by_src.values():
                candidates.extend(src_items[:m_per_source])
            
            # Final rerank by debias score (sim) - sort_idx for final rerank
            sort_idx = 1 if return_raw_sims else 0
            final = sorted(candidates, key=lambda x: x[sort_idx], reverse=True)
            
            # Margin-abstain: check if top1-top2 margin is too small
            if len(final) >= 2:
                if return_raw_sims:
                    top1_score = final[0][1]  # debias sim
                    top2_score = final[1][1]  # debias sim
                else:
                    top1_score = final[0][0]  # debias sim
                    top2_score = final[1][0]  # debias sim
                margin = top1_score - top2_score
                
                if margin < 0.02:
                    # Low confidence: return top-2 with abstain indicator
                    if return_raw_sims:
                        return [(rs, s, t, mt) for rs, s, t, mt, _ in final[:2]]
                    else:
                        return [(s, t, mt) for s, t, mt, _ in final[:2]]
            
            # Remove src from tuple for compatibility
            if return_raw_sims:
                return [(rs, s, t, mt) for rs, s, t, mt, _ in final[:top_k]]
            else:
                return [(s, t, mt) for s, t, mt, _ in final[:top_k]]
        
        # Standard return (no balanced search or small result set)
        # Sort by debias sim (index 0 if not return_raw_sims, index 1 if return_raw_sims)
        sort_idx = 1 if return_raw_sims else 0
        sorted_results = sorted(results, key=lambda x: x[sort_idx], reverse=True)[:top_k]
        if return_raw_sims:
            return [(rs, s, t, mt) for rs, s, t, mt, _ in sorted_results]
        else:
            return [(s, t, mt) for s, t, mt, _ in sorted_results]

    def process_file(self, file_obj, name):
        if name in self.processed_files: return "Duplicate"
        if name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = Image.open(file_obj).convert('RGB')
            self.save_memory(img, source=name, c_type="image")
        else:
            text = file_obj.read().decode('utf-8', errors='ignore')
            self.save_memory(text, source=name, c_type="text")
        self.processed_files.add(name)
        return "Saved"

    def chat(self, user_input, image_input=None, visual_mode=False):
        context = ""
        rag_debug = []
        
        if image_input:
            self.save_memory(image_input, source="User_Upload", c_type="image")
            user_input += " [Image Uploaded]"
            visual_mode = True

        self.chat_history.append({"role": "user", "content": user_input})
        
        # RAG Search - use raw sims for threshold gate (aligned with calibration)
        matches = self.search_memory(user_input, top_k=5, visual_boost=visual_mode, return_raw_sims=True)
        
        # Dynamic threshold with margin gate for hard-negative filtering
        # Use raw sims for threshold comparison (aligned with neg_thr calibration)
        base_thr = 0.2 if visual_mode else 0.5  # Text default raised to 0.5 (was 0.4)
        thr = base_thr
        if visual_mode and self.visual_neg_thr is not None:
            thr = max(base_thr, self.visual_neg_thr)
        elif (not visual_mode) and self.neg_thr is not None:
            thr = max(base_thr, self.neg_thr - 0.02)
        
        # Margin gate: filter hard-negative FPs (low margin = scattered similarity)
        margin_min = 0.02 if visual_mode else 0.04
        margin_abstain = 0.02  # Threshold for abstain (uncertain response)
        strong = False
        uncertain = False
        
        if len(matches) >= 2:
            raw1, deb1, text1, type1 = matches[0]
            raw2, deb2, text2, type2 = matches[1]
            margin = raw1 - raw2  # Use raw sims for margin check
            strong = (raw1 >= thr) and (margin >= margin_min or raw1 >= thr + 0.08)
            # Abstain: if margin is too small, indicate uncertainty
            uncertain = (raw1 >= thr) and (margin < margin_abstain) and not strong
        elif len(matches) == 1:
            raw1, deb1, text1, type1 = matches[0]
            strong = raw1 >= thr + 0.05
        
        # Two-stage gate with abstain logic
        context = ""
        rag_debug = []
        if strong:
            # Filter by raw sim threshold
            valid_matches = [m for m in matches if m[0] >= thr]  # m[0] is raw_sim
            if valid_matches:
                context = "\n".join([f"SOURCE: {m[2]}" for m in valid_matches[:3]])  # m[2] is text
                rag_debug = valid_matches[:3]
        elif uncertain:
            # Margin too small: provide top-2 context and indicate uncertainty
            valid_matches = [m for m in matches if m[0] >= thr]  # m[0] is raw_sim
            if valid_matches:
                context = "\n".join([f"SOURCE: {m[2]}" for m in valid_matches[:2]])  # m[2] is text
                rag_debug = valid_matches[:2]
                # Add uncertainty indicator to system prompt
                context = f"[UNCERTAIN: Multiple similar matches found. Providing top-2 sources for context.]\n{context}"
            
        sys_prompt = "You are Dragon. Use the provided MEMORY SOURCES to answer. If no sources, chat naturally."
        if context: sys_prompt += f"\nMEMORY SOURCES:\n{context}"
        
        resp = self.llm.chat(self.chat_history[-6:], system_prompt=sys_prompt)
        self.chat_history.append({"role": "assistant", "content": resp['content'], "rag_info": rag_debug})
        
        return resp['content']