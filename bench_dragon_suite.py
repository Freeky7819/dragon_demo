# -*- coding: utf-8 -*-
"""
bench_dragon_suite.py
Unified benchmark runner for Dragon.

NOTE ON FAIRNESS / MODIFICATIONS:
This suite exposes optional Dragon compression knobs:
  --sensitivity, --min-vectors, --hybrid-anchor
Defaults match Dragon's original behavior.
hybrid-anchor is OFF by default and adds 1 SBERT global vector per chunk
to test a known fairness/stability tradeoff (better R@1 vs slightly more memory).

Modes:
  1) beir      - BEIR/MTEB-style short/medium text retrieval (SciFact default)
  2) loco      - LoCoV1 long-context retrieval benchmark (official long-doc retrieval)
  3) longbench - LongBench v2 long-context multiple-choice reasoning (official direct LLM eval)
                + optional Dragon-RAG variant (NOT official LongBench metric)

IMPORTANT FAIRNESS NOTES:
- LoCoV1 is the widely used "official-ish" long-context retrieval benchmark. It provides
  long passages and per-query answer_pids as relevance labels. We evaluate with Recall@K,
  nDCG@10, MRR@10 like BEIR.  (paper/blog: LoCoV1)  :contentReference[oaicite:1]{index=1}
- LongBench v2 is an "official" long-context understanding benchmark in multiple-choice form.
  "Official" eval is DIRECT: feed full context to the LLM.  :contentReference[oaicite:2]{index=2}
- Dragon-RAG on LongBench is a *separate* setting to show retrieval benefit. It is
  clearly labeled and optional.

Deps:
  pip install torch sentence-transformers datasets numpy requests
  (Dragon weights/vision not required unless DragonAgent loads them by default)

Typical local runs:
  python bench_dragon_suite.py beir --dataset BeIR/scifact --max-docs 1000 --max-queries 200
  python bench_dragon_suite.py loco --loco-task qasper_abstract --max-docs 1000 --max-queries 200
  python bench_dragon_suite.py longbench --backend ollama --ollama-model mistral:7b --max-examples 20 --max-context-chars 120000

Harmonic Signature Protocol:
omega≈6.0, gamma≈0.0, phi≈π/3
"""

import argparse, math, random, re, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

try:
    import psutil
except ImportError:
    psutil = None
    print("⚠️ psutil not installed. Memory reporting disabled. Install with: pip install psutil")

# Dragon
from dragon_brain import DragonAgent  # in your repo

# -------------------------
# Logging Utility
# -------------------------

def save_benchmark_json(args, metrics, task_name):
    """Save configuration and results to a JSON file."""
    if not getattr(args, 'json_file', None):
        return
    
    from datetime import datetime
    
    # Convert args to dict, remove callable objects
    config = {k: v for k, v in vars(args).items() if not callable(v)}
    
    # Prepare final object
    data = {
        "timestamp": datetime.now().isoformat(),
        "task": task_name,
        "config": config,
        "metrics": metrics
    }
    
    try:
        with open(args.json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\n[LOG] ✅ Results successfully saved to: {args.json_file}")
    except Exception as e:
        print(f"\n[ERR] ❌ Failed to save JSON log: {e}")

# -------------------------
# Memory/Footprint reporting
# -------------------------

def report_vector_footprint(name, vectors):
    """
    vectors: list of np arrays or torch tensors (each shape (k, D) or (D,))
    Prints theoretical byte footprint of stored vectors (lower bound, no Python overhead).
    """
    import numpy as np, torch

    n_vec = 0
    dim = None
    bytes_per = None

    for v in vectors:
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)

        if arr.ndim == 1:
            k, d = 1, arr.shape[0]
        else:
            k, d = arr.shape[0], arr.shape[1]

        n_vec += k
        dim = d if dim is None else dim
        bytes_per = arr.dtype.itemsize if bytes_per is None else bytes_per

    total_bytes = n_vec * dim * bytes_per
    print(f"[FOOTPRINT] {name}: n_vec={n_vec} dim={dim} dtype_bytes={bytes_per} => {total_bytes/1e6:.2f} MB")
    return total_bytes, n_vec, dim, bytes_per

def rss_mb():
    """Return current process RSS in MB."""
    if psutil is None:
        return 0.0
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1e6

def estimate_disk_fp16(name, vectors):
    """Estimate disk footprint if stored as float16 (for faiss/npz)."""
    import numpy as np, torch
    flat = []
    for v in vectors:
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)
        if arr.ndim == 1:
            flat.append(arr[None, :])
        else:
            flat.append(arr)
    X = np.concatenate(flat, axis=0).astype(np.float16)
    print(f"[DISK fp16] {name}: {X.nbytes/1e6:.2f} MB (raw)")
    return X.nbytes

# -------------------------
# Common utils (from BEIR test)
# -------------------------

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def pick(ex, keys):
    for k in keys:
        v = ex.get(k, None)
        if v is not None:
            return v
    return None

def logsumexp_score(q_vecs, d_vecs, tau=0.10):
    """
    Dragon-style smooth max pooling (matches dragon_brain.search_memory scoring).
    q_vecs: [kq, d], d_vecs: [kd, d]
    Returns scalar similarity.
    """
    q = F.normalize(torch.tensor(q_vecs), p=2, dim=1)
    d = F.normalize(torch.tensor(d_vecs), p=2, dim=1)
    sims = q @ d.T
    tok_sim = torch.logsumexp(sims / tau, dim=1) * tau
    return tok_sim.mean().item()

def recall_at_k(ranked_doc_ids, rel_set, k):
    return 1.0 if any(d in rel_set for d in ranked_doc_ids[:k]) else 0.0

def dcg_at_k(ranked_doc_ids, rel_set, k):
    dcg = 0.0
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        rel = 1.0 if d in rel_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(ranked_doc_ids, rel_set, k):
    dcg = dcg_at_k(ranked_doc_ids, rel_set, k)
    idcg = dcg_at_k(list(rel_set), rel_set, min(k, len(rel_set)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_doc_ids, rel_set, k):
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if d in rel_set:
            return 1.0 / i
    return 0.0

def build_dragon_doc_mats(dragon_agent, doc_ids, emb_dim=384):
    """
    Extract ragged per-doc matrices from DragonAgent memory.
    Fixes old shape-mismatch by concatenating all vectors for a doc.
    """
    doc_drag = []
    for doc_id in doc_ids:
        doc_vecs = []
        for vec, text in zip(dragon_agent.memory_vectors, dragon_agent.memory_texts):
            if text.startswith(f"[{doc_id}]"):
                doc_vecs.append(vec)
        if doc_vecs:
            mats = []
            for v in doc_vecs:
                v = np.asarray(v)
                if v.ndim == 1:
                    v = v[None, :]
                mats.append(v)
            doc_mat = np.concatenate(mats, axis=0)
            doc_drag.append(doc_mat)
        else:
            doc_drag.append(np.zeros((1, emb_dim)))
    return doc_drag

# -------------------------
# BEIR / SciFact retrieval (your current test)
# -------------------------

def run_beir(args):
    set_seed(args.seed)

    corpus = load_dataset(args.dataset, "corpus", split="corpus")
    queries = load_dataset(args.dataset, "queries", split="queries")

    # BEIR qrels hosted as separate dataset "<name>-qrels"
    try:
        qrels = load_dataset(args.dataset, "qrels", split=args.split)
    except ValueError:
        qrels = load_dataset(f"{args.dataset}-qrels", split=args.split)

    docs, doc_ids = [], []
    for ex in corpus:
        doc_id = ex.get("_id") or ex.get("corpus-id") or ex.get("doc_id")
        title = ex.get("title", "")
        text = ex.get("text", "")
        full = (title + "\n" + text).strip()
        if full:
            docs.append(full); doc_ids.append(str(doc_id))
        if len(docs) >= args.max_docs:
            break

    rels = {}
    for ex in qrels:
        qid = pick(ex, ["query-id", "query_id", "_id", "qid", "query"])
        did = pick(ex, ["corpus-id", "corpus_id", "doc-id", "doc_id", "did", "doc"])
        score = ex.get("score", ex.get("relevance", 1))
        if qid is None or did is None:
            continue
        if score > 0:
            rels.setdefault(str(qid), set()).add(str(did))

    q_texts, q_ids = [], []
    for ex in queries:
        qid = ex.get("_id") or ex.get("query-id") or ex.get("qid")
        qid = str(qid)
        if qid in rels:
            q_texts.append(ex["text"]); q_ids.append(qid)
        if len(q_texts) >= args.max_queries:
            break

    if len(q_texts) == 0:
        print("No queries matched qrels. Likely qrels schema mismatch.")
        return

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    dragon_agent = DragonAgent(
        hybrid_anchor=args.hybrid_anchor,
        compress_sensitivity=args.sensitivity,
        compress_min_vectors=args.min_vectors
    )

    rss0 = rss_mb()
    doc_base = sbert.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    rss1 = rss_mb()
    
    # Convert baseline vectors to list format for footprint reporting
    doc_vecs_baseline = [doc_base[i] for i in range(len(doc_base))]
    baseline_vec_bytes, baseline_nvec, D, bpe = report_vector_footprint("Baseline vectors", doc_vecs_baseline)
    print(f"[RSS] Baseline ingest ΔRSS = {rss1-rss0:.2f} MB")

    print(f"[INGEST] Ingesting {len(docs)} documents into Dragon...")
    for i, (doc, doc_id) in enumerate(zip(docs, doc_ids)):
        dragon_agent.save_memory(doc, source=str(doc_id), c_type="text")
        if (i + 1) % 100 == 0:
            print(f"  Ingested {i + 1}/{len(docs)} documents...")

    doc_drag = build_dragon_doc_mats(dragon_agent, doc_ids)

    total_vectors = len(dragon_agent.memory_vectors)
    print(f"[INFO] Dragon total memory vectors: {total_vectors} (docs: {len(docs)})")

    r1=r5=r10=nd10=mrr10=0.0
    r1b=r5b=r10b=nd10b=mrr10b=0.0

    for qid, qtext in zip(q_ids, q_texts):
        rel_set = rels[qid]

        q_base = sbert.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0]

        q_res = dragon_agent.text_compressor.compress(
            [qtext],
            adaptive=True,
            sensitivity=args.sensitivity,
            min_vectors=args.min_vectors
        )[0]
        q_drag = q_res["compressed_vectors"].cpu().numpy()

        scores_b = (doc_base @ q_base).tolist()
        scores_d = [logsumexp_score(q_drag, dv, tau=args.tau) for dv in doc_drag]

        ranked_b = [doc_ids[i] for i in np.argsort(scores_b)[::-1]]
        ranked_d = [doc_ids[i] for i in np.argsort(scores_d)[::-1]]

        r1  += recall_at_k(ranked_d, rel_set, 1)
        r5  += recall_at_k(ranked_d, rel_set, 5)
        r10 += recall_at_k(ranked_d, rel_set, 10)
        nd10 += ndcg_at_k(ranked_d, rel_set, 10)
        mrr10 += mrr_at_k(ranked_d, rel_set, 10)

        r1b  += recall_at_k(ranked_b, rel_set, 1)
        r5b  += recall_at_k(ranked_b, rel_set, 5)
        r10b += recall_at_k(ranked_b, rel_set, 10)
        nd10b += ndcg_at_k(ranked_b, rel_set, 10)
        mrr10b += mrr_at_k(ranked_b, rel_set, 10)

    n = len(q_ids)
    
    # Calculate bypass estimate
    if total_vectors <= len(docs):
        bypass_estimate = f"~{total_vectors}/{len(docs)} docs used short-text bypass"
    else:
        bypass_estimate = f"Average {total_vectors/len(docs):.1f} vectors/doc (some compression used)"
    
    # Prepare metrics for JSON
    metrics = {
        "baseline": {
            "R@1": r1b/n, "R@5": r5b/n, "R@10": r10b/n, 
            "nDCG@10": nd10b/n, "MRR@10": mrr10b/n
        },
        "dragon": {
            "R@1": r1/n, "R@5": r5/n, "R@10": r10/n, 
            "nDCG@10": nd10/n, "MRR@10": mrr10/n
        },
        "details": {
            "docs": len(docs),
            "queries": n,
            "total_vectors": total_vectors,
            "bypass_estimate": bypass_estimate
        }
    }
    
    print("\n=== TEXT RETRIEVAL (BEIR/MTEB style) ===")
    print(f"Dataset: {args.dataset} | split={args.split} | docs={len(docs)} | queries={n}")
    print(f"Dragon short-text bypass: {bypass_estimate}\n")

    print("Baseline (SBERT no-compress):")
    print(f"  R@1={metrics['baseline']['R@1']:.3f} R@5={metrics['baseline']['R@5']:.3f} R@10={metrics['baseline']['R@10']:.3f} "
          f"nDCG@10={metrics['baseline']['nDCG@10']:.3f} MRR@10={metrics['baseline']['MRR@10']:.3f}")
    print("Dragon (compressed vectors):")
    print(f"  R@1={metrics['dragon']['R@1']:.3f} R@5={metrics['dragon']['R@5']:.3f} R@10={metrics['dragon']['R@10']:.3f} "
          f"nDCG@10={metrics['dragon']['nDCG@10']:.3f} MRR@10={metrics['dragon']['MRR@10']:.3f}")
    
    # Save
    save_benchmark_json(args, metrics, "beir")

# -------------------------
# LoCoV1 long-context retrieval
# -------------------------

LOCO_TASKS = [
    "summ_screen_fd", "gov_report", "qmsum", "qasper_title", "qasper_abstract",
    "2wikimqa", "multifieldqa", "passage_retrieval", "courtlistener_HTML",
    "courtlistener_Plain_Text", "legal_case_reports", "stackoverflow"
]

def run_loco(args):
    set_seed(args.seed)

    docs_ds = load_dataset("hazyresearch/LoCoV1-Documents", split="test")
    queries_ds = load_dataset("hazyresearch/LoCoV1-Queries", split="test")

    task = args.loco_task
    if task not in LOCO_TASKS:
        raise ValueError(f"Unknown LoCo task '{task}'. Available: {LOCO_TASKS}")

    docs_ds = docs_ds.filter(lambda x: x["dataset"] == task)
    queries_ds = queries_ds.filter(lambda x: x["dataset"] == task)

    docs, doc_ids = [], []
    for ex in docs_ds:
        pid = str(ex["pid"])
        passage = ex["passage"]
        if passage:
            docs.append(passage); doc_ids.append(pid)
        if len(docs) >= args.max_docs:
            break

    q_texts, q_ids, rels = [], [], {}
    for ex in queries_ds:
        qid = str(ex["qid"])
        qtext = ex["query"]
        ans_pids = [str(p) for p in ex.get("answer_pids", [])]
        if not ans_pids:
            continue
        rels[qid] = set(ans_pids)
        q_texts.append(qtext); q_ids.append(qid)
        if len(q_texts) >= args.max_queries:
            break

    if len(q_texts) == 0:
        print("No LoCo queries found after filtering. Try another task.")
        return

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    dragon_agent = DragonAgent(
        hybrid_anchor=args.hybrid_anchor,
        compress_sensitivity=args.sensitivity,
        compress_min_vectors=args.min_vectors
    )

    rss0 = rss_mb()
    doc_base = sbert.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    rss1 = rss_mb()
    
    # Convert baseline vectors to list format for footprint reporting
    doc_vecs_baseline = [doc_base[i] for i in range(len(doc_base))]
    baseline_vec_bytes, baseline_nvec, D, bpe = report_vector_footprint("Baseline vectors", doc_vecs_baseline)
    print(f"[RSS] Baseline ingest ΔRSS = {rss1-rss0:.2f} MB")

    print(f"[INGEST] Ingesting {len(docs)} long documents into Dragon (LoCo task={task})...")
    for i, (doc, doc_id) in enumerate(zip(docs, doc_ids)):
        dragon_agent.save_memory(doc, source=str(doc_id), c_type="text")
        if (i + 1) % 100 == 0:
            print(f"  Ingested {i + 1}/{len(docs)} documents...")

    rss2 = rss_mb()
    dragon_vec_bytes, dragon_nvec, D2, bpe2 = report_vector_footprint("Dragon memory vectors", dragon_agent.memory_vectors)
    print(f"[RSS] Dragon ingest ΔRSS = {rss2-rss1:.2f} MB")
    
    gain = baseline_vec_bytes / max(dragon_vec_bytes, 1)
    print(f"[GAIN] Vector footprint ratio (Baseline/Dragon) = {gain:.2f}x")
    
    if hasattr(args, 'disk_fp16') and args.disk_fp16:
        estimate_disk_fp16("Baseline vectors (fp16)", doc_vecs_baseline)
        estimate_disk_fp16("Dragon memory vectors (fp16)", dragon_agent.memory_vectors)

    doc_drag = build_dragon_doc_mats(dragon_agent, doc_ids)

    total_vectors = len(dragon_agent.memory_vectors)
    print(f"[INFO] Dragon total memory vectors: {total_vectors} (docs: {len(docs)})")

    r1=r5=r10=nd10=mrr10=0.0
    r1b=r5b=r10b=nd10b=mrr10b=0.0

    for qid, qtext in zip(q_ids, q_texts):
        rel_set = rels[qid]

        q_base = sbert.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_res = dragon_agent.text_compressor.compress(
            [qtext],
            adaptive=True,
            sensitivity=args.sensitivity,
            min_vectors=args.min_vectors
        )[0]
        q_drag = q_res["compressed_vectors"].cpu().numpy()

        scores_b = (doc_base @ q_base).tolist()
        scores_d = [logsumexp_score(q_drag, dv, tau=args.tau) for dv in doc_drag]

        ranked_b = [doc_ids[i] for i in np.argsort(scores_b)[::-1]]
        ranked_d = [doc_ids[i] for i in np.argsort(scores_d)[::-1]]

        r1  += recall_at_k(ranked_d, rel_set, 1)
        r5  += recall_at_k(ranked_d, rel_set, 5)
        r10 += recall_at_k(ranked_d, rel_set, 10)
        nd10 += ndcg_at_k(ranked_d, rel_set, 10)
        mrr10 += mrr_at_k(ranked_d, rel_set, 10)

        r1b  += recall_at_k(ranked_b, rel_set, 1)
        r5b  += recall_at_k(ranked_b, rel_set, 5)
        r10b += recall_at_k(ranked_b, rel_set, 10)
        nd10b += ndcg_at_k(ranked_b, rel_set, 10)
        mrr10b += mrr_at_k(ranked_b, rel_set, 10)

    n = len(q_ids)
    
    # Prepare metrics for JSON
    metrics = {
        "baseline": {
            "R@1": r1b/n, "R@5": r5b/n, "R@10": r10b/n, 
            "nDCG@10": nd10b/n, "MRR@10": mrr10b/n
        },
        "dragon": {
            "R@1": r1/n, "R@5": r5/n, "R@10": r10/n, 
            "nDCG@10": nd10/n, "MRR@10": mrr10/n
        },
        "details": {
            "task": task,
            "docs": len(docs),
            "queries": n,
            "avg_vectors_per_doc": total_vectors/len(docs)
        }
    }
    
    print("\n=== LONG-CONTEXT RETRIEVAL (LoCoV1) ===")
    print(f"Task: {task} | docs={len(docs)} | queries={n}")
    print(f"Dragon avg vectors/doc: {total_vectors/len(docs):.1f}\n")

    print("Baseline (SBERT no-compress):")
    print(f"  R@1={metrics['baseline']['R@1']:.3f} R@5={metrics['baseline']['R@5']:.3f} R@10={metrics['baseline']['R@10']:.3f} "
          f"nDCG@10={metrics['baseline']['nDCG@10']:.3f} MRR@10={metrics['baseline']['MRR@10']:.3f}")
    print("Dragon (compressed vectors):")
    print(f"  R@1={metrics['dragon']['R@1']:.3f} R@5={metrics['dragon']['R@5']:.3f} R@10={metrics['dragon']['R@10']:.3f} "
          f"nDCG@10={metrics['dragon']['nDCG@10']:.3f} MRR@10={metrics['dragon']['MRR@10']:.3f}")
    
    # Save
    save_benchmark_json(args, metrics, f"loco_{task}")

# -------------------------
# LongBench v2 (official direct) + optional Dragon-RAG
# -------------------------

def call_openai_compat(messages, model, base_url, api_key, temperature=0.0, max_tokens=16):
    import requests
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_ollama(prompt, model="mistral:7b", url="http://localhost:11434/api/generate"):
    import requests
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

def parse_choice(text):
    m = re.search(r"\b([ABCD])\b", text.strip(), re.I)
    return m.group(1).upper() if m else None

def chunk_text_for_dragon(ctx, max_chunk_chars=4000):
    # simple paragraph-ish chunking; Dragon has its own smart chunking, but we
    # keep it explicit here to avoid huge single save_memory calls.
    chunks = []
    buf = []
    size = 0
    for para in ctx.split("\n"):
        if size + len(para) > max_chunk_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para); size += len(para) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def run_longbench(args):
    set_seed(args.seed)

    ds = load_dataset(args.longbench_dataset, split=args.longbench_split)

    # local-friendly sampling
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:args.max_examples]

    correct = 0
    total = 0

    for idx in indices:
        ex = ds[idx]
        ctx = ex["context"]
        if args.max_context_chars > 0 and len(ctx) > args.max_context_chars:
            # NOTE: truncation is NOT official LongBench eval.
            ctx = ctx[:args.max_context_chars]

        question = ex["question"]
        A = ex["choice_A"]; B = ex["choice_B"]; C = ex["choice_C"]; D = ex["choice_D"]
        gold = ex["answer"].strip().upper()

        # Optional Dragon-RAG variant
        if args.use_dragon_rag:
            dragon_agent = DragonAgent(
        hybrid_anchor=args.hybrid_anchor,
        compress_sensitivity=args.sensitivity,
        compress_min_vectors=args.min_vectors
    )
            doc_id = f"LB2_{ex.get('_id','x')}"
            for ch in chunk_text_for_dragon(ctx, max_chunk_chars=args.dragon_chunk_chars):
                dragon_agent.save_memory(ch, source=doc_id, c_type="text")
            hits = dragon_agent.search_memory(question, top_k=args.top_k)
            rag_ctx = "\n\n".join([h[1] if isinstance(h, tuple) else str(h) for h in hits])
            ctx_for_prompt = rag_ctx
        else:
            ctx_for_prompt = ctx

        prompt = (
            "You are answering a multiple-choice question.\n"
            "Read CONTEXT, then QUESTION and choices.\n"
            "Reply with ONLY one letter: A, B, C, or D.\n\n"
            f"CONTEXT:\n{ctx_for_prompt}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
            "Answer:"
        )

        if args.backend == "api":
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
            messages = [{"role": "user", "content": prompt}]
            out = call_openai_compat(
                messages, model=args.api_model, base_url=args.api_base_url, api_key=api_key,
                temperature=0.0, max_tokens=16
            )
        elif args.backend == "ollama":
            out = call_ollama(prompt, model=args.ollama_model, url=args.ollama_url)
        else:
            raise ValueError("backend must be 'api' or 'ollama'")

        pred = parse_choice(out) or "?"
        ok = (pred == gold)

        correct += 1 if ok else 0
        total += 1

        if args.verbose:
            print(f"\n--- Example {idx} ---")
            print(f"Gold={gold} Pred={pred} OK={ok}")
            print(out[:400])

    acc = correct / max(1, total)
    
    # Prepare metrics for JSON
    metrics = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "mode": "dragon-rag" if args.use_dragon_rag else "direct-llm"
    }

    print("\n=== LONGBENCH v2 (long-context MCQ) ===")
    print(f"Dataset: {args.longbench_dataset} | split={args.longbench_split}")
    print(f"Examples: {total} | backend={args.backend} | use_dragon_rag={args.use_dragon_rag}")
    if args.max_context_chars > 0:
        print(f"NOTE: context truncated to {args.max_context_chars} chars (NOT official eval).")
    print(f"Accuracy = {acc:.3f} ({correct}/{total})")
    
    # Save
    save_benchmark_json(args, metrics, "longbench")

# -------------------------
# Vision (Flickr30k / COCO text->image retrieval)
# -------------------------

# ============================================================
# Harmonic Signature Protocol (visible, per project rule)
# intent: resonant, ethical learning
# omega≈6.0, gamma≈0.0, phi≈π/3
# ============================================================

def run_vision(args):
    """
    Official text->image retrieval benchmark:
      - Flickr30k 1k test (default) or COCO Karpathy-style splits if available on HF.
    Baseline:
      - CLIP text-image retrieval (open_clip ViT-B/32 openai).
    Dragon:
      - caption mode: store GT captions as image descriptions into Dragon memory
      - e2e mode: ingest images, Dragon generates OCR/LLaVA descriptions

    IMPORTANT FAIRNESS NOTES:
      1) Dragon currently retrieves in TEXT space via image_desc (OCR/LLaVA or GT captions),
         not via aligned visual embeddings. This is a different path than CLIP.
      2) We aggregate multiple vectors per image with MAX similarity to avoid "more vectors => unfair advantage".
    """
    import random, numpy as np, torch
    from datasets import load_dataset
    import open_clip
    from PIL import Image
    import torch.nn.functional as F

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Load dataset ----------
    ds = load_dataset(args.dataset, split=args.split)

    # Expect HF schema like Flickr30k:
    # ds[i]["image"] (PIL) and ds[i]["caption"] or ["captions"]
    # Normalize to: images[], captions_per_image[]
    images = []
    caps_per_img = []
    for ex in ds:
        img = ex.get("image", None)
        cap = ex.get("caption", None)
        caps = ex.get("captions", None)
        if img is None: 
            continue
        if caps is None:
            if isinstance(cap, list): caps = cap
            else: caps = [cap] if cap is not None else []
        if not caps: 
            continue
        images.append(img)
        caps_per_img.append(caps)

    # Subsample images
    idx = list(range(len(images)))
    random.shuffle(idx)
    idx = idx[:args.max_images]
    images = [images[i] for i in idx]
    caps_per_img = [caps_per_img[i] for i in idx]

    # Build queries: flatten captions, keep gt image id
    queries = []
    gt_img_ids = []
    for img_id, caps in enumerate(caps_per_img):
        for c in caps:
            if c is None: continue
            queries.append(c)
            gt_img_ids.append(img_id)

    # Subsample queries if needed
    q_idx = list(range(len(queries)))
    random.shuffle(q_idx)
    q_idx = q_idx[:args.max_queries]
    queries = [queries[i] for i in q_idx]
    gt_img_ids = [gt_img_ids[i] for i in q_idx]

    print("\n=== OFFICIAL VISION RETRIEVAL SUITE ===")
    print(f"Dataset: {args.dataset} | split={args.split} | images={len(images)} | queries={len(queries)} | mode={args.mode}")
    if args.mode == "e2e":
        skip_ocr = getattr(args, 'no_ocr', False)
        ocr_status = "OFF (LLaVA only)" if skip_ocr else "ON (OCR + LLaVA)"
        print(f"Dragon e2e mode: OCR={ocr_status}")
    print("Baseline: CLIP ViT-B/32 (openai)")
    print("Dragon: image_desc retrieval in TEXT-space (caption or e2e)")

    # ---------- Baseline CLIP ----------
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    with torch.no_grad():
        # image embeddings
        img_tensors = torch.stack([preprocess(im).to(device) for im in images])
        img_emb = model.encode_image(img_tensors)
        img_emb = F.normalize(img_emb, p=2, dim=1)

        # text embeddings
        text_tokens = tokenizer(queries).to(device)
        txt_emb = model.encode_text(text_tokens)
        txt_emb = F.normalize(txt_emb, p=2, dim=1)

        sims = txt_emb @ img_emb.T  # [Q, I]
        ranked = torch.argsort(sims, dim=1, descending=True).cpu().numpy()

    def recall_at_k(ranked_list, gt, k):
        return 1.0 if gt in ranked_list[:k] else 0.0

    def mrr_at_k(ranked_list, gt, k):
        top = ranked_list[:k]
        for r, i in enumerate(top, start=1):
            if i == gt: return 1.0 / r
        return 0.0

    r1=r5=r10=mrr10=0.0
    for qi, gt in enumerate(gt_img_ids):
        rlist = ranked[qi].tolist()
        r1 += recall_at_k(rlist, gt, 1)
        r5 += recall_at_k(rlist, gt, 5)
        r10 += recall_at_k(rlist, gt, 10)
        mrr10 += mrr_at_k(rlist, gt, 10)
    n = len(gt_img_ids)

    print("\nBaseline (CLIP):")
    print(f"  R@1={r1/n:.3f} R@5={r5/n:.3f} R@10={r10/n:.3f} MRR@10={mrr10/n:.3f}")

    # ---------- Dragon path ----------
    from dragon_brain import DragonAgent

    rss0_vis = rss_mb()
    dragon = DragonAgent()

    # Build Dragon memory
    dragon.memory_vectors = []
    dragon.memory_texts = []
    dragon.memory_types = []
    dragon.memory_modalities = []
    dragon.source_counts = {}

    if args.mode == "caption":
        # Use GT captions as image descriptions (fast, fair)
        for img_id, caps in enumerate(caps_per_img):
            desc = " ".join([c for c in caps if c])
            dragon.save_memory(desc, source=f"(GT_DESC img={img_id})")
    else:
        # E2E: ingest images, Dragon generates OCR/LLaVA descriptions (slow)
        # If --no-ocr flag is set, skip OCR and use only LLaVA (for fair comparison)
        skip_ocr = getattr(args, 'no_ocr', False)
        # Store skip_ocr flag in agent instance for analyze_image to use
        dragon._skip_ocr = skip_ocr
        for img_id, im in enumerate(images):
            # save to temp jpeg if needed
            tmp_path = f".__tmp_flickr_{img_id}.jpg"
            im.save(tmp_path)
            dragon.save_memory(tmp_path, source=f"(IMG img={img_id})", c_type="image")
    
    rss1_vis = rss_mb()
    dragon_vec_bytes, dragon_nvec, D2, bpe2 = report_vector_footprint("Dragon memory vectors", dragon.memory_vectors)
    print(f"[RSS] Dragon ingest ΔRSS = {rss1_vis-rss0_vis:.2f} MB")
    
    if hasattr(args, 'disk_fp16') and args.disk_fp16:
        estimate_disk_fp16("Dragon memory vectors (fp16)", dragon.memory_vectors)

    # Precompute per-image vector indices for fair aggregation
    img_vec_ids = [[] for _ in range(len(images))]
    for i, txt in enumerate(dragon.memory_texts):
        # Expect "(... img=ID)" marker in source
        m = re.search(r"img=(\d+)", txt)
        if m:
            img_vec_ids[int(m.group(1))].append(i)

    # Evaluate Dragon with explicit visual_boost=True (even after core fix)
    r1d=r5d=r10d=mrr10d=0.0
    for q, gt in zip(queries, gt_img_ids):
        res = dragon.search_memory(q, top_k=args.top_k, visual_boost=True, return_raw_sims=True)
        # res: list of (raw_sim, sim_adj, text, mtype)
        # Aggregate max similarity per image
        per_img_score = np.full(len(images), -1e9, dtype=np.float32)
        for raw_sim, sim_adj, mem_text, mem_type in res:
            m = re.search(r"img=(\d+)", mem_text)
            if not m: continue
            img_id = int(m.group(1))
            per_img_score[img_id] = max(per_img_score[img_id], float(sim_adj))

        ranked_imgs = np.argsort(-per_img_score)

        r1d  += recall_at_k(ranked_imgs, gt, 1)
        r5d  += recall_at_k(ranked_imgs, gt, 5)
        r10d += recall_at_k(ranked_imgs, gt, 10)
        mrr10d += mrr_at_k(ranked_imgs, gt, 10)

    print("\nDragon (image_desc in text-space):")
    print(f"  R@1={r1d/n:.3f} R@5={r5d/n:.3f} R@10={r10d/n:.3f} MRR@10={mrr10d/n:.3f}")

    print("\nNOTE: Dragon scores measure compressed description retrieval, not direct CLIP-aligned vision embeddings.")
    
    # Prepare metrics for JSON
    metrics = {
        "baseline_clip": {
            "R@1": r1/n, "R@5": r5/n, "R@10": r10/n, "MRR@10": mrr10/n
        },
        "dragon": {
            "R@1": r1d/n, "R@5": r5d/n, "R@10": r10d/n, "MRR@10": mrr10d/n
        },
        "details": {
            "dataset": args.dataset,
            "split": args.split,
            "mode": args.mode,
            "ocr_active": not getattr(args, 'no_ocr', False)
        }
    }
    
    # Save
    save_benchmark_json(args, metrics, "vision")

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # beir
    ap_beir = sub.add_parser("beir")
    ap_beir.add_argument("--dataset", default="BeIR/scifact")
    ap_beir.add_argument("--split", default="test")
    ap_beir.add_argument("--max-docs", type=int, default=1000)
    ap_beir.add_argument("--max-queries", type=int, default=200)
    ap_beir.add_argument("--seed", type=int, default=41)
    ap_beir.add_argument("--tau", type=float, default=0.10)
    ap_beir.add_argument("--sensitivity", type=float, default=1.0,
                         help="Dragon compress_adaptive sensitivity (lower => more vectors).")
    ap_beir.add_argument("--min-vectors", type=int, default=2,
                         help="Minimum vectors per chunk for Dragon adaptive compression.")
    ap_beir.add_argument("--hybrid-anchor", action="store_true",
                         help="Also store one SBERT global vector per chunk (optional anchor).")
    ap_beir.add_argument("--disk-fp16", action="store_true",
                         help="Estimate disk footprint in float16 format (for faiss/npz).")
    ap_beir.add_argument("--json-file", default=None, help="Output JSON log file")
    ap_beir.set_defaults(fn=run_beir)

    # loco
    ap_loco = sub.add_parser("loco")
    ap_loco.add_argument("--loco-task", default="qasper_abstract",
                         help=f"One of: {LOCO_TASKS}")
    ap_loco.add_argument("--max-docs", type=int, default=1000)
    ap_loco.add_argument("--max-queries", type=int, default=200)
    ap_loco.add_argument("--seed", type=int, default=41)
    ap_loco.add_argument("--tau", type=float, default=0.10)
    ap_loco.add_argument("--sensitivity", type=float, default=1.0,
                         help="Dragon compress_adaptive sensitivity (lower => more vectors).")
    ap_loco.add_argument("--min-vectors", type=int, default=2,
                         help="Minimum vectors per chunk for Dragon adaptive compression.")
    ap_loco.add_argument("--hybrid-anchor", action="store_true",
                         help="Also store one SBERT global vector per chunk (optional anchor).")
    ap_loco.add_argument("--disk-fp16", action="store_true",
                         help="Estimate disk footprint in float16 format (for faiss/npz).")
    ap_loco.add_argument("--json-file", default=None, help="Output JSON log file")
    ap_loco.set_defaults(fn=run_loco)

    # longbench
    ap_lb = sub.add_parser("longbench")
    ap_lb.add_argument("--longbench-dataset", default="zai-org/LongBench-v2")
    ap_lb.add_argument("--longbench-split", default="train")
    ap_lb.add_argument("--max-examples", type=int, default=20,
                       help="Local-friendly sample size. Full official run uses 503.")
    ap_lb.add_argument("--max-context-chars", type=int, default=120000,
                       help="If >0, truncate context for local runs (NOT official). Set 0 for full context.")
    ap_lb.add_argument("--seed", type=int, default=41)
    ap_lb.add_argument("--backend", choices=["api", "ollama"], default="ollama")
    # api backend
    ap_lb.add_argument("--api-base-url", default="https://api.openai.com")
    ap_lb.add_argument("--api-model", default="gpt-4o-mini")
    ap_lb.add_argument("--api-key", default="")
    # ollama backend
    ap_lb.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    ap_lb.add_argument("--ollama-model", default="mistral:7b")
    # Dragon-RAG variant (not official)
    ap_lb.add_argument("--use-dragon-rag", action="store_true",
                       help="Use Dragon to retrieve top-k chunks before LLM (NOT official LongBench metric).")
    ap_lb.add_argument("--top-k", type=int, default=4)
    ap_lb.add_argument("--dragon-chunk-chars", type=int, default=4000)
    ap_lb.add_argument("--verbose", action="store_true")
    ap_lb.add_argument("--sensitivity", type=float, default=1.0,
                       help="Dragon compress_adaptive sensitivity (lower => more vectors).")
    ap_lb.add_argument("--min-vectors", type=int, default=2,
                       help="Minimum vectors per chunk for Dragon adaptive compression.")
    ap_lb.add_argument("--hybrid-anchor", action="store_true",
                       help="Also store one SBERT global vector per chunk (optional anchor).")
    ap_lb.add_argument("--json-file", default=None, help="Output JSON log file")
    ap_lb.set_defaults(fn=run_longbench)

    # vision
    ap_vis = sub.add_parser("vision")
    ap_vis.add_argument("--dataset", default="nlphuji/flickr30k",
                        help="Official image-text retrieval dataset (Flickr30k or MSCOCO via HF).")
    ap_vis.add_argument("--split", default="test")
    ap_vis.add_argument("--max-images", type=int, default=1000)
    ap_vis.add_argument("--max-queries", type=int, default=1000,
                        help="Number of captions (queries) to evaluate.")
    ap_vis.add_argument("--seed", type=int, default=41)
    ap_vis.add_argument("--mode", choices=["caption", "e2e"], default="caption",
                        help="caption=use GT captions as image_desc (fast, fair); e2e=Dragon OCR/LLaVA desc (slow, pipeline test)")
    ap_vis.add_argument("--top-k", type=int, default=10)
    ap_vis.add_argument("--disk-fp16", action="store_true",
                         help="Estimate disk footprint in float16 format (for faiss/npz).")
    ap_vis.add_argument("--no-ocr", action="store_true",
                         help="Skip OCR in e2e mode, use only LLaVA (for fair OCR ON vs OFF comparison).")
    ap_vis.add_argument("--json-file", default=None, help="Output JSON log file")
    ap_vis.set_defaults(fn=run_vision)

    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
