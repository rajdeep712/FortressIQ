"""Local retrieval benchmark harness with a full run log.

Runs the *real* LangGraph RAG pipeline (the same graph the chat API uses)
for a given ``user_id`` and optional ``selected_doc_ids``, inside a real
persistent LangGraph session (Postgres checkpointer keyed on
``thread_id = chat_id``).

Every run writes an exhaustive log to
``backend/bench_logs/bench_retrieval_<timestamp>.log`` containing, per turn:

  * the Qdrant **dense** and **BM25 sparse** retrieval arms (which points,
    with what scores) BEFORE Qdrant fusion
  * the Qdrant dense+BM25 fused result
  * the conversation hits with their cosine scores
  * the graph's second RRF fusion (conversation x documents) in ORDER with
    the RRF score of every item
  * the points selected after reranking, with rerank scores
  * the exact **system + user prompt** handed to the final LLM (after the
    child-chunk parents have been folded into the context)

The console shows the compact summary; the file has the full detail.

Config
------
Edit the ``BENCH_*`` constants at the top of this file. CLI flags
(``--user-id/--doc-ids/--queries``) take precedence when given.

Usage
-----
    python scripts/bench_retrieval.py [--user-id <UUID>] \
        [--doc-ids doc_x,doc_y] [--chat-id <hex>] [--top-k 5] \
        [--queries "first q" "follow-up"] [--no-llm] [--list-docs]

With no ``--queries`` and no ``BENCH_QUERIES`` the harness drops into an
interactive prompt loop so you can ask follow-ups in the same session.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from qdrant_client import models

from app.core.config import settings
from app.core.database import SessionLocal, init_db
from app.graph.checkpointer import build_checkpointer
from app.graph.graph import build_graph
from app.graph.nodes import GraphDeps
from app.ingestion.embedding import EmbeddingService, SparseEmbeddingService
from app.models.document import Document
from app.repositories.conversation_repository import ConversationRepository
from app.retrieval.reranker import FlashRankReranker
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService

if sys.platform == "win32":
    # psycopg async + AsyncPostgresSaver need a selector loop on Windows.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _to_console(text: str) -> str:
    """Strip characters the console encoding cannot represent."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    if encoding.lower().replace("-", "") == "utf8":
        return text
    return text.encode(encoding, errors="replace").decode(encoding)

# ---------------------------------------------------------------------------
# Configuration (edit here)
# ---------------------------------------------------------------------------

# The user_id the benchmark runs as. Override with --user-id.
BENCH_USER_ID: str = "user_d0e3a63f2c274f3786c3cad0beb61c84"

# Restrict retrieval to these docs. None = all of the user's documents.
BENCH_DOC_IDS: list[str] | None = None

# Default queries (ignored when --queries is passed). Empty list drops into
# the interactive prompt loop.
BENCH_QUERIES: list[str] = [
    "What are the main ML project topics covered in this document?",
]

# Where the per-run logs land (backend/bench_logs/).
LOG_DIR = Path(__file__).resolve().parent.parent / "bench_logs"


class _StubLLM:
    """Deterministic stand-in that skips the Groq round-trip (--no-llm)."""

    model = "stub"

    async def stream(self, system: str, user_prompt: str):
        yield "[stub] Grounded answer placeholder (retrieval benchmarking only)."


def _single_line(text: str, limit: int = 140) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _section(section_path) -> str:
    if not section_path:
        return "-"
    if isinstance(section_path, (list, tuple)):
        return " / ".join(str(p) for p in section_path if str(p))
    return str(section_path)


def _location(payload: dict) -> str:
    parts = []
    if payload.get("pages"):
        parts.append("p." + ",".join(str(p) for p in payload["pages"]))
    if payload.get("slides"):
        parts.append("s." + ",".join(str(p) for p in payload["slides"]))
    if payload.get("sheets"):
        parts.append("sheet " + ",".join(str(p) for p in payload["sheets"]))
    if payload.get("row_start") is not None:
        parts.append(f"row {payload['row_start']}-{payload.get('row_end', '?')}")
    return ",".join(parts) if parts else "-"


def _doc_label(payload: dict) -> str:
    return (
        f"{payload.get('doc_id', '-')} | "
        f"{_section(payload.get('section_path'))} | "
        f"{_location(payload)}"
    )


class BenchLog:
    """Writes the run record to a file; ``info`` lines also reach the console."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")

    def info(self, text: str) -> None:
        print(_to_console(text))
        self.write(text)

    def detail(self, text: str) -> None:
        self.write(text)

    def write(self, text: str) -> None:
        self.fh.write(text + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the real LangGraph RAG retrieval pipeline "
        "for a user, with optional document subsetting. Full run log is "
        "written to backend/bench_logs/.",
    )
    p.add_argument(
        "--user-id",
        default=None,
        help=f"The user id to run the query as (default: file constant, "
        f"{BENCH_USER_ID!r}).",
    )
    p.add_argument(
        "--doc-ids",
        default=None,
        help="Comma/space separated doc_ids to restrict retrieval to "
        "(default: all of the user's documents).",
    )
    p.add_argument(
        "--chat-id",
        default=None,
        help="Reuse an existing conversation owned by the user; if omitted a "
        "new conversation is created (its chat_id is printed).",
    )
    p.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Queries to run in this session (overrides file constants). "
        "Without any configured query, an interactive prompt loop is used.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=settings.conversation_search_top_k,
        help=f"Final number of hits (default: {settings.conversation_search_top_k}).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the Groq round-trip with a deterministic stub generator "
        "(pure retrieval timing; prompt is still logged).",
    )
    p.add_argument(
        "--list-docs",
        action="store_true",
        help="Print the user's documents and exit (no pipeline runs).",
    )
    return p.parse_args()


def _split_doc_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    ids = [tok.strip() for tok in raw.replace(",", " ").split() if tok.strip()]
    return ids or None


def _build_deps(db, args: argparse.Namespace) -> GraphDeps:
    qdrant = None
    if settings.qdrant_url:
        qdrant = QdrantService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            vector_size=settings.embedding_dimension,
            collection_name=settings.qdrant_collection_name,
            timeout_seconds=settings.qdrant_timeout_seconds,
            upsert_batch_size=settings.qdrant_upsert_batch_size,
        )
    llm: LLMService | _StubLLM = _StubLLM() if args.no_llm else LLMService()
    return GraphDeps(
        qdrant=qdrant,
        embedder=EmbeddingService(),
        sparse_embedder=SparseEmbeddingService(),
        reranker=FlashRankReranker(),
        llm=llm,
        conversation_repository=ConversationRepository(db),
        conversation_top_k=args.top_k,
    )


def _print_docs(db, user_id: str) -> None:
    rows = db.execute(
        select(
            Document.doc_id,
            Document.original_filename,
            Document.status,
            Document.created_at,
        )
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    ).all()
    if not rows:
        print("No documents found for this user yet.")
        return
    print(f"{'doc_id':<38} {'status':<12} {'filename'}")
    for doc_id, filename, status, created in rows:
        print(f"{doc_id:<38} {status:<12} {filename}")


def _resolve_chat(db, args: argparse.Namespace, title_hint: str | None = None) -> str:
    repo = ConversationRepository(db)
    if args.chat_id:
        conv = repo.get_conversation(args.chat_id, args.user_id)
        if conv is None:
            print(
                f"ERROR: chat_id {args.chat_id!r} does not exist for this user.",
                file=sys.stderr,
            )
            sys.exit(2)
        return args.chat_id
    title = (title_hint or "retrieval benchmark")[:60]
    conv = repo.create_conversation(
        chat_id=uuid.uuid4().hex,
        user_id=args.user_id,
        title=title,
    )
    return conv.chat_id


async def run_turn(
    graph,
    chat_id: str,
    user_id: str,
    query: str,
    selected_doc_ids: list[str] | None,
) -> dict:
    """Run one graph turn, timestamping each completed node."""
    config = {"configurable": {"thread_id": chat_id}}
    input_state = {
        "query": query,
        "user_id": user_id,
        "chat_id": chat_id,
        "selected_doc_ids": selected_doc_ids,
        "messages": [],
    }

    start = time.perf_counter()
    prev = start
    node_times: dict[str, float] = {}
    node_updates: dict[str, dict] = {}
    token_chunks = 0
    token_chars = 0

    async for mode, data in graph.astream(
        input_state,
        config=config,
        stream_mode=["custom", "updates"],
    ):
        now = time.perf_counter()
        if mode == "custom":
            token_chunks += 1
            token_chars += len(data or "")
            continue
        for node_name, out in (data or {}).items():
            if node_name.startswith("__"):
                continue
            node_times[node_name] = (now - prev) * 1000.0
            node_updates[node_name] = out if isinstance(out, dict) else {}
        prev = now

    snap = await graph.aget_state(config)
    total_ms = (time.perf_counter() - start) * 1000.0

    return {
        "node_times": node_times,
        "updates": node_updates,
        "values": snap.values,
        "token_chunks": token_chunks,
        "token_chars": token_chars,
        "total_ms": total_ms,
    }


def _arm_query_points(
    qdrant: QdrantService,
    *,
    query_dense: list[float] | None,
    query_sparse: dict | None,
    user_id: str,
    doc_ids: list[str] | None,
    limit: int,
) -> list[dict]:
    """Run a single Qdrant arm (dense or sparse) with the tenant filter."""
    qfilter = qdrant.tenant_filter(user_id, doc_ids=doc_ids, chunk_type="child")

    if query_dense is not None:
        prefetch = [
            models.Prefetch(query=query_dense, using="", limit=limit, filter=qfilter)
        ]
    else:
        prefetch = [
            models.Prefetch(
                query=qdrant._sparse_query(query_sparse),
                using=qdrant.SPARSE_VECTOR_NAME,
                limit=limit,
                filter=qfilter,
            )
        ]

    result = qdrant.client.query_points(
        collection_name=qdrant.collection_name,
        prefetch=prefetch,
        query=qdrant._fusion_query("rrf"),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "id": str(point.id),
            "score": float(point.score),
            "payload": (point.payload or {}),
        }
        for point in result.points
    ]


def _raw_qdrant_search(
    deps: GraphDeps,
    query: str,
    user_id: str,
    selected_doc_ids: list[str] | None,
    top_k: int,
) -> dict:
    """Embed the query once, then run dense arm + sparse arm + hybrid fusion."""
    out: dict = {
        "dense": [],
        "sparse": [],
        "fused": None,
        "ms_arms": 0.0,
        "ms_fused": 0.0,
        "fusion": None,
    }
    if deps.qdrant is None:
        return out

    t0 = time.perf_counter()
    sparse_vectors = deps.sparse_embedder.query_embed(query)
    dense_vectors = deps.embedder.embed([query])
    ms_embed = (time.perf_counter() - t0) * 1000.0

    query_dense = (
        dense_vectors[0]
        if dense_vectors
        else [0.0] * deps.embedder.dimension
    )
    prefetch_dense = settings.retrieval_prefetch_dense
    prefetch_sparse = settings.retrieval_prefetch_sparse
    fusion = settings.retrieval_fusion

    t0 = time.perf_counter()
    out["dense"] = _arm_query_points(
        deps.qdrant,
        query_dense=query_dense,
        query_sparse=None,
        user_id=user_id,
        doc_ids=selected_doc_ids,
        limit=prefetch_dense,
    )
    out["sparse"] = _arm_query_points(
        deps.qdrant,
        query_dense=None,
        query_sparse=sparse_vectors,
        user_id=user_id,
        doc_ids=selected_doc_ids,
        limit=prefetch_sparse,
    )
    out["ms_arms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    out["fused"] = deps.qdrant.hybrid_search(
        query_dense,
        sparse_vectors,
        user_id=user_id,
        doc_ids=selected_doc_ids,
        chunk_type="child",
        top_k=max(top_k, prefetch_dense, prefetch_sparse),
        prefetch_dense=prefetch_dense,
        prefetch_sparse=prefetch_sparse,
        fusion=fusion,
    )
    out["ms_fused"] = (time.perf_counter() - t0) * 1000.0
    out["fusion"] = fusion
    out["ms_embed"] = ms_embed
    return out


def _embed_and_store(embedder, repo, message_id: str, content: str) -> None:
    if not (content or "").strip():
        return
    try:
        vec = embedder.embed([content])[0]
        repo.update_embedding(message_id, vec)
    except Exception as exc:  # noqa: BLE001 - retention must never crash the bench
        print(f"  ! could not store embedding for {message_id}: {exc}")


def _persist_turn(
    db,
    deps: GraphDeps,
    chat_id: str,
    query: str,
    answer: str | None,
    citations: list[dict],
    error: str | None,
) -> None:
    repo = deps.conversation_repository
    user_id = uuid.uuid4().hex
    repo.add_message(user_id, chat_id, "user", query)
    _embed_and_store(deps.embedder, repo, user_id, query)

    assistant_id = uuid.uuid4().hex
    if error is not None:
        content = (
            "I couldn't generate an answer for that request. "
            "Please try again. " + f"({error})" if error else ""
        )
        repo.add_message(assistant_id, chat_id, "assistant", content, citations=[])
    else:
        repo.add_message(
            assistant_id,
            chat_id,
            "assistant",
            answer or "",
            citations=citations,
        )
    _embed_and_store(deps.embedder, repo, assistant_id, answer or "")


def _render_separator(title: str, width: int = 78) -> str:
    return "\n" + "=" * width + "\n" + title + "\n" + "=" * width


def _render_arm_log(name: str, arm_points: list[dict], limit: int) -> list[str]:
    lines = [
        f"{name} arm (top={limit}) - which points were retrieved, with scores:"
    ]
    if not arm_points:
        lines.append("  (no hits)")
    for i, pt in enumerate(arm_points, start=1):
        pl = pt.get("payload") or {}
        lines.append(
            f"  {i:>2}. {pt.get('id', '-')} | {_doc_label(pl)} | "
            f"score={pt.get('score', 0.0):.4f}"
        )
        lines.append(f"         {_single_line(str(pl.get('text', '')), 120)}")
    return lines


def _render_turn(
    log: BenchLog,
    idx: int,
    query: str,
    turn: dict,
    raw: dict,
    args: argparse.Namespace,
    chat_id: str,
) -> None:
    updates = turn["updates"]
    values = turn["values"]

    log.info(_render_separator(f"TURN {idx} | chat_id={chat_id}"))

    search = updates.get("search_conversation", {})
    conv_search = search.get("conversation_search") or {}
    route = updates.get("route", {})
    retrieve = updates.get("retrieve_documents", {})
    fuse = updates.get("fuse", {})
    rerank = updates.get("rerank", {})
    enrich = updates.get("enrich", {})
    generate = updates.get("generate", {})

    log.info("\nquery      : " + _single_line(query, 200))
    log.info(
        "route      : "
        + str(route.get("route_decision"))
        + "  ("
        + _single_line(str(route.get("route_reason") or "-"), 120)
        + ")"
    )
    log.info(
        "conversation: "
        + f"{conv_search.get('num_message_turns', 0)} turns searched, "
        + f"{conv_search.get('num_searched', 0)} embedded, "
        + f"best_score={conv_search.get('best_score', 0.0):.3f}"
    )

    # --------------------------------------------------------------- detail
    # Conversation hits (all, with scores)
    conv_hits = conv_search.get("hits") or []
    log.detail("\n[detail] conversation hits (all, with cosine scores):")
    for i, h in enumerate(conv_hits, start=1):
        msg = h.get("message") or {}
        lines = [
            f"  {i:>2}. [{msg.get('role', '?')}] "
            f"id={msg.get('message_id', '-')} score={h.get('score', 0.0):.4f}",
        ]
        lines.append(f"         {_single_line(str(msg.get('content', '')), 200)}")
        if msg.get("timestamp"):
            lines.append(f"         ts={msg.get('timestamp')}")
        log.detail("\n".join(lines))

    # --------------------------------------------------------------- raw Qdrant
    if raw["fused"] is None:
        log.info("\nQdrant: disabled (no qdrant_url) - document retrieval skipped")
    else:
        prefetch_dense = settings.retrieval_prefetch_dense
        prefetch_sparse = settings.retrieval_prefetch_sparse
        log.detail("\n".join(_render_arm_log(
            "dense (cosine)", raw["dense"], prefetch_dense
        )))
        log.detail("")
        log.detail("\n".join(_render_arm_log(
            "sparse (BM25)", raw["sparse"], prefetch_sparse
        )))
        log.detail(
            f"\n[detail] Qdrant per-arm latency {raw.get('ms_embed', 0.0):.1f} ms "
            f"(embed) + {raw.get('ms_arms', 0.0):.1f} ms (arms)"

        )

        fused = raw["fused"] or []
        log.info(
            f"\nQdrant dense+BM25 fused (fusion={raw['fusion']}, "
            f"top={max(args.top_k, settings.retrieval_prefetch_dense, settings.retrieval_prefetch_sparse)}) "
            f"{raw['ms_fused']:.1f} ms"
        )
        for i, pt in enumerate(fused[: args.top_k], start=1):
            pl = pt.get("payload") or {}
            log.info(
                f"  {i:>2}. {_doc_label(pl)} | "
                f"score={pt.get('score', 0.0):.3f}"
            )
            log.detail(
                f"       {_single_line(str(pl.get('text', '')), 150)}"
            )

    # --------------------------------------------------------------- graph fusion (RRF)
    graph_fused = fuse.get("fused_hits") or []
    if graph_fused:
        log.info(
            "\nGraph RRF fusion (conversation + docs, "
            f"fusible={bool(fuse.get('fusible', False))}): "
            f"{len(graph_fused)} items in order"
        )
    for i, f in enumerate(graph_fused[: args.top_k], start=1):
        tag = f.get("source", "?")
        score = f.get("rrf_score")
        if tag == "document":
            label = _doc_label(f.get("payload") or {})
        else:
            pl = f.get("payload") or {}
            label = f"[{pl.get('role', '?')}] {_single_line(str(pl.get('content', '')), 60)}"
        rrf = f"rrf={score:.4f}" if isinstance(score, float) else "rrf=-"
        log.info(f"  {i:>2}. {tag:<12} {rrf:<24} {label}")

    log.detail("\n[detail] full RRF fused ordering (all items, before rerank):")
    for i, f in enumerate(graph_fused, start=1):
        tag = f.get("source", "?")
        rrf = f.get("rrf_score")
        pl = f.get("payload") or {}
        if tag == "document":
            label = (
                f"id={f.get('item_id')} {_doc_label(pl)} | "
                f"qdrant_fused_score={pl.get('score', 0.0):.3f}"
            )
        else:
            label = (
                f"id={f.get('item_id')} [{pl.get('role', '?')}] "
                f"{_single_line(str(pl.get('content', '')), 90)}"
            )
        rrf_s = f"{rrf:.4f}" if isinstance(rrf, float) else "-"
        log.detail(f"  {i:>2}. {tag:<12} rrf={rrf_s:<24} {label}")

    # --------------------------------------------------------------- pre/post rerank
    pre_rerank_hits = retrieve.get("document_hits")
    if pre_rerank_hits:
        log.detail(
            "\n[detail] document hits entering rerank (retrieve/node order, "
            f"{len(pre_rerank_hits)}):"
        )
        for i, d in enumerate(pre_rerank_hits[: args.top_k], start=1):
            log.detail(
                f"  {i:>2}. {d.get('chunk_id', '-')} | {_doc_label(d)} | "
                f"score={d.get('score', 0.0):.3f} "
                f"rerank={d.get('rerank_score', 0.0):.4f}"
            )

    final_hits = rerank.get("document_hits")
    if final_hits is None:
        final_hits = pre_rerank_hits or []
    if final_hits:
        log.info(f"\nReranked final document hits ({len(final_hits)}):")
        for i, d in enumerate(final_hits[: args.top_k], start=1):
            rerank_display = (
                f"{d['rerank_score']:.4f}"
                if isinstance(d.get("rerank_score"), float)
                else "-"
            )
            log.info(
                f"  {i:>2}. {d.get('doc_id', '-')} | "
                f"{_section(d.get('section_path'))} | {_location(d)} | "
                f"score={d.get('score', 0.0):.3f} rerank={rerank_display}"
            )
        log.detail(
            "\n[detail] full reranked selection (all points, with scores):"
        )
        for i, d in enumerate(final_hits, start=1):
            log.detail(
                f"  {i:>2}. {d.get('chunk_id', '-')} | {_doc_label(d)} | "
                f"score={d.get('score', 0.0):.3f} "
                f"rerank={d.get('rerank_score', 0.0):.4f}"
            )

    # --------------------------------------------------------------- enrichment
    parents = enrich.get("parents") or values.get("parents") or []
    children = enrich.get("children") or values.get("children") or []
    log.info(f"\nenrich     : {len(parents)} parents / {len(children)} children")
    log.detail("\n[detail] enrichment inputs for the final LLM:")
    for i, p in enumerate(parents, start=1):
        log.detail(
            f"  parent {i:>2}: {p.get('chunk_id', '-')} | "
            f"{_section(p.get('section_path'))} "
            f"| chars={len(str(p.get('text') or ''))} "
            "(full text is in the user prompt)"
        )
    for i, c in enumerate(children, start=1):
        log.detail(
            f"  child  {i:>2}: {c.get('chunk_id', '-')} | "
            f"{_section(c.get('section_path'))} | {_location(c)}"
        )

    # --------------------------------------------------------------- final prompt
    system_prompt = generate.get("system_prompt") or ""
    user_prompt = generate.get("user_prompt") or ""
    if system_prompt or user_prompt:
        log.detail(_render_separator(
            f"FINAL LLM PROMPT (turn {idx}) - after enrichment + parent folding"
        ))
        log.detail("[SYSTEM PROMPT]")
        log.detail(system_prompt)
        log.detail("\n[USER PROMPT]")
        log.detail(user_prompt)
        log.detail("")
        log.info(
            "\nfinal prompt: captured "
            f"(system={len(system_prompt)} chars, user={len(user_prompt)} chars)"
        )

    # --------------------------------------------------------------- latency
    log.info("\nper-node latency (ms):")
    order = [
        "preprocess",
        "search_conversation",
        "route",
        "retrieve_documents",
        "fuse",
        "rerank",
        "enrich",
        "generate",
        "persist",
    ]
    node_sum = 0.0
    for name in order:
        if name in turn["node_times"]:
            ms = turn["node_times"][name]
            node_sum += ms
            log.info(f"  {name:<20} {ms:8.1f}")
    missing = [n for n in order if n not in turn["node_times"]]
    if missing:
        log.info(f"  (skipped: {', '.join(missing)})")

    retrieval_ms = values.get("retrieval_latency_ms", 0.0)
    log.info(
        f"\ntotals     : graph wall {turn['total_ms']:.1f} ms | "
        f"state retrieval_latency_ms={retrieval_ms:.1f} | "
        f"node sum {node_sum:.1f} ms"
    )
    log.info(
        f"generate   : model={values.get('model', '-')} | "
        f"{turn['token_chunks']} token chunks / {turn['token_chars']} chars streamed "
        f"({turn['node_times'].get('generate', 0.0):.1f} ms)"
    )
    log.info(
        "answer     : "
        + _single_line(str(values.get("answer", "")), 220)
    )
    log.info(f"citations  : {len(values.get('citations', []))}")


async def main(args: argparse.Namespace) -> None:
    user_id = args.user_id or BENCH_USER_ID
    if not user_id:
        print("ERROR: --user-id is required.", file=sys.stderr)
        sys.exit(2)

    init_db()
    db = SessionLocal()

    if args.list_docs:
        _print_docs(db, user_id)
        db.close()
        return

    selected_doc_ids = _split_doc_ids(args.doc_ids) if args.doc_ids else BENCH_DOC_IDS
    if selected_doc_ids:
        print(f"Restricting retrieval to {len(selected_doc_ids)} doc(s).")
    else:
        print("Retrieving over all of the user's documents.")

    if args.queries:
        queries = list(args.queries)
    elif BENCH_QUERIES:
        queries = list(BENCH_QUERIES)
    else:
        queries = None

    chat_id = _resolve_chat(
        db, args, title_hint=(queries[0] if queries else None)
    )
    print(f"chat_id    : {chat_id}   (reuse with --chat-id {chat_id})")
    print(
        f"top_k      : {args.top_k} | "
        f"llm        : {'stub' if args.no_llm else 'groq (' + settings.groq_chat_model + ')'}"
    )

    log_path = LOG_DIR / f"bench_retrieval_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = BenchLog(log_path)
    log.info(_render_separator("RETRIEVAL BENCHMARK"))
    log.info(f"started  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"user_id  : {user_id}")
    log.info(f"doc_ids  : {selected_doc_ids or 'ALL'}")
    log.info(f"top_k    : {args.top_k} | no_llm={bool(args.no_llm)}")
    log.info(f"chat_id  : {chat_id}")
    log.info(f"log file : {log_path.resolve()}")
    log.info("")

    deps = _build_deps(db, args)
    checkpointer = await build_checkpointer()
    try:
        graph = build_graph(deps, checkpointer=checkpointer)

        async def _run_one(query: str, idx: int) -> None:
            t0 = time.perf_counter()
            error = None
            turn = None
            try:
                turn = await run_turn(
                    graph,
                    chat_id,
                    user_id,
                    query,
                    selected_doc_ids,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                turn = {
                    "node_times": {},
                    "updates": {},
                    "values": {},
                    "token_chunks": 0,
                    "token_chars": 0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                }
                print(f"  ! graph turn failed: {exc}", file=sys.stderr)

            raw = _raw_qdrant_search(
                deps,
                query,
                user_id,
                selected_doc_ids,
                args.top_k,
            )
            values = turn["values"]
            _persist_turn(
                db,
                deps,
                chat_id,
                query,
                values.get("answer"),
                values.get("citations", []),
                error,
            )
            _render_turn(log, idx, query, turn, raw, args, chat_id)
            print()

        idx = 0
        if queries:
            for query in queries:
                if not query.strip():
                    continue
                idx += 1
                await _run_one(query.strip(), idx)
        else:
            while True:
                try:
                    prompt = input("\n> (message; 'exit' to quit) ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                prompt = prompt.strip()
                if prompt.lower() in ("exit", "quit"):
                    break
                if not prompt:
                    continue
                idx += 1
                await _run_one(prompt, idx)
    finally:
        log.info(_render_separator("BENCHMARK COMPLETE"))
        log.info(f"full log : {log_path.resolve()}")
        log.close()
        conn = getattr(checkpointer, "conn", None)
        if conn is not None:
            await conn.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))