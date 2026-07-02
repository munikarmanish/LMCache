# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

logger = init_logger(__name__)

router = APIRouter()


class LookupHitsRequest(BaseModel):
    """Request body for the ``/lookup_hits`` endpoint.

    Attributes:
        token_ids: The prompt's token IDs, in order. The endpoint reports how
            long a chunk-aligned prefix of this sequence is resident in L1.
        model_name: The served model name the KV chunks are keyed under. Must
            match a model whose KV caches are registered on this server.
        world_size: Tensor/pipeline world size the chunks are keyed under.
            Defaults to 1 (single-GPU node). The probe checks the node-local
            shard for ``worker_id`` 0.
        cache_salt: Per-user isolation salt. Must match the salt the chunks
            were stored under, or nothing will hit. Defaults to "".
    """

    token_ids: list[int] = Field(..., min_length=1)
    model_name: str
    world_size: int = 1
    cache_salt: str = ""


@router.post("/lookup_hits")
async def lookup_hits(request: Request, body: LookupHitsRequest) -> JSONResponse:
    """Report how many prefix tokens of a prompt are resident, per cache tier.

    Mirrors the server's lookup key path — hash the tokens into chunk hashes
    with the engine's :class:`TokenHasher`, build the per-chunk
    :class:`ObjectKey` s via :func:`ipc_key_to_object_keys` — then walks each
    tier's index counting the consecutive resident prefix. The probe is
    read-only: it does not pin, lock, or touch cache state, so it does not
    perturb eviction order. Intended for an external KV-aware router that fans
    this query out to every node and routes a request to the node holding the
    longest prefix at the highest tier.

    Tiers, highest (fastest) first:

    - ``l0``: vLLM's GPU KV cache (HBM), via
      :meth:`StorageManager.l0_prefix_hit_chunks`. Currently always 0 — vLLM
      runs with ``--no-enable-prefix-caching`` and LMCache keeps no L0 index,
      so there is no cross-request L0 residency to report. The field is
      present so the routing contract already carries the tier.
    - ``l1``: LMCache's DRAM (CPU) pool, via
      :meth:`StorageManager.l1_prefix_hit_chunks`.

    Args:
        request: FastAPI request; the live engine context is read from
            ``request.app.state.engine.context``.
        body: Parsed :class:`LookupHitsRequest`.

    Returns:
        JSON body on success. Per-tier counts plus a ``best_*`` aggregate (the
        highest tier with any hit, for routers that just want one number)::

            {
                "l0_hit_chunks": <int>,   # GPU/HBM resident prefix, in chunks
                "l0_hit_tokens": <int>,
                "l1_hit_chunks": <int>,   # DRAM resident prefix, in chunks
                "l1_hit_tokens": <int>,
                "best_hit_chunks": <int>, # max over tiers (== l0 or l1)
                "best_hit_tokens": <int>,
                "best_tier": <str>,       # "l0" | "l1" | "none"
                "num_tokens": <int>,      # len(token_ids), for hit-rate math
                "chunk_size": <int>,      # tokens per chunk on this server
            }

    HTTP status codes:
        200: success.
        404: no KV caches registered for ``(model_name, world_size)``
            (e.g. vLLM has not connected yet for this model).
        503: engine not yet initialised on ``app.state``.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )

    ctx = engine.context

    # A request for an unregistered (model, world_size) can never hit; surface
    # it as 404 rather than silently returning 0, which would otherwise look
    # like a cold cache and mislead the router.
    if ctx.layout_desc_registry.find(body.model_name, body.world_size) is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "no KV caches registered for (model_name=%s, "
                "world_size=%d)" % (body.model_name, body.world_size)
            },
        )

    chunk_size = ctx.chunk_size

    def _count_hits() -> tuple[int, int]:
        # Build the same per-chunk keys the lookup path uses. worker_id=0
        # pins the expansion to this node's local shard (one ObjectKey per
        # chunk), so the returned counts map 1:1 onto chunks.
        chunk_hashes = ctx.token_hasher.compute_chunk_hashes(body.token_ids)
        if not chunk_hashes:
            return 0, 0
        ipc_key = IPCCacheEngineKey(
            model_name=body.model_name,
            world_size=body.world_size,
            worker_id=0,
            token_ids=tuple(body.token_ids),
            start=0,
            end=len(body.token_ids),
            request_id="lookup_hits",
            cache_salt=body.cache_salt,
        )
        obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)
        sm = ctx.storage_manager
        return sm.l0_prefix_hit_chunks(obj_keys), sm.l1_prefix_hit_chunks(obj_keys)

    # Hashing + the index walks are synchronous and CPU-bound; run them off
    # the event loop so concurrent router probes don't serialise on it.
    loop = asyncio.get_running_loop()
    l0_chunks, l1_chunks = await loop.run_in_executor(None, _count_hits)

    # Aggregate: the highest tier with any hit. Ties go to the higher tier
    # (L0 before L1) since that is the cheaper place to serve from.
    if l0_chunks > 0:
        best_tier, best_chunks = "l0", l0_chunks
    elif l1_chunks > 0:
        best_tier, best_chunks = "l1", l1_chunks
    else:
        best_tier, best_chunks = "none", 0

    return JSONResponse(
        content={
            "l0_hit_chunks": l0_chunks,
            "l0_hit_tokens": l0_chunks * chunk_size,
            "l1_hit_chunks": l1_chunks,
            "l1_hit_tokens": l1_chunks * chunk_size,
            "best_hit_chunks": best_chunks,
            "best_hit_tokens": best_chunks * chunk_size,
            "best_tier": best_tier,
            "num_tokens": len(body.token_ids),
            "chunk_size": chunk_size,
        }
    )
