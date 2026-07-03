#!/usr/bin/env python
"""Lightweight KV-aware router for the 2-node cxl harness.

Sits in front of N vLLM nodes (static list passed at startup) and forwards
OpenAI-compatible requests to one of them per a selectable routing strategy.
Replaces NVIDIA Dynamo's frontend for routing-strategy experiments without
pinning vLLM to an old version.

Strategies (``--strategy``):
    round_robin : next node in rotation; ignores all signals.
    random      : uniformly random node.
    gpu_load    : node with the fewest running requests (KV-cache usage as a
                  tiebreaker), scraped from each node's vLLM /metrics.
    max_prefix  : node with the longest cached prompt prefix, via each node's
                  LMCache /lookup_hits endpoint.
    weighted    : argmax of ``w_prefix * norm(prefix_hit) - w_load * norm(load)``,
                  combining the prefix and GPU-load signals.

Signals:
    - KV prefix hit: POST {model_name, token_ids} to each node's LMCache MP
      server /lookup_hits; uses returned ``best_hit_tokens`` (highest tier
      with a hit). Tokenisation happens
      in the router so the IDs match what the node would cache (the prompt is
      sent to vLLM as token IDs, not text, to keep them consistent).
    - GPU load: each node's vLLM /metrics, scraped on a background timer so it
      is off the request hot path. Robust to metric-name drift across vLLM
      versions (tries several known gauge names).

The router exposes the OpenAI surface it proxies (``/v1/completions`` and
``/v1/chat/completions``) plus ``/health`` and ``/router/nodes`` (debug).

Usage:
    python router.py \
        --nodes c1=http://192.168.128.31:8010,c2=http://192.168.128.32:8010 \
        --lookup c1=http://192.168.128.31:8090,c2=http://192.168.128.32:8090 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --strategy weighted --w-prefix 0.7 --w-load 0.3 \
        --port 8000
"""
# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import argparse
import asyncio
import itertools
import logging
import random
import re

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer
import httpx
import uvicorn

logger = logging.getLogger("router")

# Routing strategies. Kept as string constants (not an enum) only to match the
# CLI surface; validated in parse_args.
STRATEGIES = (
    "round_robin",
    "random",
    "gpu_load",
    "max_prefix",
    "weighted",
    "anti_affinity",
)

# Strategies that consume the per-request KV-prefix signal and therefore need
# the prompt tokenised. Listed once so the tokenisation gate in _proxy and the
# strategies stay in sync.
_PREFIX_AWARE = ("max_prefix", "weighted", "anti_affinity")

# Candidate Prometheus gauge names for the GPU-load signals, newest first.
# vLLM has renamed these across releases (e.g. gpu_cache_usage_perc ->
# kv_cache_usage_perc), so the scraper tries each and uses the first present.
_RUNNING_METRIC_NAMES = ("vllm:num_requests_running",)
_KV_USAGE_METRIC_NAMES = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)

# A Prometheus sample line: ``name{labels} value``. Labels are optional.
_PROM_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)")


@dataclass
class NodeState:
    """A routable node and its latest scraped load signals.

    Attributes:
        name: Short label for logs / debug endpoint.
        serve_url: Base URL of the node's vLLM OpenAI server.
        lookup_url: Base URL of the node's LMCache MP HTTP server (the host
            of ``/lookup_hits``).
        num_running: Running-request count from the last /metrics scrape
            (``inf`` until first successful scrape, so an unscraped node is
            never preferred by gpu_load).
        kv_usage: KV-cache usage fraction [0,1] from the last scrape.
    """

    name: str
    serve_url: str
    lookup_url: str
    num_running: float = float("inf")
    kv_usage: float = 0.0


@dataclass
class RoutingDecision:
    """The outcome of a single routing decision, for logging.

    Attributes:
        node: Name of the chosen node.
        reason: Short tag for why (e.g. ``"max-prefix"``, ``"tie-rr"``,
            ``"anti-affinity-cold"``, ``"anti-affinity-fallback"``).
        hits: Per-node KV-prefix hit chunks used by the decision, or an empty
            dict for strategies that don't consult the cache (round_robin /
            random / gpu_load).
        loads: Per-node running-request counts, for load-aware strategies;
            empty otherwise.
    """

    node: str
    reason: str
    hits: dict[str, int] = field(default_factory=dict)
    loads: dict[str, float] = field(default_factory=dict)


@dataclass
class RouterConfig:
    """Resolved router configuration (see module docstring / parse_args)."""

    nodes: list[NodeState]
    model: str
    strategy: str
    w_prefix: float
    w_load: float
    port: int
    host: str
    metrics_interval: float
    lookup_timeout: float
    log_level: str
    # Mutable runtime state lives here so the FastAPI handlers can reach it
    # via app.state without globals.
    _rr_counter: "itertools.count" = field(default_factory=lambda: itertools.count())


def parse_node_map(serve_arg: str, lookup_arg: str) -> list[NodeState]:
    """Parse the ``--nodes`` and ``--lookup`` ``name=url,...`` lists into nodes.

    Args:
        serve_arg: Comma-separated ``name=serve_url`` pairs (vLLM endpoints).
        lookup_arg: Comma-separated ``name=lookup_url`` pairs (MP HTTP
            endpoints). Must cover the same names as ``serve_arg``.

    Returns:
        One :class:`NodeState` per name, in the order given by ``serve_arg``.

    Raises:
        ValueError: On malformed entries or a name/serve/lookup mismatch.
    """

    def _parse(arg: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in arg.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"expected name=url, got {entry!r}")
            name, url = entry.split("=", 1)
            out[name.strip()] = url.strip().rstrip("/")
        return out

    serve = _parse(serve_arg)
    lookup = _parse(lookup_arg)
    if set(serve) != set(lookup):
        raise ValueError(
            f"--nodes names {sorted(serve)} != --lookup names {sorted(lookup)}"
        )
    if not serve:
        raise ValueError("at least one node is required")
    return [NodeState(name=n, serve_url=serve[n], lookup_url=lookup[n]) for n in serve]


# ---------------------------------------------------------------------------
# Load-signal scraping.
# ---------------------------------------------------------------------------
def _parse_prom_metric(text: str, names: tuple[str, ...]) -> float | None:
    """Sum the values of the first present metric name across its label sets.

    vLLM emits one sample per (metric, label-set); a single-model node has one
    relevant sample, but summing is correct and robust if labels split it.

    Args:
        text: Raw Prometheus exposition text.
        names: Candidate metric names, in priority order.

    Returns:
        The summed value of the first name that appears, or None if none do.
    """
    for name in names:
        total = 0.0
        found = False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = _PROM_LINE.match(line)
            if m and m.group("name") == name:
                try:
                    total += float(m.group("value"))
                    found = True
                except ValueError:
                    continue
        if found:
            return total
    return None


async def scrape_loop(app: FastAPI) -> None:
    """Periodically refresh each node's load signals from its vLLM /metrics.

    Runs for the life of the app. Scrape failures are swallowed (the node
    keeps its last values) so a transient blip doesn't crash routing.

    Args:
        app: The FastAPI app; reads config + client from ``app.state``.
    """
    cfg: RouterConfig = app.state.cfg
    client: httpx.AsyncClient = app.state.client
    while True:
        async def _scrape(node: NodeState) -> None:
            try:
                resp = await client.get(f"{node.serve_url}/metrics", timeout=2.0)
                resp.raise_for_status()
            except (httpx.HTTPError, asyncio.TimeoutError):
                return
            running = _parse_prom_metric(resp.text, _RUNNING_METRIC_NAMES)
            usage = _parse_prom_metric(resp.text, _KV_USAGE_METRIC_NAMES)
            if running is not None:
                node.num_running = running
            if usage is not None:
                node.kv_usage = usage

        await asyncio.gather(*(_scrape(n) for n in cfg.nodes))
        await asyncio.sleep(cfg.metrics_interval)


# ---------------------------------------------------------------------------
# Prefix-hit probing.
# ---------------------------------------------------------------------------
async def prefix_hits(app: FastAPI, token_ids: list[int]) -> dict[str, int]:
    """Query every node's /lookup_hits for the cached prefix length.

    Args:
        app: The FastAPI app; reads config + client from ``app.state``.
        token_ids: The request's prompt token IDs.

    Returns:
        Mapping of node name -> cached prefix length in CHUNKS. A node that
        errors or 404s (model not registered yet) maps to 0. Chunks (not
        tokens) are used because the routing decisions (max_prefix argmax,
        weighted normalisation) are scale-invariant and chunks read more
        naturally in the decision log.
    """
    cfg: RouterConfig = app.state.cfg
    client: httpx.AsyncClient = app.state.client

    async def _one(node: NodeState) -> tuple[str, int]:
        try:
            resp = await client.post(
                f"{node.lookup_url}/lookup_hits",
                json={"model_name": cfg.model, "token_ids": token_ids},
                timeout=cfg.lookup_timeout,
            )
            if resp.status_code != 200:
                return node.name, 0
            # best_hit_chunks = highest-tier resident prefix (L0 then L1), in
            # chunks. Routing on the best tier maximises reuse at the fastest
            # level; chunk granularity matches how LMCache keys the cache.
            return node.name, int(resp.json().get("best_hit_chunks", 0))
        except (httpx.HTTPError, ValueError, asyncio.TimeoutError):
            return node.name, 0

    pairs = await asyncio.gather(*(_one(n) for n in cfg.nodes))
    return dict(pairs)


# ---------------------------------------------------------------------------
# Strategy selection.
# ---------------------------------------------------------------------------
def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalise to [0,1]; all-equal inputs map to 0 (no signal)."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


async def choose_node(
    app: FastAPI, token_ids: list[int]
) -> tuple[NodeState, RoutingDecision]:
    """Pick a node for this request per the configured strategy.

    Args:
        app: The FastAPI app; reads config + runtime state from ``app.state``.
        token_ids: Prompt token IDs (needed by prefix-aware strategies).

    Returns:
        The chosen :class:`NodeState` and a :class:`RoutingDecision` capturing
        the signals behind the choice, for the caller to log.
    """
    cfg: RouterConfig = app.state.cfg
    nodes = cfg.nodes

    if cfg.strategy == "round_robin":
        node = nodes[next(cfg._rr_counter) % len(nodes)]
        return node, RoutingDecision(node=node.name, reason="round-robin")

    if cfg.strategy == "random":
        node = random.choice(nodes)
        return node, RoutingDecision(node=node.name, reason="random")

    if cfg.strategy == "gpu_load":
        # Fewest running requests; KV-cache usage breaks ties.
        loads = {n.name: n.num_running for n in nodes}
        node = min(nodes, key=lambda n: (n.num_running, n.kv_usage))
        return node, RoutingDecision(node=node.name, reason="min-load", loads=loads)

    if cfg.strategy == "max_prefix":
        hits = await prefix_hits(app, token_ids)
        # Ties (e.g. all-zero cold cache) resolve to round-robin so cold load
        # still spreads instead of always hammering node 0.
        best = max(hits.values()) if hits else 0
        contenders = [n for n in nodes if hits.get(n.name, 0) == best]
        if len(contenders) == 1:
            node, reason = contenders[0], "max-prefix"
        else:
            node = contenders[next(cfg._rr_counter) % len(contenders)]
            reason = "max-prefix-tie-rr"
        return node, RoutingDecision(node=node.name, reason=reason, hits=hits)

    if cfg.strategy == "anti_affinity":
        # Cache anti-affinity: route AWAY from nodes that already hold the
        # prompt, forcing a fresh / cross-node fetch. First occurrence (no
        # node has it cached) goes to a random node; repeats go to a random
        # node among those WITHOUT the prefix cached. This is keyed on live
        # cache state (/lookup_hits), not on remembering the first node, so
        # it needs no router-side memory and self-corrects after eviction.
        hits = await prefix_hits(app, token_ids)
        cold = [n for n in nodes if hits.get(n.name, 0) == 0]
        if cold:
            reason = "anti-affinity-cold"
        else:
            # Every node already has it (e.g. 2-node topology after both have
            # served it). Anti-affinity is unsatisfiable, so fall back to the
            # least-cached node to keep spreading rather than always node 0.
            fewest = min(hits.values())
            cold = [n for n in nodes if hits.get(n.name, 0) == fewest]
            reason = "anti-affinity-fallback"
        node = random.choice(cold)
        return node, RoutingDecision(node=node.name, reason=reason, hits=hits)

    # weighted: maximise prefix overlap while avoiding loaded nodes.
    hits = await prefix_hits(app, token_ids)
    loads = {n.name: n.num_running for n in nodes}
    hit_norm = _normalize({n.name: float(hits.get(n.name, 0)) for n in nodes})
    load_norm = _normalize({n.name: n.num_running for n in nodes})
    scores = {
        n.name: cfg.w_prefix * hit_norm[n.name] - cfg.w_load * load_norm[n.name]
        for n in nodes
    }
    best_name = max(scores, key=lambda k: scores[k])
    node = next(n for n in nodes if n.name == best_name)
    return node, RoutingDecision(
        node=node.name, reason="weighted", hits=hits, loads=loads
    )


# ---------------------------------------------------------------------------
# Proxying.
# ---------------------------------------------------------------------------
def _routing_token_ids(body: dict, path: str, tokenizer) -> list[int]:
    """Tokenise a request EXACTLY as the backend vLLM will, for /lookup_hits.

    The routing prefix signal only works if the router's token IDs match the
    ones vLLM stores KV under. The two OpenAI surfaces tokenise differently:

    - ``/v1/completions``: vLLM tokenises the raw ``prompt`` string with the
      special tokens it prepends (BOS). We mirror that with
      ``encode(prompt, add_special_tokens=True)``.
    - ``/v1/chat/completions``: vLLM does NOT tokenise the joined message
      contents — it renders ``messages`` through the model's chat template
      (role headers, system preamble, ``<|eot_id|>`` etc., plus a trailing
      generation prompt) and tokenises THAT. Concatenating contents produces
      a completely different sequence (no shared prefix), so we must use
      ``apply_chat_template`` with ``add_generation_prompt=True`` to match.

    Args:
        body: Parsed request JSON.
        path: OpenAI sub-path being proxied.
        tokenizer: The HF tokenizer for the served model.

    Returns:
        Token IDs matching vLLM's tokenisation, or an empty list if the body
        has no usable prompt/messages.
    """
    if path.endswith("/completions") and not path.endswith("/chat/completions"):
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return tokenizer.encode(prompt, add_special_tokens=True)
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            return tokenizer.encode("".join(prompt), add_special_tokens=True)
        return []

    # chat: render through the model's chat template exactly as vLLM does.
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return []
    try:
        # return_dict=False is REQUIRED: with tokenize=True, transformers>=5
        # returns a BatchEncoding dict ({"input_ids": [...]}) by default, whose
        # len() is the key count, not the token count. We need the flat list of
        # IDs to send to /lookup_hits.
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
    except Exception:
        # A malformed messages list shouldn't crash routing; fall back to no
        # signal (the strategy then treats it as a cold/tie request).
        return []


async def _proxy(app: FastAPI, request: Request, path: str) -> JSONResponse | StreamingResponse:
    """Route then reverse-proxy an OpenAI request to the chosen node.

    Args:
        app: The FastAPI app.
        request: The incoming client request.
        path: The OpenAI sub-path to forward (e.g. ``/v1/completions``).

    Returns:
        The node's response, streamed through if the client asked to stream.
    """
    cfg: RouterConfig = app.state.cfg
    client: httpx.AsyncClient = app.state.client
    body = await request.json()

    # Compute the routing token IDs only when a prefix-aware strategy needs
    # them, to keep round_robin/random/gpu_load free of tokenisation cost.
    # The IDs must match vLLM's tokenisation exactly (BOS for completions,
    # chat template for chat) or /lookup_hits never matches — see
    # _routing_token_ids.
    token_ids: list[int] = []
    if cfg.strategy in _PREFIX_AWARE:
        token_ids = _routing_token_ids(body, path, app.state.tokenizer)

    node, decision = await choose_node(app, token_ids)

    # One structured line per routing decision. hits/loads are only populated
    # for strategies that consulted them, so the line stays terse for
    # round_robin / random. ntok is the routing prompt length (0 when the
    # strategy didn't tokenise).
    hits_str = (
        " hits_chunks={"
        + ",".join(f"{k}:{v}" for k, v in decision.hits.items())
        + "}"
        if decision.hits
        else ""
    )
    loads_str = (
        " loads={"
        + ",".join(f"{k}:{v:g}" for k, v in decision.loads.items())
        + "}"
        if decision.loads
        else ""
    )
    logger.debug(
        "route %s -> %s (%s) ntok=%d%s%s",
        cfg.strategy,
        decision.node,
        decision.reason,
        len(token_ids),
        hits_str,
        loads_str,
    )

    stream = bool(body.get("stream", False))
    url = f"{node.serve_url}{path}"
    headers = {"content-type": "application/json"}

    if not stream:
        resp = await client.post(url, json=body, headers=headers, timeout=None)
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json(),
            headers={"x-router-node": node.name},
        )

    async def _iter():
        async with client.stream(
            "POST", url, json=body, headers=headers, timeout=None
        ) as resp:
            async for chunk in resp.aiter_raw():
                yield chunk

    return StreamingResponse(
        _iter(),
        media_type="text/event-stream",
        headers={"x-router-node": node.name},
    )


# ---------------------------------------------------------------------------
# App wiring.
# ---------------------------------------------------------------------------
def build_app(cfg: RouterConfig) -> FastAPI:
    """Construct the FastAPI router app for ``cfg``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.client = httpx.AsyncClient()
        # Tokeniser only needed for prefix-aware strategies; load it anyway so
        # a strategy switch at runtime (future) doesn't need a restart.
        app.state.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        scraper = asyncio.create_task(scrape_loop(app))
        yield
        scraper.cancel()
        await app.state.client.aclose()

    app = FastAPI(title="cxl lightweight router", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "strategy": cfg.strategy, "nodes": len(cfg.nodes)}

    @app.get("/router/nodes")
    async def nodes_debug() -> list[dict]:
        return [
            {
                "name": n.name,
                "serve_url": n.serve_url,
                "lookup_url": n.lookup_url,
                "num_running": n.num_running,
                "kv_usage": n.kv_usage,
            }
            for n in cfg.nodes
        ]

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        # Backends all serve the same model, so forward the first node's real
        # /v1/models rather than synthesise one. This makes the router a
        # complete OpenAI endpoint so clients that auto-detect the model
        # (e.g. `lmcache bench engine`) work against it. Falls through the
        # node list so a single down node doesn't break discovery.
        client: httpx.AsyncClient = app.state.client
        last_err = "no nodes configured"
        for node in cfg.nodes:
            try:
                resp = await client.get(f"{node.serve_url}/v1/models", timeout=5.0)
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            except (httpx.HTTPError, ValueError) as e:
                last_err = f"{node.name}: {e}"
                continue
        return JSONResponse(
            status_code=503,
            content={"error": f"no node could serve /v1/models ({last_err})"},
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy(app, request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy(app, request, "/v1/chat/completions")

    return app


def parse_args() -> RouterConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        required=True,
        help="comma-separated name=serve_url (vLLM OpenAI endpoints)",
    )
    parser.add_argument(
        "--lookup",
        required=True,
        help="comma-separated name=lookup_url (LMCache MP HTTP endpoints)",
    )
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--strategy", default="round_robin", choices=STRATEGIES, help="routing strategy"
    )
    parser.add_argument("--w-prefix", type=float, default=0.7, help="weighted: prefix weight")
    parser.add_argument("--w-load", type=float, default=0.3, help="weighted: load weight")
    parser.add_argument("--port", type=int, default=8000, help="router listen port")
    parser.add_argument("--host", default="0.0.0.0", help="router listen host")
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=1.0,
        help="seconds between /metrics scrapes",
    )
    parser.add_argument(
        "--lookup-timeout",
        type=float,
        default=1.0,
        help="per-node /lookup_hits timeout (seconds)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="router log verbosity; per-request routing lines are DEBUG "
        "(default: info)",
    )
    args = parser.parse_args()

    nodes = parse_node_map(args.nodes, args.lookup)
    return RouterConfig(
        nodes=nodes,
        model=args.model,
        strategy=args.strategy,
        w_prefix=args.w_prefix,
        w_load=args.w_load,
        port=args.port,
        host=args.host,
        metrics_interval=args.metrics_interval,
        lookup_timeout=args.lookup_timeout,
        log_level=args.log_level,
    )


def main() -> None:
    cfg = parse_args()
    # Configure the "router" logger so per-request decision lines reach
    # stdout (and the tee'd router.log) with a timestamp. uvicorn manages its
    # own loggers separately; this is just for our decision log.
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(levelname)s:\t%(message)s",
    )
    # httpx logs an INFO line per outbound request; at ~2 metrics scrapes/sec
    # plus per-request lookups/proxies this drowns the routing decision lines.
    # Quiet it to WARNING so only failures show. (httpcore is its noisier
    # transport-level logger; silence it too.)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
