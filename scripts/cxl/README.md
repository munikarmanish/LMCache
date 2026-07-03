# Two-node L2 end-to-end harness — tutorial (CXL vs NIXL-peer/RDMA)

This harness brings up the LMCache MP server + vLLM on **two nodes** and lets
you benchmark a cross-node KV-cache hit under two different L2 adapters, plus an
optional KV-aware router in front of them. It is the reference setup for the CXL
and NIXL-peer L2 adapters:

| mode | L2 adapter | L2 tier | Cross-node data path |
|---|---|---|---|
| `cxl` | `cxl` | shared CXL pool (`/dev/dax0.0`) | `PushKVToCXL` RPC + CXL read; hits can go GPU-direct |
| `nixl` | `nixl_peer` | remote peer's L1 over RDMA | one-sided NIXL READ over `mlx5_0` into L1 |

Design docs for the adapters:
[CXL](../../docs/design/v1/distributed/l2_adapters/cxl_l2_adapter.md) ·
[NIXL-peer](../../docs/design/v1/distributed/l2_adapters/nixl_rdma_peer.md).

> The `nixl` **mode name** is short for the adapter whose spec declares
> `"type": "nixl_peer"`; only the user-facing name is shortened.

---

## 0. Prerequisites (read this first)

**Hardware / topology.** The scripts hardcode a 2-node topology:

- **node 0 = `c1` = `192.168.128.31`** — the CXL **initializer** (bootstraps the
  pool) and where you run the router.
- **node 1 = `c2` = `192.168.128.32`**.

To use your own hosts, edit `NODE0_HOST=` / `NODE1_HOST=` at the top of
`launch_node.sh`, `launch_router.sh`, `clear_cache.sh`, and `run_bench.sh` (they
each hardcode the pair). Each node needs **one GPU** (the launcher pins
`CUDA_VISIBLE_DEVICES=0`).

- **CXL arm:** a DAX device at `/dev/dax0.0` visible to *both* nodes as the same
  shared pool. Change via `dev_path` in `config/cxl.base.json`.
- **NIXL arm:** an RDMA NIC. The launcher pins UCX to `mlx5_0` port 1
  (`UCX_TLS=rc`, `UCX_NET_DEVICES=mlx5_0:1`); override those env vars for a
  different device/transport.

**Software.** Work inside a Python [uv](https://docs.astral.sh/uv/) virtualenv
with `torch` (built against CUDA 13.x) and `vllm` installed from pip, and
`lmcache` installed from source (this repo). This provides the `lmcache` and
`vllm` CLIs the scripts call. Also needs `jq` and `curl` on PATH.

> The shipped scripts source a venv at `~/.virtualenvs/lmcache`; if yours lives
> elsewhere, point `VENV=` at it in `launch_node.sh`, `launch_router.sh`, and
> `lmc_bench.sh` (or activate it yourself and drop the `source` lines).

**Permissions.** Mapping `/dev/dax0.0` needs device access, so the CXL arm is
typically run under `sudo`. Note: `launch_node.sh` does **not** itself call
`sudo` or check for root — the DAX open happens inside `lmcache server`. If your
user already has read/write on `/dev/dax0.0`, `sudo` may be unnecessary.

**Ports.**

| Port | Purpose | Scope |
|---|---|---|
| 5555 | MP server ↔ vLLM (ZMQ) | localhost |
| 8090 | MP server HTTP: `/healthcheck`, `/lookup_hits`, `/clear-cache`, metrics | 0.0.0.0 |
| 8010 | vLLM OpenAI API: `/health`, `/v1/...`, `/metrics` | 0.0.0.0 |
| 8447 | **cxl arm:** `PushKVToCXL` P2P server | node↔node |
| 8500–8501 | **nixl arm:** control RPC (8500) + NIXL handshake (8501) | node↔node |
| 8000 | router public OpenAI endpoint | clients |

Open **8447** between the nodes for the `cxl` arm, or **8500–8501** for the
`nixl` arm.

---

## 1. Files at a glance

| File | What it does |
|---|---|
| `launch_node.sh <0\|1> <cxl\|nixl>` | Bring up ONE node: MP server + vLLM, wait for both healthchecks, print a READY banner. Foreground; Ctrl-C tears both down. |
| `launch_router.sh [strategy] [w_prefix] [w_load]` | Bring up the KV-aware router in front of both nodes (run on node 0). |
| `run_bench.sh <label> [seed]` | The TTFT benchmark (wraps `bench_ttft.py`); writes `results/results-<label>.csv`. |
| `lmc_bench.sh <host> <vllm_port> <lmc_port> [seed]` | LMCache's built-in `lmcache bench engine` load driver against one endpoint. |
| `clear_cache.sh` | POST `/clear-cache` to each node (clears **L1/DRAM only**, not the CXL pool). |
| `send_request.py <host:port>` | Fire a single length-controlled completion at one endpoint (manual smoke test / warm a prefix). |
| `config/` | Adapter specs (`{cxl,nixl}.{base,node0,node1}.json`) + vLLM engine config (`lmcache.yaml`). |
| `bench_h2d.py`, `bench_cxl_write.py`, `probe_nixl_rdma.py`, `gen_prompt.py` | Standalone microbenchmarks / probes (§6). |
| `logs/`, `results/` | Log and CSV output (both gitignored). |

---

## 2. Run an arm end to end

### Step 1 — launch both nodes

On **both** machines (**node 0 first** for the `cxl` arm — it initializes the
pool; for `nixl` either order works once both are up):

```bash
# on c1 (192.168.128.31):
sudo ./launch_node.sh 0 cxl      # or: ... 0 nixl
# on c2 (192.168.128.32):
sudo ./launch_node.sh 1 cxl      # or: ... 1 nixl
```

Each invocation:
1. merges `config/<mode>.base.json` + `config/<mode>.node<N>.json` with `jq` and
   substitutes the two host IPs, producing the `--l2-adapter` JSON;
2. starts the MP server (`lmcache server`, teed to terminal **and**
   `logs/node<N>-<mode>.log`), waits for `:8090/healthcheck`;
3. starts vLLM (`vllm serve`, logged to `logs/node<N>-vllm.log`), waits for
   `:8010/health`;
4. prints a `nodeN READY` banner.

**"Up"** = both terminals show the READY banner. Leave them running; Ctrl-C in a
terminal tears that node's MP server + vLLM down.

Key flags the launcher sets (edit `launch_node.sh` to change): the MP server runs
`--l1-size-gb 92 --eviction-policy LRU`, and vLLM runs
`--no-enable-prefix-caching` (so the *only* cache is LMCache) `--enforce-eager
--dtype float16`. `PYTHONHASHSEED=0` is exported so token→chunk hashing is
byte-identical across nodes (required for cross-node hits).

> **Store policy.** As shipped, `launch_node.sh` uses
> `--l2-store-policy default` (the `lazy` line is commented out). With `default`,
> node 0 proactively stores its KV to L2; with `lazy`, a chunk reaches the peer
> tier only when that peer asks for it. To reproduce the "pull-only" behavior,
> set `L2_STORE_POLICY="lazy"` near the top of `launch_node.sh`.

### Step 2 — run the TTFT benchmark

From any host that can reach both vLLMs on `:8010`:

```bash
./run_bench.sh cxl          # -> results/results-cxl.csv
./run_bench.sh nixl 1234    # -> results/results-nixl.csv, reproducible seed
```

`run_bench.sh` sweeps prompt lengths `1000 3000 9000 27000`, `--repeat 3`. For
each `(length, trial)` it generates a fresh deterministic prompt and issues four
single-token completions, recording:

| CSV column | Meaning |
|---|---|
| `label`, `prompt_tokens`, `repeat` | run tag / sweep point / trial index |
| `ttft_a_local` | TTFT on node A after A already stored the prompt (local hit) |
| `ttft_b_cold` | TTFT on node B's **first** request — triggers the cross-node fetch |
| `ttft_b_warm` | TTFT on node B's **second** request — the post-fetch hit |

TTFT is in seconds. The interesting number is `ttft_b_cold` (the cross-node
fetch cost of that arm) vs `ttft_b_warm` (steady-state). `bench_ttft.py` also
prints a per-length **median** table to stderr.

### Step 3 (optional) — reset between arms

```bash
./clear_cache.sh            # clears L1 (DRAM) on both nodes
```

This does **not** clear the shared CXL pool. To cold-start CXL, restart node 0
(which has `"initialize": true`): `sudo ./launch_node.sh 0 cxl`.

### Switching arms

Relaunch **both** nodes with the other mode as the 2nd arg (Ctrl-C the running
ones first), then re-run the benchmark.

### Fire a single request by hand (`send_request.py`)

For a quick manual poke at one endpoint — a smoke test, or to warm a specific
prefix on one node before checking whether the *other* node hits it — use
`send_request.py`. It generates a prompt of an exact token length and POSTs it
to a server's `/v1/completions`, printing the raw response:

```bash
# warm a 2000-token prompt on node 0 (seeded so it's reproducible):
python send_request.py 192.168.128.31:8010 -n 2000 -s 42

# then send the SAME prompt to node 1 and watch it fetch cross-node:
python send_request.py 192.168.128.32:8010 -n 2000 -s 42
```

Args: positional `HOST:PORT`, `-m/--model` (default the harness model),
`-n/--prompt-length` (tokens, default 100), `-t/--max-tokens` (default 1),
`-T/--temperature` (default 0), `-s/--seed` (default = timestamp, printed for
replay), `--timeout` (default 120 s). With the same `-s` seed and `-n` length,
two invocations produce byte-identical prompts — that's what makes the "warm on
A, hit on B" check reproducible. Point it at the router (`<node0>:8000`) to
exercise routing instead of a single node.

---

## 3. The KV-aware router (optional)

To experiment with routing strategies, run the router on **node 0** after both
nodes are up:

```bash
./launch_router.sh                 # round_robin (default)
./launch_router.sh max_prefix
./launch_router.sh weighted 0.7 0.3
```

It listens on `:8000` and proxies `/v1/completions`, `/v1/chat/completions`,
`/v1/models`, `/health`, `/router/nodes`. Point your client at
`http://<node0>:8000` instead of a single vLLM.

Strategies (`--strategy`):

| Strategy | Routes to |
|---|---|
| `round_robin` | next node in rotation (ignores signals) |
| `random` | a uniformly random node |
| `gpu_load` | fewest running requests (KV-cache usage as tiebreaker) |
| `max_prefix` | the node with the longest cached prompt prefix |
| `weighted` | `argmax(w_prefix·norm(prefix_hit) − w_load·norm(load))` |
| `anti_affinity` | *away* from nodes that already hold the prompt (forces a cross-node fetch — useful for exercising the L2 path) |

Signals: the **prefix** signal comes from each node's LMCache `/lookup_hits`
(`:8090`) — the router tokenizes the prompt in-process to match vLLM and uses the
returned `best_hit_chunks`. The **GPU-load** signal is scraped from each node's
vLLM `/metrics` (`:8010`) on a background timer (`vllm:num_requests_running`,
`vllm:*_cache_usage_perc`).

---

## 4. Sophisticated workloads (`lmc_bench.sh`)

`run_bench.sh` is a minimal cold/warm TTFT probe. For concurrency, QPS, and
realistic multi-turn workloads, use `lmc_bench.sh`, which wraps LMCache's
built-in `lmcache bench engine` driver against one endpoint:

```bash
# against node 0's vLLM (8010) using its MP server (8090) to auto-resolve tokens/GB:
./lmc_bench.sh 192.168.128.31 8010 8090

# sweep document length + concurrency:
WORKLOAD=long-doc-qa DOCUMENT_LENGTH=8000 NUM_INFLIGHT_REQUESTS=16 \
    ./lmc_bench.sh 192.168.128.31 8010 8090

# multi-round-chat workload:
WORKLOAD=multi-round-chat QPS=2 DURATION=120 \
    ./lmc_bench.sh 192.168.128.31 8010 8090
```

> Pass **8090** (the MP HTTP port) as the third arg — the in-script example that
> shows `9000` is a typo.

Point the first arg at the **router** (`<node0> 8000 8090`) to benchmark through
the routing layer instead of a single node. Overridable workload knobs (with
defaults) are listed in the `lmc_bench.sh` header: `KV_CACHE_VOLUME=64`,
`DOCUMENT_LENGTH=30000`, `QUERY_PER_DOCUMENT=2`, `NUM_INFLIGHT_REQUESTS=1`, and
the `multi-round-chat` set (`SHARED_PROMPT_LENGTH`, `CHAT_HISTORY_LENGTH`,
`USER_INPUT_LENGTH`, `OUTPUT_LENGTH`, `QPS`, `DURATION`). Output lands in
`results/bench_results.csv` + `results/bench_summary.json` (per-request rows +
aggregate p50/p90/p99 TTFT, throughput, token totals).

**Ideas for labmates extending this:** add new `--strategy` values in
`router.py`; sweep `chunk_size_bytes` / `region_size` in `config/cxl.base.json`;
vary `--l1-size-gb` (nixl arm: keep it a multiple of 2 MiB); enable
`LMC_PROFILE=1` in `launch_node.sh` to get the per-request `PROFILE` stage
breakdown in the MP server log.

---

## 5. Config files & what to edit

All under `config/`. `launch_node.sh` deep-merges `<mode>.base.json` with the
per-node override and substitutes `NODE0_HOST`/`NODE1_HOST`.

**`cxl.base.json`** (shared): `dev_path` (`/dev/dax0.0`), `chunk_size_bytes`
(32 MiB), `region_size` (256 MiB), `pool_size_override` (128 GiB),
`generation`, `max_nodes: 2`, and the geometry (`model_name`, `world_size`,
`kv_dtype_str`, `kv_shape`, `use_mla`, `cluster_chunk_size`).
`cxl.node0.json` sets `initialize: true` + `run_lock_manager: true` (node 0 is
the sole initializer and arbiter host); `cxl.node1.json` sets both `false`.
Both list the peer's `PushKVToCXL` URL on `:8447`.

**`nixl.base.json`** (shared): `type: nixl_peer`, `device: cpu`,
`nixl_backends: ["UCX"]`, `control_timeout_ms`, `lease_ms`, plus the same
geometry block. `nixl.node{0,1}.json` set the control/init bind URLs (`:8500` /
`:8501`) and the peer's control/init URLs.

**`lmcache.yaml`**: the vLLM-side engine config — one line, `chunk_size: 256`
(the **token** chunk size; matches `cluster_chunk_size` in the specs).

**To adapt to your setup**, edit:
- host IPs → the `NODE0_HOST`/`NODE1_HOST` vars in the shell scripts;
- `model_name` (and matching `kv_shape`/`kv_dtype_str`/`use_mla`) in both
  `*.base.json`, plus `MODEL=` in `launch_node.sh` / `launch_router.sh`;
- `dev_path` in `cxl.base.json` for a different DAX device;
- pool sizing (`pool_size_override`, `region_size`, `chunk_size_bytes`) in
  `cxl.base.json`.

The geometry fields feed the CXL `geom_hash` / NIXL page contract and **must
match across both nodes**, or cross-node hits silently fail.

---

## 6. Standalone probes (isolate a layer)

These run without vLLM and help attribute where time/bandwidth goes. Invoke with
the venv python (`~/.virtualenvs/lmcache/bin/python`).

- **`bench_h2d.py`** — host↔GPU DMA bandwidth vs payload size, comparing DRAM and
  the CXL pool (cold-by-default to defeat DDIO inflation). E.g.
  `bench_h2d.py --cxl-dev /dev/dax0.0`.
- **`bench_cxl_write.py`** — DRAM→CXL *write* bandwidth by copy method
  (`memmove` vs NT streaming stores vs torch/numpy/cuda), to see whether the
  donor write is software-fixable or the device's write ceiling. E.g.
  `bench_cxl_write.py --dev /dev/dax0.0 --chunk-mib 32 --chunks 105`.
- **`probe_nixl_rdma.py {donor|reader}`** — a bare two-node NIXL one-sided READ
  correctness + bandwidth probe (no control plane). Run `donor` on c1, then
  `reader --peer-init <c1>:9600` on c2. Sweep `--page-kib` (must match on both
  roles) to study the descriptor-size effect described in the NIXL design doc.
- **`gen_prompt.py <num_tokens>`** — emit an exact-length prompt + its token IDs
  to feed a node's `/lookup_hits` endpoint.

---

## 7. Log & result locations

| Path | Contents |
|---|---|
| `logs/node<N>-<mode>.log` | MP server (also echoed to the terminal) |
| `logs/node<N>-vllm.log` | vLLM |
| `logs/router.log` | router |
| `results/results-<label>.csv` | `run_bench.sh` TTFT rows |
| `results/bench_results.csv`, `results/bench_summary.json` | `lmc_bench.sh` output |

All `control_*` / `init_*` URLs accept either `host:port` or `tcp://host:port` —
the scheme is optional and normalized.
