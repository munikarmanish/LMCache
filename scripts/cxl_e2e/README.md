# Two-node L2 end-to-end harness (CXL vs NIXL-peer/RDMA)

Brings up the LMCache MP server + vLLM on two nodes (`c1` = node 0 =
`192.168.128.31`, `c2` = node 1 = `192.168.128.32`) and benchmarks TTFT
for a cross-node KV hit. The same harness drives two L2 adapters:

| mode | L2 adapter | Tier | Data path |
|---|---|---|---|
| `cxl` | `cxl` | shared CXL pool (`/dev/dax0.0`) | load/store over CXL; cross-node `PushKVToCXL` |
| `nixl` | `nixl_peer` | remote peer's L1 over RDMA | one-sided NIXL READ over `mlx5_0` |

(The `nixl` mode selects the adapter whose spec declares
`"type": "nixl_peer"`; only the user-facing mode name is shortened.)

Both use `--l2-store-policy lazy`, so a chunk reaches a peer only when
that peer is asked for it (CXL push / RDMA pull) — never proactively.

## Files

- `launch_node.sh <node_id 0|1> <mode cxl|nixl>` — bring up one node:
  starts the MP server (output → terminal **and** log file), waits for
  its HTTP healthcheck, starts vLLM (output → log file only), waits for
  vLLM `/health`, then prints a config summary. Merges the adapter's
  base spec with the per-node override via `jq`, substitutes
  `NODE0_HOST`/`NODE1_HOST`, and passes one `--l2-adapter`. Peers are
  **not** taken from the CLI — the topology is static (the two IPs above).
- `config/` — adapter specs and the vLLM-side LMCache config:
  - `config/cxl.base.json`, `config/cxl.node0.json`, `config/cxl.node1.json` — CXL specs.
  - `config/nixl.base.json`, `config/nixl.node0.json`, `config/nixl.node1.json` — NIXL-peer specs.
  - `config/lmcache.yaml` — vLLM-side LMCache engine config (chunk size etc.).
- `run_bench.sh <label> [seed]` / `bench_ttft.py` — TTFT benchmark;
  writes `results/results-<label>.csv`.

### Logs

All logs land under `logs/`:

| File | Contents |
|---|---|
| `logs/node<N>-cxl.log` / `logs/node<N>-nixl.log` | MP server (also echoed to the terminal) |
| `logs/node<N>-vllm.log` | vLLM |

## Adapter selection

The 2nd arg to `launch_node.sh` picks the adapter:

```bash
sudo ./launch_node.sh 0 cxl     # node0, CXL
sudo ./launch_node.sh 0 nixl    # node0, RDMA peer adapter
```

For `nixl` the launcher pins NIXL's UCX transport to the direct RDMA
link by exporting `UCX_TLS=rc` and `UCX_NET_DEVICES=mlx5_0:1` (matching
`scripts/p2p`). Override either in the env to use a different
device/transport.

## Port map

| Port | Purpose |
|---|---|
| 5555 | MP server ↔ vLLM (ZMQ, localhost) |
| 8090 | MP server HTTP (`/healthcheck`, metrics) |
| 8010 | vLLM OpenAI API (`/health`, `/v1/...`) |
| 8500 | `nixl`: `NixlPeerControlServer` — lookup/unlock RPC (ZMQ REP) |
| 8501 | `nixl`: NIXL agent handshake side-channel (ZMQ REP) |
| 8447 | `cxl`: `PushKVToCXL` P2P server |

Open 8500–8501 between the two nodes for the `nixl` arm, or 8447 for the
`cxl` arm.

All `control_*` and `init_*` URLs accept either `host:port` or
`tcp://host:port` — the scheme is optional and normalized. (The shipped
config files write `control_*` with `tcp://` and `init_*` bare, but
either works for both.)

## Run an arm end to end

On **both** nodes (node 0 first — it's the CXL initializer; for `nixl`
either order works once both servers are up):

```bash
sudo ./launch_node.sh 0 nixl    # c1
sudo ./launch_node.sh 1 nixl    # c2
```

Each invocation runs the MP server + vLLM in the foreground and tears
both down on Ctrl-C. Then from anywhere that can reach both vLLMs:

```bash
./run_bench.sh nixl     # -> results/results-nixl.csv
./run_bench.sh cxl      # the CXL arm, for comparison
```
