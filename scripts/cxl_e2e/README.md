# Two-node L2 end-to-end harness (CXL vs NIXL-peer/RDMA)

Brings up the LMCache MP server + vLLM on two nodes (`c1` = node_a =
`192.168.128.31`, `c2` = node_b = `192.168.128.32`) and benchmarks TTFT
for a cross-node KV hit. The same harness drives two L2 adapters:

| L2 adapter | Tier | Data path |
|---|---|---|
| `cxl` (default) | shared CXL pool (`/dev/dax0.0`) | load/store over CXL; cross-node `PushKVToCXL` |
| `nixl_peer` | remote peer's L1 over RDMA | one-sided NIXL READ over `mlx5_0` |

Both use `--l2-store-policy lazy`, so a chunk reaches a peer only when
that peer is asked for it (CXL push / RDMA pull) — never proactively.

## Files

- `launch_mp_server.sh node_a|node_b [cxl|nixl_peer]` — start the MP
  server. Merges the adapter's base spec with the per-node override via
  `jq`, substitutes `NODE_A_HOST`/`NODE_B_HOST`, passes one
  `--l2-adapter`. Logs to `${NODE}-${L2}.log`.
- `cxl.base.json`, `cxl.node_a.json`, `cxl.node_b.json` — CXL specs.
- `nixl_peer.base.json`, `nixl_peer.node_a.json`,
  `nixl_peer.node_b.json` — NIXL-peer specs.
- `launch_vllm.sh <port> [gpu]` — start vLLM against the local MP server
  (adapter-agnostic).
- `run_bench.sh <label> [seed]` / `bench_ttft.py` — TTFT benchmark;
  writes `results-<label>.csv`.

## Adapter selection

The 2nd arg to `launch_mp_server.sh` (or the `L2` env var) picks the
adapter; it defaults to `cxl`:

```bash
sudo ./launch_mp_server.sh node_a              # cxl (default)
sudo ./launch_mp_server.sh node_a nixl_peer    # RDMA peer adapter
L2=nixl_peer sudo ./launch_mp_server.sh node_a # same, via env
```

For `nixl_peer` the launcher pins NIXL's UCX transport to the direct
RDMA link by exporting `UCX_TLS=rc` and `UCX_NET_DEVICES=mlx5_0:1`
(matching `scripts/p2p`). Override either in the env to use a different
device/transport.

## Port map (nixl_peer)

| Port | Purpose |
|---|---|
| 5555 | MP server ↔ vLLM (ZMQ, localhost) |
| 8500 | `NixlPeerControlServer` — lookup/unlock RPC (ZMQ REP) |
| 8501 | NIXL agent handshake side-channel (ZMQ REP) |

(CXL uses 8447 for its `PushKVToCXL` P2P server instead of 8500/8501.)
Open 8500–8501 between the two nodes for the `nixl_peer` arm.

All `control_*` and `init_*` URLs accept either `host:port` or
`tcp://host:port` — the scheme is optional and normalized. (The shipped
config files write `control_*` with `tcp://` and `init_*` bare, but
either works for both.)

## Run an arm end to end

On **both** nodes (node_a first — it's the CXL initializer; for
`nixl_peer` either order works once both servers are up):

```bash
# terminal 1 (per node): MP server
sudo ./launch_mp_server.sh node_a nixl_peer    # c1
sudo ./launch_mp_server.sh node_b nixl_peer    # c2

# terminal 2 (per node): vLLM
./launch_vllm.sh 8010
```

Then from anywhere that can reach both vLLMs:

```bash
./run_bench.sh nixl_peer        # -> results-nixl_peer.csv
./run_bench.sh cxl              # the CXL arm, for comparison
```
