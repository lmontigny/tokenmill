# Topology and scope

**tokenmill simulates a single scale-up domain — one rack / one pod.**
DCN (data-center network), Ethernet, InfiniBand, cross-rack traffic, and
inter-pod RDMA are out of scope.

This page documents which fabric is treated as the "scale-up" layer for each
accelerator family, why we draw the line where we do, and the few situations
where the assumption matters.

## What counts as "in scope"

For every accelerator, the simulator's `scale_up_bandwidth` and `scale_up_latency`
fields carry the vendor's scale-up fabric — the high-bandwidth, low-latency
interconnect that exists inside a single server, rack, or pod:

| Family | Scale-up fabric | Topology | Typical scope |
|---|---|---|---|
| **NVIDIA** Hopper / Blackwell | NVLink + NVSwitch (full bisection) | Fat-tree | 1 HGX/DGX (8 GPUs), or 1 NVL72 (72 GPUs) |
| **AMD** CDNA 3 / 4 | Infinity Fabric | Mesh | 1 server (8 GPUs) |
| **Google TPU** v7 Ironwood / 8t | ICI | 3D torus | 1 superpod (256 – 9 600 chips) |
| **Google TPU** 8i | ICI + OCS | Boardfly (Dragonfly) | 1 pod (1 024 chips) |
| **Groq** LPU v1 | C2C copper / fibre | High-radix mesh | 1 GroqRack / GroqPod (≤256 chips) |

All collective formulas in [`src/hardware/cluster.rs`](../src/hardware/cluster.rs)
(`all_reduce_latency`, `ep_all_to_all_latency`, `pp_transfer_latency`) use these
two fields. The same ring-allreduce algebra — `2(N−1) × (α + chunk/β)` — runs
for every vendor; only the per-link bandwidth and latency change.

## What's NOT modelled

The simulator deliberately does **not** model:

- **DCN / Ethernet / InfiniBand** between racks or pods. Cluster build-outs that
  span racks (e.g. multi-rack NVL72 + IB Quantum, GPU clouds with leaf-spine)
  pay a real cost for cross-rack collectives that we don't account for.
- **Network congestion** from other tenants sharing the fabric. Single-tenant
  pod assumption.
- **Topology-aware placement** within the pod. TP=8 spanning two NVLink switches
  costs more than TP=8 inside one — we use a single uniform bandwidth.
- **Failure / degraded links**. Every link runs at spec.

## The one place DCN sneaks in: disaggregated prefill / decode

`--disaggregate` separates prefill and decode onto two GPU pools. The KV cache
must be transferred between them — and the two pools could be on different racks.
We model this with a separate `internode_bw` field on `ClusterConfig`,
configurable via `--internode-bw-gbps` (default 200 GB/s — typical 200 Gbps
RoCE / IB). The transfer cost is added once per request, at handoff. PP transfer
between pools also uses `internode_bw` when the second stage is cross-node.

## Where the assumption breaks down

Three cases where this matters in practice:

1. **TP groups that span the OCS layer on TPU 8i Boardfly** — the simulator's
   ring-allreduce formula is correct for TP ≤ 32 (within one Boardfly group),
   but under-estimates collective cost by ~10–20% for TP > 32 (when the group
   reaches across the Optical Circuit Switch layer between groups).

2. **Very high TP on Groq** (TP=358 for llama-70b-fp8) — the per-hop α captures
   chip-mesh latency, but the simulator approximates the high-radix mesh with a
   ring algorithm. Real Groq deployments use deterministic compiler-scheduled
   dataflow that maps better than a generic ring; observed TPOT can be a few
   ms instead of the >12 ms we predict at TP=358.

3. **Multi-rack clusters** (e.g. 32×B200 split across 4 racks) — the simulator
   uses one uniform `scale_up_bandwidth` for the whole cluster, ignoring that
   cross-rack traffic should drop to `internode_bw` rates. To approximate, set
   `--internode-bw-gbps` to the actual cross-rack link rate and use `--disaggregate`
   to model the boundary; for general multi-rack TP, you'd need to extend the model.

## Why this scope

LLM inference is overwhelmingly served from single-rack / single-pod
configurations: NVL72 is one rack, HGX H100 is one server, a TPU 8i pod is one
unit, GroqRack is one rack. The simulator targets these production deployments
with kernel-time accuracy. Multi-rack training-style topologies (where DCN
dominates) are a different modelling problem — out of scope here, and well-
covered by existing tools like NVIDIA's SimAI and Google's XLA cost model.
