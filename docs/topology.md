# Topology and scope

**tokenmill simulates one scale-up domain by default, with an optional scale-out
network tier for multi-node TP/PP/EP.**

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
| **Cerebras** CS-3 / WSE-3 | SwarmX / system I/O | Wafer-scale systems | 1 CS-3 or CS-3 cluster |

All collective formulas in [`src/hardware/cluster.rs`](../src/hardware/cluster.rs)
(`all_reduce_latency`, `ep_all_to_all_latency`, `pp_transfer_latency`) use these
fields. When no scale-out network is configured, the same ring-allreduce algebra
— `2(N−1) × (α + chunk/β)` — runs for every vendor; only the per-link bandwidth
and latency change.

## Scale-out networking

By default, `--scale-out-bw-gbps 0` preserves the legacy assumption that TP/PP/EP
run inside one uniform scale-up fabric. To model multi-node serving, set:

```bash
tokenmill --gpu b200 --model llama-70b-fp8 \
  --tp 16 --gpus-per-node 8 \
  --scale-out-bw-gbps 50 --scale-out-latency-us 5
```

The simulator then treats parallel groups larger than `--gpus-per-node` as
cross-node:

- **TP all-reduce** uses a hierarchical approximation: reduce within each node,
  all-reduce across nodes over scale-out, then broadcast within each node.
- **PP activation transfer** uses an average boundary cost. In-node boundaries
  use scale-up; boundaries between nodes use scale-out.
- **EP all-to-all** splits per-GPU expert traffic into local and remote fractions.
  Remote traffic uses scale-out bandwidth and latency.

Typical `--scale-out-bw-gbps` examples:

| Network | Usable per-GPU direction | Example flag |
|---|---:|---|
| 200 GbE / HDR IB | ~25 GB/s | `--scale-out-bw-gbps 25` |
| 400 GbE / NDR IB | ~50 GB/s | `--scale-out-bw-gbps 50` |
| 800 GbE / XDR-class IB | ~100 GB/s | `--scale-out-bw-gbps 100` |

Use the effective bandwidth available to one accelerator after NIC sharing,
PCIe, routing, and congestion, not just the switch headline.

## What's still not modelled

The simulator deliberately does **not** model:

- **Network congestion** from other tenants or jobs sharing the fabric.
- **Topology-aware placement** within the pod. TP=8 spanning two NVLink switches
  costs more than TP=8 inside one — we use a single uniform bandwidth.
- **Failure / degraded links**. Every link runs at spec.

## The one place DCN sneaks in: disaggregated prefill / decode

`--disaggregate` separates prefill and decode onto two GPU pools. The KV cache
must be transferred between them — and the two pools could be on different racks.
We model this with a separate `internode_bw` field on `ClusterConfig`,
configurable via `--internode-bw-gbps` (default 200 GB/s). The transfer cost is
added once per request, at handoff. PP transfer
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

3. **Multi-rack clusters** (e.g. 32×B200 split across 4 racks) — the scale-out
   tier captures the first-order Ethernet/IB bandwidth and latency penalty, but
   it does not model leaf-spine oversubscription, adaptive routing, congestion,
   or topology-aware rank placement.

4. **Cerebras multi-system TP** — one CS-3 has a 214 Pb/s internal wafer fabric,
   but TP across CS-3 systems crosses the external system boundary. The preset
   uses 1.2 Tb/s system I/O as a conservative cross-system bandwidth rather than
   pretending the wafer fabric extends across machines.

## Why this scope

LLM inference is often served from single-rack / single-pod configurations:
NVL72 is one rack, HGX H100 is one server, a TPU 8i pod is one unit, GroqRack is
one rack. The simulator targets these deployments first, then adds a lightweight
scale-out tier for serving studies that spill beyond one node. Full topology
simulation with congestion and route-level placement remains a different
modelling problem.
