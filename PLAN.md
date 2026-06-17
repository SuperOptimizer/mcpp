# matter-compressor++ (mcpp) — Master Plan

> **Status:** Authoritative design plan. This document supersedes the earlier day-zero plan (which assumed integer/Q14 fixed-point, cross-ISA byte identity, `-ffast-math` forbidden, u8-only, v7/`.mca` compatibility, and dynamic-append crash-safety — **all reversed**). Generated from the finalized design decisions. Planning only; no implementation code yet.
>
> **Adversarially reviewed** (see `PLAN_REVIEW.md`). Dispositions folded in: τ max-error is **best-effort, not guaranteed**; `std::mdspan` → in-house **`mcpp::mdview`** (GCC 15 lacks `<mdspan>`); **abort-and-die happy-path-only** is the deliberate failure posture **except streaming I/O, which is robust**; **no cache pinning** (allocate must-keep data separately); BSP is a **barrier tree** with a **canonical apply order** (now specified — see §1). Explicitly **dismissed**: on-disk endianness, whole-process memory budget, a formal failure contract. Deferred-to-later (non-blocking): RD-regression gates, cross-ISA tolerance derivation, VC3D migration sizing, oversize-chunk store-raw fallback.

## Table of Contents

1. [Foundations: Guiding Principles, the BSP Invariant & Shared Modules](#1-foundations-guiding-principles-the-bsp-invariant--shared-modules)
2. [Subsystem #1 — Codec](#2-subsystem-1--codec)
3. [Subsystem #2 — Archive (`.mcp`)](#3-subsystem-2--archive-mcp)
4. [Subsystem #3 — Cache](#4-subsystem-3--cache)
5. [Subsystem #4 — Streaming](#5-subsystem-4--streaming)
6. [Subsystem #5 — Sampling](#6-subsystem-5--sampling)
7. [Subsystem #6 — Rendering & Geometry](#7-subsystem-6--rendering--geometry)
8. [Everything Else (#7) & Quality Engineering](#8-everything-else-7--quality-engineering)


---

## 1. Foundations: Guiding Principles, the BSP Invariant & Shared Modules

### Part A — Guiding Principles & the BSP Invariant

#### Executive summary

`mcpp` (matter-compressor++) is a from-scratch, modern **C++26** reimplementation that replaces large parts of `volume-cartographer/core` for the Vesuvius-Challenge 100TB+ masked micro-CT data plane: a lossy volumetric codec, an appendable mmap archive, a cache, streaming/chunked access, sampling/slicing, compositing, rendering, **and** the surface/quad/mesh geometry layer (QuadSurface, PlaneSurface, Surface, Rect3D, point grids). It does **not** replace tracer/segmentation/flattening/atlas algorithms — those keep calling `mcpp`. It is **totally greenfield**: `mcpp` defines its own ideal C++26 API, implements **no** `IChunkedArray` shim, and VC3D will be modified to call `mcpp`. The VC core interface map is reference for real-world conventions (column-major z-fastest chunks `data[lz + ly*cz + lx*cz*cy]`, `[z,y,x]` storage, LOD scale `1/2^L`, whole-chunk fetch — `mcpp` per-block partial decode is an upgrade), **not a contract**.

Build priority order is fixed: **codec → archive → cache → streaming → sampling → rendering → everything else.**

#### The one and only concurrency model: Bulk Synchronous Parallel

**BSP is the single concurrency model across the ENTIRE codebase.** Every subsystem — codec, archive, cache, coordinator, streaming, sampling, rendering, the frame loop — conforms to the *same* superstep shape. **No application/subsystem code contains synchronization.** No locks, mutexes, spinlocks, rwlocks, seqlocks, contention-sharding, atomics-as-synchronization, async tickets, lock-free miss rings, or concurrent-mutation apparatus *in any subsystem*. The *one* audited exception is the **phase barrier primitive itself** (which necessarily carries an acquire/release publish edge) plus the worker pool — both confined to `mcpp::os`, used by no one directly. (A plain tick counter is allowed, never as a sync primitive.)

The canonical superstep, which every component follows exactly:

1. **READ / COMPUTE-LOCAL phase.** All state is frozen/immutable. Every worker reads concurrently with **zero synchronization** — not "careful atomics," but literally *there are no writers, so there is nothing to synchronize.* Plain loads. Each worker accumulates its outputs/requests/deltas into its **own local buffer** (disjoint, unshared → no races).
2. **BARRIER** — the tick boundary. This is the *only* synchronization point in the whole system: a simple join/barrier, not lock-based protection.
3. **MUTATE phase (isolated).** A **global coordinator** gathers all local buffers; each **dumb component** applies its batched deltas as a **pure transform** — `(frozen old state + gathered deltas) → new state`, reading nothing else that is mutating. A single logical mutator per component.
4. **BARRIER**, re-freeze, repeat.

**Dumb components.** A component exposes lock-free reads plus a single batched mutate, e.g. `cache.apply(std::span<const FreshPair>)`. Components **never** reach into other components, own threads, coordinate, track external state, decode/fetch on their own, or classify. The coordinator and IO workers own all of that.

**Divide-and-conquer within a phase.** The only threading that exists is embarrassingly-parallel **fork-join map-then-barrier**: split work across workers over disjoint partitions, each writes only its own local buffer, join. No mutual exclusion needed — partitions are disjoint and nothing shared mutates mid-phase. This is not "multithreading code" in the dangerous sense.

```cpp
// The shape every subsystem reuses — illustrative only.
template <class Component, class Delta>
void superstep(Component& c, std::span<Worker> workers) {
  for (auto& w : workers) w.read_and_accumulate_local();   // 1. frozen reads, local deltas
  barrier();                                                // 2. the ONLY sync point
  c.apply(gather<Delta>(workers));                          // 3. single batched pure mutate
  barrier();                                                // 4. re-freeze
}
```

**Why BSP wins on all four axes.** *Correctness:* data races are designed *out* — given correct disjoint partitioning and a correct barrier, there is no concurrent mutation to race (TSan is a backstop, not a proof: a partition-boundary off-by-one is still a real bug, caught only if the overlapping write executes — so we add adversarial partition/barrier-timing fuzzing under TSan). *Simplicity:* no lock ordering, no memory-ordering reasoning, no contention tuning, far less code. *Speed:* the read phase is plain full-bandwidth memory access with zero atomic/lock overhead on the hot path. *Generality:* one mental model — read-frozen / accumulate-local / barrier / mutate-isolated / barrier — for the entire system.

**Determinism note.** f32 compute is non-deterministic, **but** the *coordination* (which deltas applied in what batched order) is deterministic and testable because it is a pure transform — even though decoded values are not bit-reproducible. **This determinism holds only at a fixed worker count and partition geometry** unless we impose a canonical order — see below.

#### BSP composition: it is a barrier *tree*, not one flat loop (review — must get right)

The headline "one superstep / a single barrier as the only sync point" is a *simplification*. The real system is a **barrier tree**: a render tick's MUTATE phase *contains* nested fork-join sub-phases — the streaming superstep (parallel per-chunk GETs), the codec superstep (parallel per-block decode), the cache `apply()`, and a spatial-tree rebuild. BSP is therefore **recursive / multi-clock**, and this is the single most important thing to specify correctly (it is core to the whole design). The rules:

1. **Nesting is explicit and bounded-depth.** Outer loop = the interactive frame/tick (or the offline drive-to-completion loop). Inner supersteps (stream → decode → apply) run *within* the outer mutate phase, each its own fork-join with its own barrier. A barrier only ever synchronizes the workers of *its own* fork-join; an inner barrier is not "the" sync point, just *a* local join.
2. **The render loop and the IO loop run at different clocks (a pipeline).** IO/fetch spans multiple ticks; fetches issued in the mutate phase of tick *T* are harvested at a later thaw (tick *T+k*). "No writers during read" means: **no writer to any slot a frozen reader can observe** — IO workers write only not-yet-occupied slots, flipped to visible exactly at the barrier. The cross-loop handoff (render misses → coordinator request set → IO results → `apply`) is the defined seam.
3. **One shared thread pool, owned by the top-level driver.** Nested fork-joins draw from the same pool (no pool-per-subsystem, no oversubscription). The pool + the barrier primitive are the *only* concurrency primitives, confined to `mcpp::os`; no subsystem/application code contains synchronization.
4. **Canonical merge/apply order.** To keep "coordination is deterministic" true regardless of worker count or partition geometry, the coordinator sorts the gathered delta set into a **canonical order** (e.g. by `BlockKey`/`linear_slot_index`) before `apply()`. Without this, equal-generation eviction and apply order depend on scheduling — so the canonical sort is **required**, not optional.
5. **The coordinator is a first-class component** (its own module, API, and milestone), not an implicit actor — it owns the gather/dedup/dedup-vs-in-flight/canonical-order/dispatch logic and the cross-tick in-flight set.

#### Project posture

- **Greenfield, no compatibility.** No v7/`.mca` format compat, no `IChunkedArray`, no shims. `mcpp` defines its own ideal API; callers adapt.
- **f32 compute everywhere.** Every input dtype loads → f32 → DCT/quant/entropy in f32 → decode in f32 → round-to-nearest + clamp to output dtype. **No integer compute, none whatsoever.** One generic separable-DCT kernel body. `-ffast-math` (incl. `-ffinite-math-only`) is an *expected, supported* build mode — FMA contraction, reassociation, fast intrinsics allowed in the hot path.
- **Tolerance-based correctness, no cross-ISA byte identity.** All codec tests are PSNR/MAE/max-error/SSIM bounds; **zero** golden-byte codec tests; no cross-fleet hash-dedup. (Archive-*structural* tests stay exact — they hash the bytes *this build* wrote, not a canonical form.)
- **No external heavy deps.** **No OpenCV, no Boost.** We write our own spatial trees and test framework. Vendored deps only, kept minimal.
- **Exercise genuinely-useful C++26** — `mcpp::mdview`, concepts, `consteval`, `std::expected`, `std::variant` dispatch, ranges — for less/simpler/faster code and extendable, generic, easy-to-use APIs. Thorough testing/fuzzing/coverage/regression/perf-benchmarking/CI-gating.
- **Scrub discipline.** Scrub float input at the ingest boundary once (and in fuzz-hardened corrupted-archive decode), then trust within the fast-math hot path.

#### KEPT invariants (hard, inherited)

- **Air/fill bit-exactness** — enforced by an **integer mask compare post-decode, OUTSIDE the f32 transform.** Air voxels (mask=0) are forced to 0 by the decoded mask; material is clamped so it never becomes 0. Harmonic SOR air-fill exists only to shape DCT energy on the encode side.
- **No cross-LOD dependency** — each LOD decodes independently; decimation propagates the zero-mask (mean over nonzero children only).
- **Max-error bound in VOXEL UNITS for surfaces (BEST-EFFORT, not a hard cross-decoder guarantee).** Explicit residual correction *targets* `|orig−decoded| ≤ τ` on every material voxel, computed against the **encoder's own reference decode**. Because decode is f32 + `-ffast-math` and non-deterministic across ISA/opt, a *different* decoder may diverge by δ — so this is **best-effort**, not a guaranteed bound on the user's decoder (review C1; the original's hard guarantee held only via an integer DCT, which mcpp dropped). Useful for parametric surfaces (τ≈0.05–0.1 voxel target); do **not** advertise as "guaranteed.")
- **Sparse-static archive mapping** — static sparse mapping with dead-simple crash-safety (replaces commit-word ordering).
- **README performance floor** (match/beat): cold single-block decode ~9 µs (invDCT 52% + range coder 41%); 1024² quad slice ~1.5 ms; 9-step trilinear composite ~8.4 ms @ 8 threads dense; partial-fetch ~96× byte reduction; rate-target within ~5%; iso-rate PSNR leads the c3d wavelet baseline by +1.4–3.7 dB.

#### DROPPED invariants (do NOT design around these, do NOT revert)

- **Cross-ISA byte-identical bitstream** — GONE. It required an integer-only hot path, incompatible with f32 + fast-math. All archives are non-deterministic across ISAs.
- **Integer / fixed-point hot paths (Q14, per-ISA s16)** — GONE. f32 everywhere; f16 is the only possible future narrow path (opt-in, native-f16 hardware only, via the `compute_t` seam).
- **Lossless / bit-reversible compression** — not a goal. DCT is not bit-reversible; archival `q≈0.5`/`τ=1` already beats zstd-19 ~1.96×.
- **v7 / `.mca` format compatibility** — none.
- **Deterministic ML sampling (`sample_boxes`)** — dropped *entirely*. No random box sampling, no sampling RNG (no xorshift64 seed), no deterministic-sampling invariant, no fraction array.
- **Crash-safe commit-word ordering** — replaced by the static sparse mapping + dead-simple crash-safety above.

---

### Part B — Shared Foundations

These are the cross-cutting modules every subsystem builds on. All are header/`consteval`-testable and dependency-light.

#### `mcpp::fpbits` — NaN/Inf detection under fast-math

`-ffast-math`/`-ffinite-math-only` fold `std::isnan`/`std::isinf`/`x != x` to `false`, making them useless. `mcpp` must detect NaN/Inf via **bit-twiddling on the IEEE-754 representation** — `std::bit_cast` to `u16`/`u32`/`u64` then exponent/mantissa masks — which the optimizer cannot elide. Centralized in one shared module, `constexpr`/`consteval`-testable:

```cpp
namespace mcpp::fpbits {
  constexpr bool is_nan(float x) noexcept;     // bit_cast<u32>, exp all-ones & mant!=0
  constexpr bool is_inf(float x) noexcept;
  constexpr bool is_finite(float x) noexcept;
  enum class fpclass { zero, subnormal, normal, inf, nan };
  constexpr fpclass classify(float x) noexcept;
  constexpr float   sanitize(float x) noexcept; // ingest-boundary scrub
}
```

Used by the **scrub discipline**: sanitize float input once at the ingest boundary (and during corrupted-archive decode), then trust the fast-math hot path.

#### Strong types + dtype vocabulary

The supported dtype set is the eight **non-64-bit** types: `u8, u16, u32, s8, s16, s32, f16, f32`. (`u64/s64/f64` are deferred — a non-breaking later addition = more `open()` visitor arms + f64 `consteval` tables.) `u8` is the speed-critical main case and stays fastest. Strong types wrap LOD levels, block/chunk coordinates, voxel-unit error τ, and dtype tags so they cannot be transposed silently. Dtype is **fully orthogonal** to use case: any of the 4 closed use cases × any of the 8 dtypes, all from one generic `codec<T, Rank, C, ComponentTransform>`.

#### `mcpp::mdview` volume views

> **Toolchain note:** GCC 15's libstdc++ does not yet ship `<mdspan>`, and the both-compiler CI gate requires it. mcpp therefore uses its **own** small non-owning multidimensional view, `mcpp::mdview<T, Rank, Layout>` (pointer + extents + custom layout, fixed-rank 2D/3D, `constexpr`-clean, ~150 LOC, zero dependency) — consistent with the project's write-our-own posture (no Boost/OpenCV, own spatial trees, own test framework). It is not `mcpp::mdview`, but our usage is narrow (fixed rank, one custom z-fastest layout) and was wrapped behind `mcpp::` types anyway.

Volumes and surface grids are first-class **`mcpp::mdview`** views with a custom layout policy matching VC conventions (`[z,y,x]`, z-fastest, column-major `data[lz + ly*cz + lx*cz*cy]`). Rank is a template parameter, so the separable DCT is *one* source folded over block extents:

```cpp
template <class T, std::size_t Rank>
using volume_view = mcpp::mdview<T, Rank, mcpp::zfast_layout>;
// 3D: DCT(z)→DCT(y)→DCT(x) over 16^3; 2D: DCT(v)→DCT(u) over 64^2 — same kernel, one fewer axis.
```

Block side is `consteval` per rank: `block_side<3>() == 16`, `block_side<2>() == 64`; each block is **4 KB** uncompressed (= one cache-arena entry / one page).

#### Portable f32 SIMD seam

One f32 SIMD abstraction; **width is a compile-time property**. `lanes_for<T>()` and the kernel width derive `consteval` from the (f32) compute type, so we instantiate the widest target SIMD per build. Target ISAs: **AVX2, AVX-512, NEON** (note AVX-512 is de-prioritized per original findings — consumer fused-off / Zen4 double-pumps 256). DCT-64 must use a **fast factored algorithm** (recursive even/odd butterflies), never a dense 64×64 matmul, or it dominates the 2D path.

#### `std::expected` error model

All fallible boundary operations (`open`, archive IO, decode of possibly-corrupt data) return `std::expected<T, mcpp::error>` rather than throwing or panicking. The hot per-voxel interior assumes already-scrubbed, already-validated inputs (no per-voxel error checks). `mcpp::error` is a small enum + context, not an exception hierarchy.

#### Concepts vocabulary

Concept-constrained policies keep the generic core honest and extensible. The key one is `ComponentTransform`: a pluggable, concept-constrained cross-component policy.

```cpp
template <class P, class T, std::size_t C>
concept ComponentTransform = requires { /* forward/inverse over C planes, per-plane quant hints */ };
```

Only **two** transforms ship at launch — **Identity** (C=1 grayscale CT volume; C=3 uncorrelated xyz surface, per-plane independent) and **YCoCg** (C=3 correlated RGB, the sole correlated case, chroma may be lower-q). The policy stays pluggable for extensibility (e.g. a future normal-map transform) but **no speculative transforms** are added.

#### `consteval` tables + `compute_t` f16 future-seam

DCT basis tables (16-pt and 64-pt), dead-zone quant ladders, and per-(rank,dtype) context priors are all **`consteval`-derived**, so DCT-16 and DCT-64 are just different basis lengths of one kernel. The compute type goes through a `compute_t<T>` seam that is **`f32` for every type today**:

```cpp
template <class T> using compute_t = float; // f16 is the ONLY future narrow path,
// opt-in, native-f16 HW only (ARMv8.2+ NEON-FP16 / AVX-512-FP16), via this seam. NO s16/s32/fixed-point ever.
```

#### `open()`-time `std::variant` dtype dispatch

Dtype/rank/component-count are resolved to the correct monomorphized `codec<T, Rank, C, Transform>` instantiation **exactly once** at the `open()`/header-read boundary, via a `std::variant`/visitor over the `(dtype, rank, components)` cross-product (mechanical generated dispatch). After that point the fast path runs with **no per-voxel dispatch**. Dtype affects only I/O conversion (load→f32) and round/clamp at the edges; the entire transform/quant/entropy interior is dtype-agnostic f32.

#### `mcpp::os` platform seam

A thin platform seam isolates mmap, file IO, page-size, threading primitives (fork-join workers + barrier only), and CPU feature detection. Platform priority is **Linux > macOS > Windows** — Linux is the primary target and gets the deepest tuning; macOS second; Windows supported behind the seam.

#### Repo / build / toolchain layout

- **CMake with presets.** Presets cover `{Debug, RelWithDebInfo, Release}` × target ISA, plus instrumented (profile-gen) and optimized (profile-use) variants.
- **Toolchains:** **Clang 21** and **GCC 15** (both C++26-capable). C++26 modules **where both compilers support them cleanly**, otherwise headers — chosen pragmatically per component given Clang21/GCC15 maturity, not dogmatically.
- **Vendored deps only, minimal — NO OpenCV, NO Boost.** Spatial trees (octree for 3D, quadtree for 2D node-trees) and the test framework are **ours**.
- **PGO + ThinLTO are NEVER enabled without profiles.** Optimized builds require a captured profile from the instrumented build (codec encode/decode + cache + sampler workloads); an optimized preset with no profile present is a build error, never a silent fallback.

---

## 2. Subsystem #1 — Codec

### 1. Codec Subsystem

The codec is the foundation of mcpp and is built first (`codec -> archive -> cache -> streaming -> sampling -> rendering`). It is a single generic `codec<T, Rank, C, ComponentTransform>` template that monomorphizes the entire encode/decode pipeline at compile time, dispatched once at `open()`-time over the dtype × rank × component cross-product. Every numeric stage computes in **f32** — there is no integer compute anywhere in the transform/quant/entropy interior; integer dtypes are pure I/O conversions at the edges. `-ffast-math` (including `-ffinite-math-only`, FMA contraction, reassociation) is a supported, expected build mode. There is **no cross-ISA byte identity** and all codec correctness is tolerance-based (PSNR/SSIM/MAE/L∞), never golden-byte.

#### Scope: the closed 4 × 8 matrix

Exactly four use cases, each independently combinable with any of eight dtypes (`u8/u16/u32, s8/s16/s32, f16/f32`). Dtype is fully orthogonal to use case.

1. **3D grayscale volume** (THE main case) — `codec<T, 3, 1, Identity>`, 3D DCT-16 over 16³, C=1. u8 primary, speed-critical; drives all downstream subsystems and the perf floor. Both quality regimes (perceptual dial + max-error).
2. **2D grayscale image** — `codec<T, 2, 1, Identity>`, 2D DCT-64 over 64², C=1. Perceptual dial.
3. **2D RGB image** — `codec<T, 2, 3, YCoCg>`, 2D DCT-64 64², C=3, YCoCg decorrelate then per-plane DCT (chroma may use lower q). Perceptual dial.
4. **2D parametric surface** — `codec<f32, 2, 3, Identity>`, 2D DCT-64 64², C=3, independent per-plane DCT (x≠y≠z fields, mutually independent → NOT YCoCg). `grid[v][u] = {x,y,z}` subvoxel coords for trilinear volume sampling. **Hard max-error-bounded mode in voxel units** (τ≈0.05–0.1), with predictor/residual against the grid's affine frame.

Prunings (less code): no 3D multichannel (volumes are always C=1); only two `ComponentTransform`s at launch — `Identity` and `YCoCg`; no 64-bit dtypes yet (deferred, non-breaking later additions = more visitor arms + f64 consteval tables); no adaptive block sizes, no lapped transforms, wavelets rejected as empirically worse.

#### Module layout

- `mcpp::fpbits` — shared IEEE-754 bit-twiddling (`is_nan/is_inf/is_finite/classify/sanitize`) that the `-ffast-math` optimizer cannot elide (`std::isnan` folds to false under `-ffinite-math-only`). `constexpr`/`consteval`-testable. Used to **scrub float input once at the ingest boundary** (and in fuzz-hardened corrupted-archive decode), then trusted in the hot path.
- `mcpp::dct` — one rank-generic separable f32 DCT kernel body, instantiated per `(rank, length)`. `consteval` basis tables per length: 16-pt (3D) and a **fast-factored** 64-pt (recursive even/odd butterflies, NOT a dense 64×64 matmul, or it dominates the 2D path).
- `mcpp::quant` — band-weighted dead-zone forward/inverse quant; `consteval` per-`(rank, dtype)` quant ladders and reconstruction offsets.
- `mcpp::entropy` — adaptive binary range coder with `consteval` per-`(dtype, rank)` trained priors (q-interpolated).
- `mcpp::transform` — `ComponentTransform` policies (`Identity`, `YCoCg`), concept-constrained and pluggable.
- `mcpp::air` — two-level mask-aware air handling + harmonic SOR air-fill (f32, encode-side energy shaping only).
- `mcpp::deblock` — decode-side deblocking, with separate 2D-64 params (bigger blocks, different ringing than 16³).
- `mcpp::codec` — the outer `codec<T,Rank,C,ComponentTransform>` orchestration + `open()`-time variant/visitor dispatch.

#### Compile-time width, open-time dispatch

```cpp
template <std::integral_or_floating T, int Rank, int C, ComponentTransformPolicy XF>
class codec;                                   // monomorphizes whole encode/decode path

consteval int block_side(int rank);            // block_side(3)==16, block_side(2)==64
template <class T> using compute_t = float;    // f32 everywhere; f16 is the only future seam
template <int Len> consteval auto dct_basis();  // 16-pt and 64-pt tables

// One generic separable kernel; 2D = 3D folded over one fewer axis.
template <int Rank, int Len>
void forward_dct(mcpp::mdview<float, /*rank*/, mcpp::zfast_layout> blk) noexcept;

// Mechanical generated dispatch at the header-read boundary, then no per-voxel dispatch.
using AnyCodec = std::variant</* dtype × rank × C × XF arms */>;
AnyCodec open(Header const& h);                // resolve once via std::visit
```

Block = 4 KB uncompressed working set = one cache-arena entry: 16³ = 4096 voxels (3D) and 64² = 4096 samples (2D). `compute_t<T>` is `float` for all current dtypes; the seam exists so f16-on-native-f16-hardware (ARMv8.2+ NEON-FP16, AVX-512-FP16) can be added later without touching the core.

#### Pipeline (per block)

Encode: load dtype → f32 (scrub once) → optional `ComponentTransform` (YCoCg for RGB) → air-fill value-0 voxels via harmonic SOR (red-black, ω≈1.6, coarse seed, padded fill buffer — shapes DCT energy only) → separable f32 DCT → band-weighted dead-zone quant (informed by original tuning: dead-zone fraction ≈0.80, HF exponent ≈0.65, mid-bin reconstruction offset) → adaptive binary range coder with `consteval` priors → optional max-error residual correction pass.

Decode: range-decode → inverse quant (fused decoder bias) → inverse f32 DCT → inverse `ComponentTransform` → deblock → round-to-nearest + clamp to output dtype. The integer air mask is decoded separately and **forces air voxels to exactly 0 outside the float transform**, preserving air/fill bit-exactness.

#### Two quality regimes

- **(A) Perceptual quality-dial** (cases 1,2,3): q-step + preset ladder, evaluated by PSNR/SSIM.
- **(B) Best-effort error-bounded, voxel-unit τ** (case 4, and available to case 1): explicit residual-correction pass *targets* `|orig − decoded| ≤ τ` on every material voxel, computed against the encoder's reference decode. For parametric surfaces, τ is in voxel units and residuals are taken against the affine-frame predictor (don't DCT raw world coords — store the small residual vs. the fitted low-order surface). **Caveat (review C1):** unlike the air mask (an integer compare outside the float path, genuinely fast-math-proof), the residual is reconstructed through the same non-deterministic f32 inverse transform, so a different decoder may breach τ by δ. This is **best-effort**, not guaranteed; acceptable per project decision.

#### Invariants upheld

- **BSP throughout**: per-block encode/decode are pure transforms over disjoint partitions (divide-and-conquer fork-join, each worker writes only its own local buffer). No locks/atomics/mutexes anywhere. The block is the natural BSP work unit feeding the coordinator's mutate phase downstream.
- **Air/fill bit-exactness** via integer mask compare post-decode, strictly outside the f32 transform. Material is clamped to [low,high] so it never collapses to the air sentinel; decimation propagates the zero-mask (mean over nonzero children only).
- **No cross-LOD dependency.**
- **fpbits scrub-at-ingest**, trust-in-hot-path discipline.
- **Tolerance-based correctness only** — zero golden-byte codec tests; differential test vs. the original C codec is also tolerance-based. (Archive-structural tests stay exact, hashing the bytes this build wrote via xxh3-64.)

#### Perf floor (match or beat the original)

Cold single-block decode ≈9 µs; decode dominated by invDCT (~52%) + range coder (~41%); encode by coef coder (~36%) + fwdDCT (~32%) + mask (~16%). Rate-target within ~5%; iso-rate PSNR leads the c3d wavelet baseline by +1.4–3.7 dB. The fast-factored DCT-64 is mandatory to keep the 2D path competitive. (Original-tested-and-rejected, do not revisit: coeff-group/gt1/gt2 coding, RDOQ, per-block QP, 32³ blocks, lossless mode, isotropic dering, AVX-512 on the codec hot path.)

#### Milestones

1. `fpbits` module + NaN/Inf-by-bits tests under `-ffast-math`; scrub harness.
2. Rank-generic f32 DCT kernel: DCT-16 forward/inverse + roundtrip tolerance tests.
3. Fast-factored DCT-64 (butterflies) + 2D path tolerance tests vs. naive reference.
4. Band-weighted dead-zone quant + `consteval` per-(rank,dtype) ladders.
5. Adaptive binary range coder + `consteval` priors; rate-target ±5% gate.
6. Two-level air mask + harmonic SOR air-fill; air/fill bit-exact post-decode test.
7. `ComponentTransform` policies (Identity, YCoCg); RGB case end-to-end.
8. Max-error-bounded mode + affine-frame predictor/residual; parametric-surface case at τ≈0.05–0.1 voxel.
9. Deblock (separate 16³ and 64² params).
10. `open()` variant/visitor dispatch over the full 4 × 8 matrix; cross-ISA tolerance CI (AVX2/AVX-512/NEON), differential test vs. original C codec.

---

## 3. Subsystem #2 — Archive (`.mcp`)

The `.mcp` archive is the on-disk volumetric format and subsystem #2 (built immediately after the codec, before cache/streaming). It replaces matter-compressor's MCA — which used a *dynamic, append-grown* index/node-tree — with a fundamentally different foundation.

**THE design goal, stated once and built around everywhere:** a chunk lives at a *computed* slot in a *sparse mmap'd* file. Static `linear_slot_index` mapping + whole-file sparse mmap is not one option among several — it is the load-bearing assumption from which the arena, LOD layout, occupancy structure, crash-safety stance, and access API all derive. No dynamic-index path survives as a co-equal fallback.

The archive is **3D-volumetric ONLY** (decided). An archive is always exactly 3 dimensions z/y/x. There is no 2D archive, no rank-generic framing, no quadtree/octree. The codec handles 2D planes (DCT-64) but that is the codec's concern; 2D parametric surfaces get their own future `.mcs` format (the matter-compressor SEGMENTS replacement) which is DEFERRED. Archive block = 16^3, chunk = 256^3, exactly 4096 blocks/chunk, period.

#### Fixed geometry and dimension cap

- **Dims = 3 x u32, hard cap 2^20 per axis** (max volume 2^20^3), strictly enforced: assert-and-die if any dim exceeds 2^20 at create/open. 2^20 fits comfortably in u32.
- **Chunk = 256^3** = unit of I/O locality, streaming, and occupancy. One streaming fetch = one whole chunk (atomic). Chunk grid up to 4096 chunks/axis -> up to ~6.8e10 chunks at the cap (the occupancy structure must scale to that, sparsely).
- **Block = 16^3**, 256^3 / 16^3 = 4096 blocks per chunk, fixed.

#### LODs (externally provided, pre-aligned, coarse-first)

- Exactly **2x2x2 downscale per level**, LOD0 = finest, max 8 LODs.
- mcpp does **NOT** generate LOD data — it is **externally provided, always** (no internal downsample cascade).
- All LODs **pre-aligned**: lower-LOD coord = higher-LOD coord / 2 per axis (pure integer divide, no phase/offset). LOD-L dims = LOD0 dims >> L.
- Each LOD is **independently coded and independently slotted** in its own contiguous region — the no-cross-LOD-dependency invariant holds: each LOD decodes alone, a missing LOD never affects another.
- **File layout is coarse-first by position:** `[header][large user-metadata carve-out][coarsest LOD]...[LOD0 finest at end]`. This hands streaming clients the tiny coarse overview levels first, for free, with zero cleverness.

#### Static slot mapping + sparse backing

Position of any chunk is a **pure function of dims + LOD**, never a grown index:

```cpp
// rank is always 3 (z,y,x); strong types prevent axis/level mixups
[[nodiscard]] consteval/constexpr u64 chunks_per_axis(u32 dim) { return (dim + 255) / 256; }
[[nodiscard]] u64 linear_slot_index(Lod lod, ChunkCoord c) const noexcept; // region_base[lod] + in_region(c)
```

Per-LOD geometry (chunk counts, region base offsets) is **derived at open()** from dims + num_lods — dims are the single source of truth, no stored per-LOD table to desync. `global_slot = lod_chunk_base[lod] + in_lod_index`; `lod_chunk_base[]` is the cumulative chunk count, computed at open.

**Sparse-file backing:** logical file size = num_chunks x slot_stride (could be ~200GB) but real compressed data is mostly far smaller (~10GB). The OS sparse file makes gaps cost zero disk. The mechanics MUST be exactly right:
- `ftruncate(fd, logical_size)` then mmap/pwrite into slots -> sparse holes, no blocks allocated, no zeroing, instant. Reading a hole returns zeros and allocates nothing. **This is the design.**
- **NEVER** `write()` zeros to fill (that materializes real blocks).
- Single-block random access faults **only the needed pages**: read the chunk header, read the one block's offset, fault that block's pages, decode. The OS never materializes the untouched 100GB.

#### Occupancy map — 3-state, 2 bits/chunk, own sparse region

The global occupancy structure stores **2 bits per chunk**, with state values renumbered so the sparse-zero default is the *safe* state:

```
0 = DONT_KNOW / ABSENT   <- sparse-zero default; correct for streaming (not fetched) AND offline (to-be-filled)
1 = ALL_ZERO             <- queried + confirmed genuinely all-zero
2 = REAL_DCT_DATA        <- must decode
3 = RESERVED
```

Critical correction baked in: **DONT_KNOW != ALL_ZERO**. A fresh sparse-mmap'd occupancy map zero-inits to all-DONT_KNOW, which is correct — an untouched chunk never looks like REAL_DCT_DATA, and a streaming client never mistakes "not yet fetched" for "confirmed empty." DONT_KNOW exists **only** in partial/streaming archives; in an offline/complete archive ingest established every chunk as either ALL_ZERO or REAL_DCT_DATA, so DONT_KNOW never surfaces.

Storage: **ONE flat 2-bit array** indexed by the *global* `linear_slot_index` across all LODs, single header offset, living in its **own sparse-mmap'd region** so empty/streaming archives cost no disk until states are written. (Max ~17GB at the 2^20^3 cap, but real scrolls are tiny and unwritten = free.) Exposed as an `mcpp::mdview` view for clean indexed access.

This 2-bit map replaces the old 1-bit occupancy *and* the earlier "compact manifest": chunk OFFSET is pure arithmetic (`linear_slot_index`), chunk LENGTH is fixed/derivable, so **no offset/len manifest is needed**. The per-chunk material-fraction array is **dropped** entirely (no ML-sampling accelerator, no reserved header slot).

**Three-state is ONLY at chunk granularity.** Within a chunk, blocks are 2-state (ALL_ZERO or REAL_DCT_DATA) because fetching a chunk from source-of-truth is atomic — once the chunk is present there is no "don't know" about an individual block.

#### Within-chunk layout — dense-pack + offset table with sentinel

There is **NO fixed stride S, NO fragmentation**. A chunk is `[chunk header][densely-packed block payloads, back-to-back]`.

The offset table is **4097 x u32**: 4096 block entries + 1 terminator (entry 4096 = end-of-payload offset).

```
offset_table[k] = byte offset of block k's payload from chunk base
offset_table[k] = 0xFFFFFFFF  => block k is ALL_ZERO (no payload stored, decode to zeros)
offset_table[4096]            => end-of-payload terminator
```

- Locate block k: `off = offset_table[k]`; if `off == 0xFFFFFFFF` fill zeros; else payload at `chunk_base + off`, fault only those pages, decode. O(1) direct-indexed, no walk.
- Block **length is derived**, never stored: `offset_table[next_real_entry] - offset_table[k]` (the terminator gives the last block's length).
- The `0xFFFFFFFF` sentinel **is** the ALL_ZERO marker — one structure does both addressing and air-marking, at u32 granularity. No separate bitmap.
- u32 is safe: max raw chunk = 256^3 x 4B = 64MB << 4GiB, and `0xFFFFFFFF` (4GiB) is never a real offset.
- Offset table = 4096 x u32 = 16KB (<0.5% of a multi-MB chunk, negligible). Always 4096 — the 2D/256-block figure does not apply to the archive.
- This is strictly better than MCA: dense-packed (no frag), single-block random access by faulting header + that block's pages, ALL_ZERO costs only its 4-byte entry, no variable-length length-walk.

#### Chunk header layout + page alignment

At `chunk_base`:

```
[xxh3 : u64][quality : f32][flags : u32][offset_table : 4097 x u32]   then dense-packed payloads
```

- xxh3 first = fixed-position 8 bytes for verify.
- Header = 8 + 4 + 4 + 4097*4 = **16404 bytes**, then **padded so payloads start page-aligned (4KB)**.
- **Every `chunk_base` is page-aligned, AND the payload start within the chunk is page-aligned.** Clean faulting (a block's pages never pull a neighbor's tail), clean header reads, `madvise` works at chunk granularity. Padding lands in sparse holes = zero real disk. The static slot stride rounds up to a page multiple.

#### File header (drives open() dispatch)

```
magic + format_version(=8) + dtype enum {u8,u16,u32,s8,s16,s32,f16,f32}
+ components(1|3) + component_xform{Identity,YCoCg}
+ dims(3 x u32, cap 2^20) + num_lods(<=8) + quality_default(f32) + flags
+ occupancy_map_offset + user_meta_offset/len
```

The dtype/components/xform tags drive the `open()` `std::variant`/visitor dispatch over the (dtype, rank=3, components) cross-product — mechanical generated dispatch, resolved once at the header boundary, then a monomorphized fast path. Then a **large user-metadata carve-out** (keep it big), then the LOD regions coarse-first.

#### Integrity (xxh3, non-canonical)

xxh3-64 computed over the chunk's **stored bytes as written** (header + packed payloads), stored in the chunk header. **Integrity only** (disk/transit corruption) — explicitly NOT a canonical-encoding hash, because the f32 codec is non-deterministic and there is no cross-ISA byte identity. The verify tool walks the occupancy map, and for each REAL_DCT_DATA chunk recomputes xxh3 and compares. (Archive-*structural* tests stay exact: they hash the bytes this build wrote.)

#### Crash-safety — lowest priority, dead simple

Write payload to sparse slot, write occupancy entry, move on. **NO** commit-word choreography, NO flush-ordering guarantees, NO hash-validated rebuild/recovery path. Static slotting makes this trivially safe: a chunk either landed in its fixed slot or it didn't — there is no half-updated grown index to tear. A torn entry or orphaned payload on crash is acceptable; re-encode/re-run. Zero engineering effort on durability. (The xxh3 hash is for intact-archive integrity verification, a separate concern.)

#### BSP conformance

The archive obeys the governing invariant (see Architecture #1). Static slotting is precisely what makes writes contention-free: every chunk knows its slot, so there is no append serialization, no commit-word ordering, **no locks/atomics/mutexes anywhere**. The write phase is the BSP mutate phase — workers compute compressed chunks into their own local buffers during the read/compute phase, then in the isolated mutation phase each chunk is pwritten to its computed slot (disjoint slots = disjoint partitions, no sharing). The occupancy map is updated as a pure batched transform. Reads (faulting pages from a frozen file) are plain loads with nothing to synchronize.

#### Oversize chunks = impossible, assert and die

The codec is lossy-and-shrinking, so compressed <= raw chunk size always for real content. A chunk exceeding raw size is an invariant violation / corruption, not something to handle gracefully: cheap guard at write `if (payload_len > raw_chunk_bytes) abort();` (in release too).

#### Platform seam: `mcpp::os`

Sparse-file mechanics diverge across OSes, so isolate them behind a small platform layer — `mcpp::os` (map/unmap, truncate-sparse, query-holes, punch-hole, advise). Archive logic above it is platform-agnostic and assumes sparse-mmap semantics unconditionally. **The seam exists from day one**, even though Windows is built last. Priority: **Linux (reference) -> macOS (secondary) -> Windows (tertiary)**.

- **Linux (first-class, clean):** `ftruncate` + `mmap MAP_SHARED`, `SEEK_HOLE`/`SEEK_DATA` (free OS-provided occupancy query), `fallocate(PUNCH_HOLE)`, `madvise(WILLNEED/DONTNEED/RANDOM)`.
- **macOS (thin shim):** mmap+ftruncate work; SEEK_HOLE/DATA exist; hole-punch = `fcntl(F_PUNCHHOLE)`; madvise flags differ slightly; APFS sparse-capable.
- **Windows (tertiary, the real divergence, most platform code, built last):** no mmap -> `CreateFileMapping`/`MapViewOfFile`; sparse needs explicit `DeviceIoControl(FSCTL_SET_SPARSE)` *before* the gap is meaningful (else `SetEndOfFile` zero-fills on NTFS); hole query = `FSCTL_QUERY_ALLOCATED_RANGES`, punch = `FSCTL_SET_ZERO_DATA`.
- **FS caveats:** sparse on ext4/XFS/Btrfs/ZFS/APFS/NTFS; NOT FAT/exFAT; NFS varies, SMB iffy -> detect + warn/fallback.

#### C++26 surface

Exercise genuinely-useful C++26 for a less/simpler/faster, easy-to-use API:

```cpp
enum class Lod : u8 {};                         // strong type, 0..7
struct ChunkCoord { u32 cz, cy, cx; };          // strong, z/y/x

struct HeaderView { /* zero-copy typed view over the mmap'd header bytes */ };

class Mmap { /* RAII region; unmaps on destruction */ };  // RAII
class Fd   { /* RAII file descriptor / HANDLE */ };       // RAII

class Archive {
  static std::expected<Archive, OpenError> open(std::string_view path);   // variant dispatch on dtype/C/xform
  static std::expected<Archive, OpenError> create(const CreateParams&);   // ftruncate-sparse to logical size

  ChunkState occupancy(Lod, ChunkCoord) const noexcept;                   // mcpp::mdview over the 2-bit map
  std::expected<DecodedChunk, ReadError> read_chunk(Lod, ChunkCoord) const;
  std::expected<DecodedBlock, ReadError> read_block(Lod, ChunkCoord, BlockIndex) const; // faults only needed pages
  std::expected<void, WriteError> write_chunk(Lod, ChunkCoord, EncodedChunk) noexcept;  // pwrite to computed slot
};
```

**Failure posture (project decision):** mcpp optimizes for the happy path — internal-invariant violations (oversize chunk, bad state, logic errors) **`abort()` early and often, in release too**; thorough error handling comes later. The **one exception is streaming external I/O** (file-not-accessible, S3/network failure, missing/corrupt remote chunk), which must be **robust** — retry/degrade-to-hole/skip, never abort. So: untrusted external I/O is robust; internal logic is fail-fast. `std::expected` is still used at fallible boundaries where graceful handling already exists (notably streaming), but is not required everywhere yet.

Use `std::expected` for all fallible open/read/write, `std::variant`/visitor for the dtype×components dispatch, `mcpp::mdview` over the occupancy region and chunk grids, `std::bit_cast` for header/IEEE-754 reads (and NaN/Inf-by-bits per fast-math discipline), `consteval`/`constexpr` for geometry constants, RAII `Mmap`/`Fd` wrappers, and `HeaderView` as a typed zero-copy view over the mapped header.

#### Open items

Archive design is **complete**. Remaining items are implementation-time micro-details only: exact magic bytes, the per-bit meanings of the `flags` field, and the format-version constant (=8).

#### Milestones

1. **`mcpp::os` Linux backend + RAII seam** — `Fd`, `Mmap`, ftruncate-sparse, SEEK_HOLE/DATA, punch-hole, madvise; unit tests proving holes cost no disk (`stat` blocks-allocated vs logical size) and hole-reads return zeros.
2. **Header + geometry math** — `HeaderView`, strong `Lod`/`ChunkCoord`, `linear_slot_index`, per-LOD region-base derivation from dims; cap enforcement (assert-die > 2^20); page-aligned slot-stride computation. Pure-function unit tests.
3. **Occupancy map** — 2-bit flat array in its own sparse region, mcpp::mdview view, 3-state semantics; sparse-zero = DONT_KNOW verified.
4. **Within-chunk layout** — 4097-entry offset table, dense-pack writer, 0xFFFFFFFF sentinel + terminator, page-aligned payload start; single-block faulting read; xxh3 over stored bytes.
5. **`Archive::create`/`open`/`write_chunk`/`read_block`** — variant dispatch on dtype/components; oversize assert-die guard; round-trip an encoded chunk through a sparse file.
6. **Integrity + structural tests (exact)** — verify-walk over occupancy -> xxh3 recompute; LOD independence (missing LOD doesn't break others); crash-leaves-no-half-index test; sparse-disk-usage assertions.
7. **BSP write path** — parallel disjoint-slot writes with no synchronization primitives; TSan-clean.
8. **macOS shim, then Windows backend** (tertiary, last) — replicate sparse-mmap semantics behind the unchanged `mcpp::os` seam.

---

## 4. Subsystem #3 — Cache

### 3. Cache Subsystem — In-RAM Decoded-Block Cache

The cache sits between the codec/archive and all read clients (render, ML, sampling) so that fully-decoded voxel blocks are never re-decoded within a tick. It is subsystem #3 in the build order (codec → archive → **cache** → streaming → sampling → rendering). The original was a 64-shard, S3-FIFO, mmap-arena, dual-mode (MtSafe + tick-phase) cache hitting ~150M zero-copy gets/s/thread. mcpp keeps the perf target and the best-LOD probe but is **dramatically simpler**: it is a single-model BSP **dumb store** with a fixed compile-time capacity, heap slabs, and exactly one mutation entry point.

#### 3.1 Hard scope (decided, non-negotiable)

- **3D single-channel volumetric ONLY.** Rank is ALWAYS 3, C is ALWAYS 1. There is NO 2D path, NO multichannel path, ever — there is no use case. This is *not* `codec<T,Rank,C>`-style generality; the cache is generic over **dtype only**.
- **`Cache<T, Capacity>`** — dtype `T` ∈ {u8,u16,u32,s8,s16,s32,f16,f32}; `Capacity` (slab count `N`) is a compile-time constant. No `Rank`, no `C`, no `Mode` template parameter (the `Mode`/`MtSafe` axis is **dropped entirely** — see 3.4).
- **Decoded-block cache.** It stores fully-decoded, ready-to-read voxels (zero-copy `const T*` on hit). Decoded f32-nondeterminism is irrelevant here: blocks are transient and never persisted. A raw-compressed-chunk cache (so an evicted block re-decodes without re-fetching) is **deferred to the streaming subsystem**, where fetch cost dominates.

#### 3.2 Block geometry — whole pages for free

A block is `16^3 = 4096` samples (the 3D DCT-16 working set). Block size in bytes = `4096 * sizeof(T)`, a compile-time constant:

- u8/s8 → 4 KB (1 page); u16/s16/f16 → 8 KB (2 pages); u32/s32/f32 → 16 KB (4 pages).

Because `sizeof(T) ∈ {1,2,4}` and `4096 × 1` is exactly one 4 KB page, **a block is always a whole number of pages** (`sizeof(T)` pages). Consequence: no sub-page slots ever exist, page-aligned arena slots hold for every dtype, slot arithmetic is trivial, and zero-copy pointers are stable. The slot stride is `4096 * sizeof(T)`. This also kills rank/C templating — `Cache<T,Capacity>` monomorphizes the whole store, type-erased only at the public boundary (mirroring the codec's `open()` variant dispatch).

#### 3.3 Arena and map structure

- **Arena = heap-allocated slabs** (decided, supersedes the original's mmap arena). `aligned_alloc`/aligned-new, page-aligned so blocks stay page-aligned and slot arithmetic is clean and zero-copy pointers stable. **No `mcpp::os` plumbing** for the cache. We lose `madvise(MADV_FREE)` on evict, but that is irrelevant for a pure in-RAM cache — evicted slots are simply overwritten.
- **Capacity is a compile-time constant.** No resize, ever. Slab count `N` is known at compile time → fixed-size arena array, fixed-capacity key→slot map. This designs out all quiesce/remap/resize-vs-frozen-reader races. Big simplification.
- **Map = open-addressed hash table (linear probing), fixed compile-time capacity** = `N / load_factor`. Entries are `{ BlockKey key, u32 slot, u32 generation }`. Cache-friendly, no per-entry allocation, no pointer chase. (Same style the original used.)

```cpp
template <class T, std::size_t Capacity>
class Cache {                       // 3D, C=1 only; dtype-only generic
  static constexpr std::size_t kBlockBytes = 4096 * sizeof(T);   // whole pages
  // fixed heap slab array + open-addressed BlockKey->slot map + generation stamps
};
```

#### 3.4 BlockKey — packed-u64 strong type

The key is an 8-byte strong type that *is* a `u64`:

```cpp
struct BlockKey {                   // static_assert(sizeof(BlockKey) == 8);
  std::uint64_t bits;               // [ lod:3 | bz:16 | by:16 | bx:16 ] (51 used, 13 spare)
  static consteval BlockKey make(unsigned lod, unsigned bz, unsigned by, unsigned bx);
  constexpr unsigned lod()  const noexcept;   // bit-extract, inlines to nothing
  constexpr unsigned bz()   const noexcept;
  constexpr unsigned by()   const noexcept;
  constexpr unsigned bx()   const noexcept;
  constexpr std::uint64_t hash() const noexcept;   // mixes bits
  bool operator==(const BlockKey&) const = default;
};
```

Coords fit in 16 bits because dims cap at `2^20` and a 16-block divides that to `≤ 2^16`. `make()` packs and bounds-asserts at compile time; accessors are constexpr bit-extracts that inline away. Trivially copyable, register-resident like a raw `u64` but readable. `hash()` feeds the open-addressed map.

#### 3.5 Single concurrency model = BSP (MtSafe DROPPED)

The cache follows the ONE global BSP model (governing invariant). There is **NO `Mode` template parameter, NO `MtSafe` variant, NO locks, NO mutexes, NO shards-for-contention, NO atomics-as-synchronization, NO async tickets, NO lock-free miss ring**. The earlier `Cache<T,Mode>` MtSafe/TickPhase split is GONE; only the freeze/thaw model remains, because the whole codebase is BSP.

- **Freeze (read phase) = totally immutable.** Readers do pure lock-free plain-load probes. The cache mutates **nothing** — not even a miss ring. ~150M zero-copy gets/s comes from *zero* sync overhead on the hot path, not from clever locking.
- **Thaw (mutate phase) = one batched `apply()`**, then refreeze.
- Sharding, if it ever appears, is ONLY a divide-and-conquer partition (disjoint), never for lock contention.
- **Determinism note:** decoded f32 values are non-deterministic, but the *coordination* (which FreshPairs are applied, in what batched order) is a deterministic, testable pure transform.

#### 3.6 The cache is a DUMB STORE (the non-leak reframe)

Everything that is *not* "store bytes and hand back pointers" is **external** to the cache: the coordinator, IO workers, dedup, in-flight tracking, occupancy classification, work-splitting, and decode all live outside. The cache NEVER decodes, fetches, owns threads, tracks in-flight, classifies occupancy, or coordinates. Its entire mutation surface is **one call**: `apply(span<FreshPair>)`.

**Public surface (the whole API):**

```cpp
// FREEZE-phase, lock-free, plain loads:
const T* get(BlockKey k) const noexcept;            // exact LOD; miss -> nullptr

struct BestLod { const T* ptr; unsigned actual_lod; };
BestLod get_best_lod(BlockKey k) const noexcept;    // render-now-refine-later

// THAW-phase, single-caller (coordinator) only:
struct FreshPair { BlockKey key; std::variant<DecodedBlock, ZeroTag> payload; };
void apply(std::span<const FreshPair> fresh);       // batch-insert + sweep-evict + refreeze
```

- `get(k)` returns `const T*` (null on miss); the caller handles its own fallback.
- `get_best_lod(k)` is the render-now-refine-later probe (see 3.8).
- `apply()` is the single batched mutate.

The caller chooses `get` vs `get_best_lod` per call site. The cache has **no opinion** about where blocks come from.

#### 3.7 Wiring = the cache wires to NOTHING

The cache knows nothing about archives or streaming. It is constructed standalone with just capacity + dtype (`Cache<T, Capacity>`). The **coordinator** holds the cache AND the archive/streaming source *separately*; the coordinator decodes/fetches and hands finished `FreshPair`s to `cache.apply()`. Cache↔source coupling is zero.

#### 3.8 `get_best_lod` — render-now-refine-later

On a miss at LOD `L`, probe coarser LODs by halving the block coords (`coord / 2` per level — pre-aligned 2× LODs make this pure arithmetic) up to LOD 7, returning the first present block as `{ptr, actual_lod}` so the renderer knows the resolution it actually got. **Separately**, the caller still records the *fine* miss for the next thaw fill. The renderer proceeds NOW with a coarse block; the fine block arrives a tick later. Cross-LOD blocks are independent — no LOD dependency is introduced.

#### 3.9 FreshPair and the single canonical zero-block

`FreshPair = { BlockKey, std::variant<DecodedBlock, ZeroTag> }`. There is exactly **ONE** ALL_ZERO block slot in the cache; *multiple* BlockKeys map to that one address.

- `apply()` copies real bytes into a fresh slab slot for a `DecodedBlock` payload.
- For a `ZeroTag` payload it points the key's map entry at the **shared canonical zero-slot** — **no slab is consumed for air blocks**, so a mostly-air volume costs one zero-slot total.
- Decoded blocks are read-only, so sharing the zero-slot is safe (no aliasing-write hazard).
- The **coordinator** decides zero-vs-real from the archive's 3-state occupancy (ALL_ZERO → emit `ZeroTag` with no decode; REAL_DCT_DATA → decode and emit bytes; DONT_KNOW → streaming fetch then decode). The cache just stores what it is told.

#### 3.10 Eviction = NRU, re-request-implies-hot

Eviction is **NRU** (not true LRU — NRU is fine), supersedes the original S3-FIFO, and exploits data the coordinator *already has*:

- Generation stamp = the tick counter. Blocks are also stamped with their fill-tick.
- **No read-phase mutation** (BSP): the coordinator already sees every consumer's interest set each tick, so any block that is present-and-requested gets its generation **bumped via the existing touch path inside `apply()`** — there is no separate full touch-list and no per-`get` bookkeeping.
- Eviction happens **inside `apply()`** (the mutate phase): sweep the slots and reclaim those older than a cutoff, oldest-first under capacity pressure. It is a batch sweep — no per-insert bookkeeping, no queue structures.
- "Re-request-implies-hot" approximates LRU using the coordinator's interest set, with zero hot-path cost.

#### 3.11 `apply()` contract — single-caller, asserted

- Only the **coordinator** calls `apply()`, **once per tick**, at the thaw boundary where no reader is mid-frozen-read (the tick barrier guarantees this).
- **No pinning (review M1).** The cache is a uniform NRU store with no pin/no-evict set. If data must survive eviction, **allocate it separately** outside the cache — "pinned" data simply is not in the cache. (Eviction never special-cases the current interest set; keep capacity comfortably above the working set.)
- No internal mutation-path locking exists. A **debug assert** enforces no concurrent `apply`/read (single logical mutator).
- Inside `apply()`, in order: (1) batch-insert/update fresh pairs (real bytes into fresh slots; zero-tags pointed at the canonical zero-slot); (2) bump generations of present-and-requested blocks via the touch path; (3) generation-sweep evict; (4) refreeze (publish the new immutable snapshot). Mutation is single-threaded by construction.

#### 3.12 Full BSP coordinator flow (cache is the leaf)

The cache is a leaf in this loop; the GATHER / COALESCE / dedup-vs-inflight / SPLIT / decode all happen **outside** it, before `apply()`:

1. **Read phase (frozen).** Each consumer (e.g. a render worker) does lock-free `get()`/`get_best_lod()` on the blocks it needs → some hit, some miss. Each consumer accumulates **its own** miss list in its **own local buffer** — no shared, cache-owned structure; zero frozen-phase contention. The cache never reaches into requester state.
2. **Barrier (tick boundary)** — the only synchronization point.
3. **Coordinator gather + dedup.** The external coordinator (main thread, NOT the cache) collects every consumer's miss list, **merges and dedupes across consumers**, then **dedupes against in-flight work** already submitted to download/disk-decode paths in prior ticks, producing the final request set.
4. **IO workers.** The coordinator dispatches the request set to IO worker(s). Each worker classifies via archive occupancy and **decodes only the specifically-missed blocks** via per-block random access (fault only that block's pages) — **NO whole-chunk decode, NO neighbor prefetch**. Work-splitting/load-balancing is a coordinator/IO concern (disjoint partitions), outside the cache.
5. **FreshPairs.** Workers return all `{BlockKey → DecodedBlock | ZeroTag}` pairs gathered during the previous tick.
6. **Thaw.** The coordinator calls `cache.apply(span<FreshPair>)` once: batch-insert, generation-sweep-evict, snapshot-swap, refreeze.
7. **Barrier, repeat.**

The thaw boundary is an **explicit handoff**, not a background drain; misses are owned by requesters during freeze, never by the cache.

#### 3.13 What is kept vs dropped relative to the original

- **Dropped (BSP makes them moot):** sharding-for-contention, dual-mode (MtSafe), async tickets with cancel, lock-free dedup miss ring, S3-FIFO queues, mmap arena, dynamic resize, all concurrent-mutation machinery.
- **Kept/adapted:** best-LOD probe (render-now-refine-later, via `get_best_lod`); the ~150M zero-copy gets/s target (now from zero-sync reads, not locking); generation-based eviction (now NRU sweep, not S3-FIFO); 4 KB-multiple page-aligned blocks; open-addressed key map.

#### 3.14 Module layout

- `mcpp/cache/block_key.hpp` — `BlockKey` packed-u64 strong type, `consteval make`, constexpr accessors/hash, `static_assert(sizeof==8)`.
- `mcpp/cache/slab_arena.hpp` — fixed heap slab array (`aligned_alloc`, page-aligned), slot stride `4096*sizeof(T)`, canonical zero-slot.
- `mcpp/cache/index_map.hpp` — open-addressed linear-probing fixed-capacity `BlockKey → {slot, generation}` map.
- `mcpp/cache/cache.hpp` — `Cache<T,Capacity>`: `get`, `get_best_lod`, `apply`, generation/eviction sweep, freeze/thaw snapshot.
- `mcpp/cache/fresh_pair.hpp` — `FreshPair`, `ZeroTag`, `DecodedBlock` view.
- `mcpp/cache/dispatch.hpp` — type-erased public boundary over dtype (variant/visitor, mirrors codec `open()`).
- Coordinator/IO/dedup/occupancy live in their own subsystems (#4 streaming + coordinator), NOT under `cache/`.

#### 3.15 Invariants (test targets)

- Rank == 3, C == 1 always; block is always a whole number of 4 KB pages; slot stride == `4096*sizeof(T)`.
- `sizeof(BlockKey) == 8`; pack/unpack round-trips; `make()` bounds-asserts at compile time.
- Frozen-phase reads mutate nothing (TSan finds nothing; no writers exist during read phase).
- `apply()` is single-caller, once per tick; debug-assert no concurrent apply/read.
- Exactly one canonical zero-slot; many ZeroTag keys → one address; mostly-air volume → one zero-slot total.
- `get_best_lod` never introduces a cross-LOD dependency; coarse hit + fine miss recorded both happen.
- Eviction is batch-sweep in `apply()` only; present-and-requested blocks survive (NRU touch path).

#### 3.16 Milestones

1. **M1 — Key + arena scaffolding.** `BlockKey` (consteval pack, constexpr accessors, hash, `static_assert`), page-aligned slab arena, open-addressed map. Unit tests for key round-trip, slot arithmetic, page alignment across all dtypes.
2. **M2 — Frozen reads.** `get()` and `get_best_lod()` (coarse-LOD probe by coord-halving). Lock-free plain-load probe tests; best-LOD-returns-coarse + records-fine-miss test.
3. **M3 — `apply()` mutate phase.** Batch-insert, canonical zero-slot sharing, generation stamps, NRU sweep-evict, snapshot-swap/refreeze, single-caller debug assert. Eviction-under-pressure and zero-block-sharing tests.
4. **M4 — BSP loop integration.** Drive via an external test coordinator: consumers accumulate local misses → merge/dedup → dedup-vs-inflight → FreshPairs → `apply()`. Determinism test on coordination (same misses → same applied set/order), TSan clean across the full freeze/thaw cycle.
5. **M5 — Perf gate.** Benchmark zero-copy `get()` throughput against the ~150M gets/s floor; CI perf-gate within tolerance.

#### 3.17 Remaining micro-details (implementation only, not design)

Exact eviction cutoff tick count; precise slab/map sizing math and chosen load factor. No open design questions: **cache design is complete.**

---

## 5. Subsystem #4 — Streaming

Subsystem #4 (after codec, archive, cache; before sampling/rendering). Streaming turns a set of `DONT_KNOW` chunk requests into bytes-on-local-disk plus occupancy updates. It is, architecturally, **the IO-worker side of the BSP coordinator** — it owns no policy, no scheduling, no concurrency model of its own; it only executes the disjoint whole-chunk fetches the coordinator hands it during the mutate/IO phase. Everything here conforms to [[mcpp-architecture]] BSP: frozen reads, local accumulation, single isolated mutation phase, and **no multithread-safety code anywhere** (no locks/mutexes/atomics-as-sync).

#### Core model: local sparse .mcp = write-through disk cache of remote

The local sparse `.mcp` is a **partial mirror** of an authoritative remote `.mcp`. There is no v7/.mca compat and no separate "downloaded volume" concept — the local file *is* the cache of the remote, persistent across runs.

Miss-resolution flow (one granularity everywhere — per chunk):
1. A consumer block miss lands on a chunk whose occupancy is `DONT_KNOW`.
2. Fetch the **whole 256^3 chunk** from remote via one ranged GET (no per-block partial fetch — the original's ~96x byte-reduction win is deliberately sacrificed for simplicity and per-chunk occupancy consistency).
3. Unpack the packed wire bytes into the chunk's **exact static slot** in the local sparse file (slot position is pure arithmetic from the static mapping — see [[mcpp-archive]]).
4. **Flip occupancy** `DONT_KNOW -> REAL_DCT_DATA` (or `ALL_ZERO`), so re-access is purely local forever after.

One granularity governs the whole subsystem: occupancy is per-chunk, fetch is per-chunk, the I/O unit is per-chunk, and cache-miss-resolution is per-chunk.

#### Wire format: remote packed, local sparse

S3/HTTPS objects cannot be sparse and we fetch whole chunks, so the **remote** chunk is stored **packed**: a chunk header offset-table followed by only the non-`ALL_ZERO` block payloads, contiguous, no holes. The **local** `.mcp` is the sparse mmap form. Two physical layouts, **one logical static mapping**.

Client path: one ranged GET pulls the packed chunk; the client walks the offset table and unpacks each present block into its sparse slot; offset-table holes (sentinel `0xFFFFFFFF`) stay holes locally (never materialized).

```cpp
// remote chunk header (packed); offsets relative to chunk-object start
struct PackedChunkHeader {
  u32 magic;
  u32 block_count;              // blocks per chunk (static)
  // offset_table[i] == 0xFFFFFFFF  => ALL_ZERO block, no payload
  std::array<u32, /*block_count*/ N> offset_table;
};
constexpr u32 BLOCK_HOLE = 0xFFFFFFFFu;
```

#### Occupancy bootstrap: remote ships the occupancy map up front

On `open()`, **one small GET** pulls the entire remote occupancy map (2 bits/chunk — tiny and sparse; this is exactly what the 3-state occupancy design in [[mcpp-archive]] is for). After that single fetch the client knows every chunk's true `ALL_ZERO` / `REAL_DCT_DATA` state **without fetching any data** — so **air chunks cost zero network, ever**. `DONT_KNOW` is therefore purely transient: it exists only before the map loads and during the in-flight window of a fetch. The map is assumed stable for the session (remotes are mostly static); no mid-session re-fetch is planned.

#### ByteSource concept — full set at launch

The remote is abstracted behind a concept-constrained `ByteSource`. **All three implementations ship at launch** (pluggable/generic, no speculative arms):

```cpp
template <class S>
concept ByteSource = requires(S s, u64 off, u64 len, std::span<std::byte> dst) {
  { s.read(off, len, dst) } -> std::same_as<std::expected<std::size_t, IoError>>;
  { s.size() }             -> std::convertible_to<u64>;
};

struct LocalFile;    // mmap-backed; "remote" on the same disk / NFS
struct S3Source;     // libcurl + anonymous SigV4 (anon request signing)
struct HttpsSource;  // plain ranged HTTPS GET
```

`read` is a single ranged read (`offset`, `len`) into a caller-owned buffer; the source never allocates the destination, never decodes, never classifies. Dtype/variant dispatch is resolved once at `open()` and is orthogonal to which `ByteSource` is in use.

#### Batching: parallel independent per-chunk GETs (no coalescing)

The coordinator hands the **deduped** `DONT_KNOW` chunk set to N IO workers; each worker issues **independent whole-chunk ranged GETs in parallel**. Chunks are disjoint, so this is BSP-clean fork-join: every worker writes only its own chunk's sparse slot and its own local occupancy-delta buffer — disjoint partitions, nothing shared mutates, no synchronization needed. This exploits S3/HTTPS request parallelism directly and removes all multi-chunk coalescing logic (coalescing/dedup is the coordinator's job, done once before fan-out, per the original's 2-round 32-way batch being subsumed into the coordinator).

```cpp
// per-worker, executed in the mutate/IO phase over a disjoint chunk subset
struct ChunkFetchResult {
  ChunkId id;
  Occupancy resolved;          // REAL_DCT_DATA | ALL_ZERO
  bool ok;                     // false => stays DONT_KNOW, retried next tick
};
ChunkFetchResult fetch_into_slot(ByteSource auto& src, ChunkId id,
                                 std::span<std::byte> local_slot);
```

#### BSP integration (the IO-worker side of the coordinator)

- **Read phase:** consumers (sampler/renderer) hit the cache lock-free; misses accumulate into each worker's own local buffer. Nothing mutates.
- **Barrier (the tick boundary — the only sync point).**
- **Thaw / coordinator:** gather all local miss buffers, dedup, classify against the occupancy map. `ALL_ZERO` -> resolved as zeros with no network. `REAL_DCT_DATA` already-local -> nothing. `DONT_KNOW` -> the streaming work, also deduped **against fetches in flight from prior ticks** so a chunk is fetched at most once.
- **Mutate/IO phase:** fan out the deduped `DONT_KNOW` set to IO workers; each does its whole-chunk GET, unpacks into the local sparse slot, and records its occupancy delta locally. No archive reader runs concurrently with these writes (single isolated mutation phase). After fetch, the needed blocks are decoded and applied via `cache.apply(span<FreshPair>)`.
- **Barrier, re-freeze, repeat.**

Determinism note (from [[mcpp-architecture]]): decoded f32 values are non-deterministic and there is **no streaming-decode byte-identity** ("streaming decode byte-identical to flat" no longer applies), but the **coordination** — which chunks were classified/fetched/applied in what batched order — is a deterministic, testable pure transform.

#### Error / retry

Transient failure (network error, partial read, signature failure) leaves the chunk **`DONT_KNOW`**: the worker returns `ok == false`, no slot write, no occupancy flip. It is naturally re-requested next tick when a consumer misses on it again. This fits BSP with zero retry machinery — no backoff threads, no async tickets, no in-band retry state beyond the persistent occupancy itself.

#### Module layout

- `mcpp/stream/byte_source.hpp` — `ByteSource` concept + `IoError`.
- `mcpp/stream/local_file.hpp/.cpp`, `s3_source.hpp/.cpp` (libcurl + anon SigV4), `https_source.hpp/.cpp`.
- `mcpp/stream/sigv4.hpp/.cpp` — anonymous SigV4 request signing (own impl; no Boost/AWS-SDK).
- `mcpp/stream/wire.hpp` — `PackedChunkHeader`, offset-table walk, packed->sparse unpack.
- `mcpp/stream/fetch.hpp/.cpp` — `fetch_into_slot`, occupancy bootstrap GET, the IO-worker entry the coordinator calls.
- `mcpp/tools/repack` — "everything else" export tool: sparse local `.mcp` -> packed remote `.mcp` (produces offset tables + occupancy map for upload).

The coordinator/dedup itself lives with the BSP coordinator, not here; streaming exposes only the dumb per-chunk fetch primitive and the source implementations.

#### Invariants

- One fetch = one whole chunk = one ranged GET. No per-block partial fetch.
- `ALL_ZERO` chunks (known from the up-front occupancy map) generate **zero network traffic**, ever.
- `DONT_KNOW` is transient only; every persisted occupancy is `ALL_ZERO` or `REAL_DCT_DATA`.
- A fetched chunk lands in its **exact static slot** by pure arithmetic; local layout never reorders.
- Fetches occur only in the mutate/IO phase; no archive reader is live concurrently.
- IO workers touch **disjoint** chunks and **disjoint** slots — no shared mutable state, no locks/atomics.
- Failed fetch == no mutation == stays `DONT_KNOW` (idempotent retry next tick).
- `ByteSource` implementations never decode, classify, allocate the destination, or coordinate.

#### Milestones

1. **M1 ByteSource + LocalFile:** concept, `IoError`, ranged-read into caller buffer; full local-only end-to-end (remote == another local file).
2. **M2 Wire format + unpack:** `PackedChunkHeader`/offset-table, packed->sparse slot unpack with `0xFFFFFFFF` holes; round-trip test sparse->packed->sparse (structural, exact byte hash of THIS build's stored bytes).
3. **M3 Occupancy bootstrap:** one-GET map load; verify air chunks resolve with zero `read` calls (assert via a counting `ByteSource`).
4. **M4 Coordinator integration:** dedup `DONT_KNOW` set, dedup against in-flight, fan-out to disjoint IO workers, `cache.apply` of decoded blocks; deterministic-coordination test under TSan (must find nothing).
5. **M5 S3Source + HttpsSource:** libcurl ranged GET, anonymous SigV4 signing; live fetch against a real public bucket; parallel per-chunk GET throughput benchmark.
6. **M6 Retry/robustness:** transient-failure leaves `DONT_KNOW` and re-resolves next tick; fault-injection `ByteSource` (truncated/erroring reads) fuzz; corrupted-packed-chunk scrub at unpack boundary (bit-twiddling NaN/Inf detection per [[mcpp-scope]] fpbits).
7. **M7 Repack tool:** sparse local -> packed remote + occupancy map producer; integrity hash (xxh3-64 over stored bytes) on the packed object.

---

## 6. Subsystem #5 — Sampling

### 5. Sampling

The sampling subsystem (build-priority #5, after codec/archive/cache/streaming) bridges raw voxel storage to the geometry/surface layer. It is the **read-phase consumer** of the BSP loop: given a frozen volume source and a piece of geometry, it produces samples plus a miss-list, reading the cache lock-free and accumulating misses locally. It performs **no IO of its own** and triggers no fetches or decodes — that is the coordinator's job at thaw. It is therefore the cleanest possible expression of the governing BSP invariant (see [[mcpp-architecture]]): dumb, stateless w.r.t. IO, plain loads only, per-worker local accumulation.

#### 5.1 Role in the BSP loop

A sampler is a pure function `(frozen VolumeView, geometry) -> (samples, miss-list)`. During the read phase everything it touches is frozen; it reads `cache.get()` (lock-free, zero-copy) for every block it needs. On a block miss it does NOT fetch — it writes a hole into the output and **accumulates the miss into its own local miss buffer**, then continues. The miss-list is handed to the coordinator using the *same per-consumer-buffer mechanism* the cache subsystem uses for its misses (no shared ring, no atomics — disjoint local buffers gathered at the barrier).

Two consumption modes, both BSP-clean:

- **Interactive (render-now-refine-later):** sample once with whatever is resident, display the partial/holey result immediately, hand misses to the coordinator, and **refine on the next tick** as freshly-decoded blocks land.
- **Offline / batch (loop-to-completion):** the caller loops `sample -> thaw (coordinator fills misses) -> re-freeze` **until the miss-list is empty**, yielding a fully-dense result.

The sampler itself is identical in both modes; only the caller's loop differs. This mirrors the cache's `get()` / `get_best_lod()` duality.

#### 5.2 Source abstraction — `VolumeView` concept and `Sampler<View>`

All voxel access goes through a single concept. The block accessor returns a pointer on hit or a miss sentinel; it never blocks and never fetches.

```cpp
template<class V>
concept VolumeView = requires(const V v, u32 lod, i64 bz, i64 by, i64 bx) {
  typename V::value_type;                       // T (u8 main case, generic over the 8 dtypes)
  { v.block(lod, bz, by, bx) }                  // const T* (resident) or miss sentinel
      -> std::same_as<typename V::block_result>;
  { v.extent() } -> /* {dz,dy,dx} in voxels at lod 0 */;
};
```

Implementations (all model the concept; the sampler is templated on `View` for **zero-overhead generic** dispatch — no virtual calls on the hot path):

- **`CachedView`** — wraps a frozen `Cache<T>`; on a missing block records the miss into its local buffer and returns the miss sentinel.
- **`DenseView`** — a plain in-RAM `mcpp::mdview` (z-fastest, `[z,y,x]`); **never misses**, miss buffer stays empty. Used for already-resident regions, tests, and the offline final pass.
- **test views** — synthetic/deterministic generators for unit and tolerance tests.

Trilinear, region-read, oriented-crop, and quad-volume sampling **all read exclusively through `view.block(...)`**, so every primitive works identically over cache, dense array, or test data with no code duplication. This is the pluggable-source idiom; `Sampler<View>` monomorphizes against the chosen view.

#### 5.3 Trilinear / nearest kernel

- f32 throughout, `-ffast-math`, SIMD-friendly. A `Filter` enum selects `nearest` or `trilinear` per call.
- Trilinear resolves the **8 corner voxels** of the fractional coordinate. Those 8 corners can straddle **up to 8 distinct blocks** at a block boundary; if **any** required block is a miss, that sample is incomplete — write a hole and accumulate the miss(es), then move on (no partial-corner blending).
- **NaN/Inf scrub at ingest only.** Float inputs are sanitized via the shared `mcpp::fpbits` module (bit-twiddling on the IEEE-754 representation — `std::isnan` is useless under `-ffinite-math-only`) at the boundary where decoded blocks enter, then **trusted within the kernel**. No per-sample NaN checks on the hot path.

```cpp
enum class Filter { nearest, trilinear };
// returns {value, ok}; ok=false => hole, miss already recorded in the view's local buffer
template<VolumeView V>
SampleResult<typename V::value_type>
sample_point(const Sampler<V>&, f32 z, f32 y, f32 x, u32 lod, Filter);
```

#### 5.4 Tiny local last-block memo (ray/region coherence)

Each sampler instance holds a **single-threaded, per-sampler scratch** that caches the last-resolved block pointer(s). Consecutive samples along a ray or scanning a region overwhelmingly hit the same block, so we skip re-probing the view's block map. This is deliberately **simpler than the original's 256-entry pointer memo + ray-coherence cache** but captures the real coherence win. It is BSP-clean: it is *local scratch, never shared*, so it introduces no synchronization and no cross-worker state. (The original's larger memo is reference, not a contract.)

#### 5.5 Kept primitives (`sample_boxes` is DROPPED)

`sample_boxes` — the ML random-box dataloader — is **dropped entirely**: no random box sampling, no fraction-based dense selection, no sampling RNG, and **the deterministic-sampling invariant is obsolete and removed** (its only consumer was `sample_boxes`; the per-chunk fraction array goes with it). The four kept primitives:

**1. Dense region read** — pull an arbitrary `(z0,y0,x0)-(dz,dy,dx)` box of voxels into a caller-provided strided output buffer. Bulk read primitive. **Strides are expressed in bytes** (per the original convention) so callers can target sub-views, interleaved layouts, or transposed buffers. Misses produce holes + accumulated misses as usual.

```cpp
template<VolumeView V>
void read_region(const Sampler<V>&, RegionBox box, u32 lod,
                 std::byte* out, Strides strides_bytes);   // strides in BYTES
```

**2. Point trilinear / nearest** — §5.3, the core interp kernel.

**3. `sample_box` (oriented crop)** — an arbitrarily-oriented cuboid resampled from the volume (e.g. crops taken along a surface normal). This is the oriented-crop primitive and is **kept** — it is unrelated to the dropped `sample_boxes`. Geometry = an origin plus `du/dv/dw` basis vectors (oriented frame); each output cell is trilinear-sampled at the mapped world coordinate.

```cpp
struct OrientedBox { Vec3 origin, du, dv, dw; u32 nu, nv, nw; };
```

**4. `sample_quad_volume`** — the parametric-surface sampler and primary ink-detection / rendering input. The quad surface is a grid of `(u,v) -> {x,y,z}` subvoxel world coordinates — i.e. **a decoded codec use-case-4 parametric surface** (`codec<f32, Rank=2, C=3, Identity>`). The sampler thus *consumes a decoded surface to sample a volume* (a clean symmetry with the codec). For each `(u,v)` cell it takes the surface point, **marches along the surface normal over `nlayers`**, trilinear-samples the volume at each layer depth, and writes **layer-major** output (one full surface plane per layer, then the next layer).

It supports **both modes, chosen per call** (mirroring `cache.get()` vs `get_best_lod()`):

- **STRICT** — hole on miss, accumulate misses, intended for offline loop-to-completion.
- **BEST-LOD** — coarse-LOD fallback so the result is **never holey**, while still recording the *fine* misses for refinement. This is the interactive ink-preview path.

```cpp
enum class QuadMode { strict, best_lod };
struct QuadSurface { /* grid[v][u] = {x,y,z} f32, decoded use-case-4 surface */ };

template<VolumeView V>
void sample_quad_volume(const Sampler<V>&, const QuadSurface&,
                        i32 nlayers, f32 layer_step, Filter, QuadMode,
                        std::span<f32> out_layer_major);
```

#### 5.6 Module layout

- `mcpp/sampling/view.hpp` — `VolumeView` concept, `CachedView`, `DenseView`, test views.
- `mcpp/sampling/sampler.hpp` — `Sampler<View>`, last-block memo scratch, `Filter`, `SampleResult`, miss-buffer accumulation.
- `mcpp/sampling/point.hpp` — trilinear / nearest kernel (f32, fast-math, fpbits scrub at ingest).
- `mcpp/sampling/region.hpp` — `read_region` (byte strides).
- `mcpp/sampling/box.hpp` — `OrientedBox` + `sample_box`.
- `mcpp/sampling/quad.hpp` — `QuadSurface`, `QuadMode`, `sample_quad_volume` (layer-major march).

#### 5.7 Invariants

- **No IO, no fetch, no decode** inside any sampler; misses are accumulated, never serviced.
- **No multithread-safety code** — no locks/atomics-for-sync/mutexes; the only state is per-sampler local scratch over disjoint partitions (divide-and-conquer fork-join within the read phase).
- All access goes through `VolumeView`; primitives are source-agnostic and zero-overhead generic.
- f32 compute, `-ffast-math`; NaN/Inf scrubbed once at ingest via `mcpp::fpbits`, trusted thereafter; **no cross-ISA byte identity — correctness is tolerance-based.**
- Trilinear is all-or-nothing per sample (8 corners present or it is a hole).
- Miss-list is handed off via the same per-consumer-buffer mechanism as cache misses.

#### 5.8 Milestones

1. **`VolumeView` + `Sampler<View>` skeleton** — concept, `DenseView`, test view; `sample_point` nearest/trilinear over a dense source; tolerance tests vs analytic fields.
2. **`CachedView` + miss accumulation** — frozen-cache reads, local miss buffer, hole semantics; last-block memo; interactive refine-next-tick and offline loop-to-completion exercised end-to-end.
3. **`read_region`** with byte strides — bulk read, strided/transposed output, miss handling.
4. **`sample_box`** — oriented-crop frame math; tolerance tests on rotated synthetic volumes.
5. **`sample_quad_volume`** — layer-major normal march consuming a decoded use-case-4 surface; STRICT and BEST-LOD modes; ink-detection-shaped tolerance + coherence/perf benchmarks against the original's perf floor (1024² quad slice ~1.5 ms; 9-step trilinear composite ~8.4 ms reference).

---

## 7. Subsystem #6 — Rendering & Geometry

Subsystem #6 layers VC3D-style surface rendering on top of the Sampler (#5), and owns the **complete geometry/surface layer** — mcpp becomes the geometry foundation that VC3D's tracer/segmentation/flattening rebuild against. Two halves: the **Renderer** (a read-phase BSP frame consumer) and the **full geometry layer** (Surface concept, QuadSurface, PlaneSurface, Rect3D, point-grid algebra, spatial indices, surface area). Both are mcpp-native f32, BSP-conformant, and contain **no OpenCV and no Boost** — we write our own vectors, grids, and spatial trees.

#### Posture: mcpp-native types, no OpenCV anywhere

VC3D's surface layer is currently drenched in `cv::Mat_<cv::Vec3f>` / `cv::Vec3f` / `cv::Vec2f` / `cv::Size` / `cv::Rect`. All of it is replaced by mcpp-native types; mcpp does **not** bend to `cv::` and promises no layout-compat shims — consumers convert to our types at their boundary.

- `Vec3f` / `Vec2f` — small POD float vectors, `-ffast-math`-friendly, SIMD-aligned, full operator/dot/cross/normalize set.
- `Grid<Vec3f>` — owning, contiguous, row-major point grid (rows = height/V, cols = width/U), with `mcpp::mdview<Vec3f, 2, mcpp::row_major>` views handed out for read-only and kernel access. Same Grid is the codec case-4 parametric surface (one data type, two views: geometry layer + codec use case).
- `Rect3D { Vec3f low, high }` — AABB. `intersect(a,b)`, `expand_rect(a,p)`.
- INVALID point sentinel = `(-1,-1,-1)` (and/or NaN); NaN detected via `mcpp::fpbits` (never `std::isnan`, which `-ffinite-math-only` folds away).

#### Surface = C++26 concept + optional type-erased handle

`Surface` is a **concept** that QuadSurface and PlaneSurface satisfy **statically**, so samplers monomorphize over the concrete surface type and pay zero virtual-call cost on `coord`/`normal`/`loc` in hot loops (the original's virtual dispatch on these was a measured cost). This matches the codec/cache/VolumeView idiom.

```cpp
template <class S>
concept Surface = requires(S s, const SurfacePtr p, Vec3f off, /*…*/ ) {
  { s.coord(p, off)  } -> std::same_as<Vec3f>;   // ptr-space -> world coord
  { s.normal(p, off) } -> std::same_as<Vec3f>;
  { s.loc(p, off)    } -> std::same_as<Vec3f>;    s.loc_raw(p);
  s.gen(/*coords,normals,size,ptr,scale,off*/);   // -> coord+normal Grid<Vec3f>
  { s.pointTo(p, /*tgt,th,iters*/) };             // nearest-surface search
  { s.valid(p, off)  } -> std::same_as<bool>;
  s.move(p, off);
  { s.size()   } -> /*Vec2i*/;  s.scale(); s.center();
};
```

`AnySurface` — a type-erased handle (small-buffer vtable wrapper) is provided **only** where runtime polymorphism is genuinely needed (heterogeneous UI surface lists). Hot sampling/render paths take the concrete concept-satisfying type; UI/registry code takes `AnySurface`.

#### Three coordinate systems (kept from VC3D)

1. **ptr-space / internal-relative** — origin at the surface `_center`, zero at center. `move`/`valid`/`loc`/`coord`/`normal`/`pointTo` all operate here.
2. **nominal / voxel-volume** (world voxel coords): `nominal = ptr/scale + offset + center`.
3. **grid-absolute** — raw `(row, col)` indices, `(0,0)` top-left.

#### QuadSurface — the key type, full operation set

Representation: `Grid<Vec3f>` point grid (rows = V/height, cols = U/width), `Vec2f scale` (per-axis nominal scale), `Vec3f center`, `Rect3D bbox`. Operations (all load-bearing for tracer / GrowSurface / GrowPatch / ABFFlattening / Segmentation):

- **Raw grid access** — `rawPoints()` / `rawPointsPtr()` for loss functions and geometry queries.
- **Validity** — `valid()`, `isPointValid(r,c)`, `isQuadValid(r,c)`, `countValid()`, a cached `validMask` (255/0), and C++20/26 **range** iteration:
  `for (auto [r,c,pt] : surf.validPoints())` and `for (auto [r,c,p00,p01,p10,p11] : surf.validQuads())`.
- **Geometry queries** — `coord`/`loc`/`loc_raw`/`normal`, plus `gridNormal(r,c)` with cached normals.
- **`pointTo`** — nearest-surface-point gradient-descent search (with optional spatial-index acceleration), plus `ptrToGrid`.
- **`gen(coords, normals, size, ptr, scale, offset)`** — THE rendering/slicing driver: affine-warp + bilinear the grid into output coord and normal grids at the requested scale/offset, with **4px padding per side**, NaN/`(-1)` validity warp, and multi-component (disconnected-region) handling. Output feeds `sample_quad_volume`.
- **Transforms** — `rotate(deg)`, `resample(factor[,x,y])`, `flipU`/`flipV`, `orientZUp`/`computeZOrientationAngle`.
- **Metadata** — `bbox`, `size`/`scale`/`center`, `dpi`.
- **Lazy load** — `ensureLoaded`/`isLoaded`/`unloadPoints`/`unloadCaches` (many large surfaces resident).
- **Channels & persistence** — ancillary masks / depth-index grids by name, overlapping-surface id tracking, `save`/`save_meta`/`saveSnapshot`/`saveOverwrite`.

#### PlaneSurface

Origin + normal + orthonormal basis `(vx, vy)` + `inPlaneRotation`. `setFromNormalAndUp(origin, normal, upHint)` builds a **stable** basis (no sign-flip discontinuity when the normal sweeps). Provides `pointDist`/`project`/`scalarp`; `valid()` is always true; `gen()` emits the plane coord grid. `pointTo` is **not** implemented for planes.

#### Spatial indices — our own trees, BSP-fit

mcpp provides `pointTo` **and** the spatial-acceleration structures it needs — no Boost, no external geometry libs:

- `PointIndex` — `nearest` / `nearestInCollection` / `queryRadius` / `kNearest` over 3D world coords; `insert` / `bulkInsert` / `buildFromGrid`.
- `SurfacePatchIndex` — patch-based index that accelerates `pointTo` (tracer depends heavily on both).

We implement the trees in-house (R-tree / k-d / BSP / octree — whichever fits). **BSP-fit:** trees are **built in the mutate phase** (once per tick from the surface) and **queried lock-free in the read phase**. They are immutable-during-read, so the natural form is a **static, cache-friendly, built-once tree** with pure read-only queries — **no concurrent-insertion machinery, no locks, no atomics.**

#### Surface area

`quadAreaVox2` / `triangleAreaVox2` / `computeSurfaceAreaVox2(grid | surface)` for raw areas, and `computeMaskArea` (returns `area_vx2` / `cm2`, median step u/v, voxelSize) for physical-area reporting.

#### Renderer — read-phase BSP frame consumer built on Sampler

`render_plane` / `render_quad` are the canonical BSP frame loop and the motivating use case for tick-phase freeze/thaw. Each tick:

1. **READ PHASE (frozen):** for each output pixel, `Surface::gen` produces the coord+normal grid; the Sampler reads the frozen cache/VolumeView lock-free; the **compositor reduces** samples along the normal over the `[t0,t1,dt]` march into the pixel; **misses accumulate locally**.
2. **BARRIER (tick boundary).**
3. **MUTATE PHASE:** coordinator merges all band miss-lists and fetches/decodes them.
4. **Re-freeze; next tick refines.**

Properties:
- **Owns no IO.** Reads exclusively through `VolumeView` (pluggable: `CachedView`, `DenseView`, test views), exactly as the Sampler does.
- **Refine-next-tick (render-now-refine-later):** first tick may use coarse LOD / leave fine misses; subsequent ticks refine via `get_best_lod`. Interactive frames are never holey; offline callers loop sample→thaw until no misses.
- **LOD-matched:** sample the LOD the current zoom can actually display (half-voxel-correct remap), so zoomed-out frames are cheap (~2-3x per level). Couples to `get_best_lod` and the pre-aligned 2x LODs (`coord/2` arithmetic).

```cpp
template <Surface S, VolumeView V, Compositor C>
Image render_quad(const S& surf, Sampler<V>& smp, C comp,
                  RenderParams rp, MissBuffer& misses /*local*/);
```

`RenderParams` carries the normal-march struct `{ float t0, t1, dt; LodPolicy lod; }` and the output region/zoom.

#### Render parallelization — band-partition, disjoint pixels

The output image is split into **bands** (rows or tiles); **one worker per band**, each samples + composites **only its own pixels** and accumulates its **own local miss-list** (divide-and-conquer + local accumulation). Disjoint output means zero contention — BSP-clean fork-join, no locks. **One Sampler per band**, each with its own single-threaded last-block memo for ray/region coherence. The coordinator merges per-band miss-lists at thaw.

#### Compositors — pluggable concept-constrained reducer policies

Compositors are reducer policies satisfying a `Compositor` concept; new ones drop in **without touching the renderer**. For each output pixel the renderer samples the volume along the surface normal over `[t0,t1,dt]` and the policy reduces the sample stack.

```cpp
template <class C>
concept Compositor = requires(C c, std::span<const float> samples) {
  { c.reduce(samples) } -> std::convertible_to<float>;
};
```

- **At launch:** `MinComposite`, `MeanComposite`, `MaxComposite`, `AlphaComposite` (min/mean/max are the ink-visibility workhorses).
- **Later (if needed):** shaded modes — emission-absorption march, gradient/shadow — added as additional policies, no renderer changes.

#### Module layout

- `geom/vec.hpp` — `Vec2f`/`Vec3f`.
- `geom/grid.hpp` — `Grid<T>` + mcpp::mdview views.
- `geom/rect3d.hpp` — `Rect3D`.
- `geom/surface.hpp` — `Surface` concept + `AnySurface`.
- `geom/quad_surface.hpp` / `plane_surface.hpp`.
- `geom/index/point_index.hpp` / `surface_patch_index.hpp` + in-house tree(s).
- `geom/area.hpp` — surface-area functions.
- `render/compositor.hpp` — `Compositor` concept + min/mean/max/alpha.
- `render/renderer.hpp` — `render_plane` / `render_quad`, `RenderParams`, band-partition driver.

#### Invariants

- **BSP everywhere:** render reads frozen state lock-free, accumulates misses per-band-local, single mutate phase merges/fetches. No locks/mutexes/atomics-for-sync anywhere in #6.
- **Surfaces satisfy the concept statically** in hot paths; `AnySurface` only for UI polymorphism.
- **Spatial trees are build-once-per-tick (mutate) / read-only (read)** — immutable during read, no concurrent-insert apparatus.
- **No OpenCV, no Boost** — mcpp-native types and trees throughout.
- **f32, `-ffast-math`, NaN/invalid via `mcpp::fpbits` and the `(-1,-1,-1)`/NaN sentinel.** No byte-identity; correctness is tolerance-based.
- Renderer owns no IO; reads only through `VolumeView`.

#### Milestones

1. **Geom core:** `Vec2f`/`Vec3f`, `Grid<Vec3f>`, `Rect3D`, `Surface` concept — with unit tests.
2. **PlaneSurface:** `setFromNormalAndUp` (stable basis), `gen`, project/dist; `render_plane` over `DenseView` with min/mean/max compositors. Hits the ~1.5ms 1024² slice perf floor.
3. **QuadSurface MVP:** grid storage, validity + `validPoints`/`validQuads` ranges, `coord`/`loc`/`normal`/`gridNormal`, `gen` (affine-warp+bilinear, padding, validity warp); `render_quad` via `sample_quad_volume`. Beat the ~8.4ms 9-step composite floor.
4. **Spatial indices:** in-house `PointIndex` + `SurfacePatchIndex`; wire `pointTo` to the patch index; build-in-mutate / query-in-read.
5. **QuadSurface full:** transforms (rotate/resample/flip/orientZUp), multi-component `gen`, channels, lazy-load, persistence, surface-area functions.
6. **BSP render loop:** band-partition workers + per-band Samplers/miss-lists, coordinator merge at thaw, refine-next-tick + LOD-matched `get_best_lod` integration; full read/refine cycle over `CachedView`.
7. **AnySurface** + shaded compositor policies as extension points (later).

---

## 8. Everything Else (#7) & Quality Engineering

### 7. Everything Else: `.mcs` Format, Ingest Tool, and What We Don't Build

This subsystem is the final tier after codec -> archive -> cache -> streaming -> sampling -> rendering+geometry. It is deliberately small: one new container format, exactly one CLI tool, and an explicit list of tools we are *not* writing because the static-mapping archive design eliminated the need for them.

#### 7.1 The `.mcs` Segment Container (TOC-Based)

`.mcp` is one fixed-geometry 3D volume whose entire on-disk position is a pure static arithmetic function of `(dims, dtype, LODs)` — that is the property that makes 100TB+ volumes addressable without an index. A **segmentation** is the opposite shape: a heterogeneous *list* of small 2D parametric surface patches, each a different `W x H`, with no global geometry. Static mapping does not apply, so `.mcs` is a separate, simpler, **table-of-contents container** — not a variant of `.mcp`.

```
[header][TOC: { id, offset, size } per surface][blob 0][blob 1]...[blob N-1]
```

Each surface blob is looked up by the TOC and is fully **self-contained** — independent blobs, no inter-blob references, no global geometry table.

**Surface blob layout (self-contained):**
- **Compressed grid** via **codec case 4**: `codec<f32, Rank=2, C=3, Identity>` — DCT-64 over `64x64` blocks, 3 xyz channels treated as mutually independent (Identity transform, *not* YCoCg), **max-error-bounded in voxel units** with an **affine-frame predictor/residual** (DCT the small residual against the grid's fitted affine frame, never raw world coords). This is the surface case from the scope: `grid[v][u] = {x,y,z}` subvoxel coordinates, quality measured as absolute L∞ coordinate error in voxel units (τ ≈ 0.05–0.1), enforced by explicit residual correction so `-ffast-math` is safe.
- **Named channels**, each compressed in-blob with the codec appropriate to its content. Validity masks are u8 0/255 -> **raw / RLE / bitpack, NOT DCT**; depth maps and similar continuous channels can use the 2D codec.
- **Per-surface JSON metadata**: scale, center, bbox, dpi, overlapping-ids, free-form fields.

```cpp
struct mcs_toc_entry { std::uint64_t id, offset, size; };
struct surface_blob {              // self-contained, codec case 4 for the grid
  // codec<f32, 2, 3, Identity> compressed xyz grid (max-error-bounded + affine residual)
  // + in-blob compressed named channels (mask: bitpack/RLE; depth: 2D codec)
  // + JSON meta
};
std::expected<mcs_archive, mcs_error> mcs_open(ByteSource&);   // shared ByteSource layer
```

**Mutation model: read-mostly, bulk-written.** A segmentation run produces all surfaces at once; the tracer works in memory and dumps a complete `.mcs` at checkpoints. Editing = rewrite the file. There is **NO incremental in-place / append-per-surface mutation** — this is the simplest model and BSP-clean: the entire bulk write is a single mutate phase over frozen in-memory state.

#### 7.2 `zarr -> .mcp` Ingest Tool (the one CLI we ship)

This is the essential data-acquisition path: Vesuvius volumes live as S3 zarr, and this tool produces a complete local `.mcp` archive from them. It **reuses the `ByteSource` layer** (S3 SigV4-anonymous / https) from the streaming subsystem rather than reimplementing remote IO.

**Pipeline:** read zarr (blosc-zstd or raw chunks) -> air-mask -> re-block to `256^3` chunks of `16^3` codec blocks -> encode (codec) -> write `.mcp` static-sparse layout + occupancy map.

Decisions baked in:
- **Whole volume only.** Ingest the *entire* zarr into a *complete* `.mcp` (no sub-volume box). Every output chunk is either `REAL_DCT_DATA` or `ALL_ZERO` — there are **no `DONT_KNOW` chunks** in a freshly ingested archive.
- **LOD: ingest LOD0, tool generates LOD1–7.** Read only the finest zarr level; the tool builds all coarser LODs by **2×2×2 averaging** and hands the archive writer all 8 pre-built LODs. This is consistent with the rule that *mcpp core never generates LODs* — the ingest tool is the external LOD provider.
- **Air-masking folded in.** A histogram-valley interior-air cut runs **before encode** (the original's `mc_mask` step) so interior air -> 0, which lets the codec's mask-aware air-fill and `ALL_ZERO` chunks pay off — a large ratio win on mostly-air scrolls. (AWS volumes arrive ROI-masked only, so we still do our own cut.)
- **Air-propagating downsample (critical correctness constraint).** Because we mask at LOD0 and *then* downsample, the 2×2×2 reducer must be **mean over NONZERO children only** so that air does not bleed into material at coarse LODs. The tool's downsampler implements exactly this propagation rule.

```cpp
// ingest tool, not core: external LOD provider feeding the archive writer
void ingest_zarr_to_mcp(ByteSource& zarr_src, std::filesystem::path out_mcp,
                        float air_valley);          // whole-volume, all 8 LODs built here
float downsample_2x2x2(std::span<const float, 8> children);  // mean over nonzero only
```

#### 7.3 Tools We Deliberately Do NOT Build

- **NO repacking tool.** With the pure static mapping there is exactly **one canonical on-disk layout** for any `(dims, dtype, LODs)`, so there is nothing to repack. (The original needed `mc_export` only because its dynamic-append layout was non-canonical and Morton-reclustering mattered; ours is deterministic by construction.) The one related operation — producing the remote *packed* form from a local *sparse* file — is a trivial sparse->packed serialization (write each occupied chunk contiguously + ship the occupancy map) that lives in the archive/streaming write path, **not** a standalone tool.
- **NO other CLI tools.** Skip `mc_mask`, `mc_bench`, `mc_export`, `mc_verify`, `mc_train`, `mc_vs_c3d`, `mc_prof`. Bench and verify exist as **tests**, not shipped binaries; integrity verification is a **library call**, not a CLI.

---

### 8. Quality Engineering: Test Framework, Fuzzing, CI Gates, Roadmap

The architecture (BSP + f32-everywhere + tolerance correctness) reshapes *what* we test, not just how. The headline consequences: the codec is never byte-exact, concurrency bugs are designed out rather than tested out, and only *structurally deterministic* things get exact goldens.

#### 8.1 Our Own Test Framework

Consistent with the in-house posture (no Boost, no OpenCV, our own spatial trees), we **write our own** lightweight C++26 test harness. No external test dependency. It provides:
- **Registration** that is constexpr/consteval-friendly.
- Ordinary assertions, plus **tolerance comparators as first-class citizens** — `PSNR`, `MAE`, `max-error`, `SSIM` assertion helpers. These *are* the codec's correctness currency, not bolt-ons.
- A **property-testing layer**: random generators + shrinking, used for roundtrip and invariant properties.

```cpp
EXPECT_PSNR_GE(decoded, original, 42.0);          // perceptual quality-dial cases
EXPECT_MAXERR_LE(decoded, original, tau);          // exact, survives fast-math
mcpp::test::property("roundtrip", gen_volume<u8>(), [](auto v){ /* ... */ });
```

#### 8.2 Codec Tests: Tolerance + Exact-Property + Structural-Exact

Three distinct regimes, applied to the right targets:
- **Tolerance-based roundtrip (codec body).** f32 + `-ffast-math` is non-deterministic, so there are **ZERO golden-byte codec tests**. Roundtrip must hold within PSNR/MAE/max-error/SSIM bounds, swept over the full matrix {8 dtypes} × {4 use cases} × {quality levels}.
- **Exact property tests (survive fast-math).** **Air/fill bit-exactness** is the one *exact* codec invariant — air decodes to *exactly* 0, checked by an integer mask compare outside the float path (`==`-style assertion). **Max-error τ is tested as BEST-EFFORT** (review C1): assert τ holds on the *encoder's reference decode* exactly, and assert the *as-built decoder* stays within τ+δ (a measured slack), NOT a hard `==`. Do not gate on a guaranteed cross-decoder τ.
- **Structural-exact archive tests.** Archive *structure* is fully deterministic = whatever bytes *this* build wrote. Exact tests cover: static-mapping arithmetic, occupancy states/transitions, offset-table integrity, xxh3-64 roundtrip, sparse-hole behavior, the `ALL_ZERO` sentinel. Core property: write N chunks at arbitrary slots -> read back identical. (`.mcs` gets the analogous TOC-integrity / blob-roundtrip structural tests.)

#### 8.3 Cross-ISA: Quality-Equivalence-Within-Tolerance

x86-64 (AVX2) vs ARM (NEON via QEMU aarch64 user-mode): both must decode to **within tolerance of each other and of the reference** — **never byte-identical**. Cross-ISA byte identity is gone (it required an integer hot path, incompatible with f32 + fast-math). The assertion is a tolerance bound, never `==`.

#### 8.4 BSP Makes Concurrency Testing Trivial

By design there are no locks, mutexes, atomics-as-sync, or concurrent mutation anywhere, so **TSan should find NOTHING**. A TSan hit is therefore *itself the bug*: it signals a **BSP violation** — the exact bug class we want to catch. We do not write lock tests; instead we test **phase discipline**: no mutation during the read phase, single-caller `apply()` (debug-asserted), and disjoint worker partitions. This is debug-assert + fuzz-check, not lock-stress.

#### 8.5 Fuzzing: AFL++ under ASan/UBSan

Coverage-guided **AFL++** (libFuzzer is present but AFL++ is chosen). Three targets, all built with ASan + UBSan:
1. **Decode-of-corrupted `.mcp`/`.mcs`** — the hardened decoder must never OOB/crash/SIGSEGV on untrusted bytes; bounded offsets, clamped reads.
2. **Encode -> decode roundtrip** — tolerance holds, no crash, random volumes.
3. **`mcpp::fpbits` NaN/Inf scrub** on garbage f32 input — the bit-twiddling classifier (which `-ffast-math`/`-ffinite-math-only` can't elide) correctly sanitizes at the ingest boundary.

CI runs a short fuzz per PR; long campaigns run nightly.

#### 8.6 CI: Full Matrix, Hard Merge Gates

Matrix: **{Clang 21, GCC 15} × {Debug+ASan/UBSan, Release}** on x86-64, plus **ARM/NEON via QEMU**. *All* of the following are hard gates — every one must pass to merge:
- both-compiler build;
- all unit + property tests;
- codec **tolerance bounds** met;
- **cross-ISA** quality-equivalence-within-tolerance;
- archive **structural-exact** tests;
- **AFL++ fuzz** — no new crashes;
- **TSan clean** (BSP => no races);
- **coverage >= threshold** (llvm-cov);
- **NO perf regression** — change-point detection on a frequency-pinned runner, matching/beating the original perf floor: ~9 µs cold single-block decode, ~150 M gets/s/thread cache, 1024² slice ~1.5 ms, 9-step trilinear composite ~8.4 ms.

Verified toolchain: Clang 21.1.8, GCC 15.2.0, CMake 4.2.3, Ninja 1.13.2, QEMU aarch64 user-mode, llvm-cov/profdata 21, AFL++.

#### 8.7 Phased Roadmap with Per-Phase Gating Tests

Phases follow the locked build-priority order, each gated by the tests that become meaningful at that phase:

| Phase | Subsystem | Gating tests added |
|---|---|---|
| P1 | **codec** | tolerance roundtrip (full dtype × use-case × quality sweep); exact max-error-bound + air bit-exactness; `fpbits` scrub fuzz; cross-ISA tolerance-equivalence; per-block decode perf floor (~9 µs) |
| P2 | **archive** | structural-exact (static mapping, occupancy, offset table, xxh3 roundtrip, `ALL_ZERO`, sparse holes); corrupted-`.mcp` decode fuzz |
| P3 | **cache** | dumb-component `apply(span<FreshPair>)` single-caller assert; frozen-read lock-free property; TSan-clean under phase discipline; ~150 M gets/s floor |
| P4 | **streaming** | `ByteSource` (S3/https) integration; partial-fetch byte-reduction property; phase-discipline / disjoint-partition checks |
| P5 | **sampling** | trilinear/nearest correctness vs analytic; LOD half-voxel remap; 1024² slice ~1.5 ms floor |
| P6 | **rendering + geometry** | composite reducer correctness; QuadSurface/PlaneSurface coordinate math (exact goldens — deterministic); 9-step composite ~8.4 ms floor |
| P7 | **everything else** | `.mcs` TOC-integrity + blob roundtrip (tolerance grid + exact mask/channel); corrupted-`.mcs` fuzz; ingest air-propagating downsample correctness (mean-over-nonzero); whole-volume ingest end-to-end |

Each phase's gates fold into the standing CI matrix; later phases never relax earlier guarantees.

#### 8.8 Open Questions / Risks

Tracked, minor, none blocking:
- **Exact coverage threshold %** not yet fixed.
- **Perf-regression specifics**: which change-point algorithm, exactly which benchmarks gate, and the regression threshold.
- **Property-test generator/shrinker design** details.
- **Golden corpus** for the deterministic-things (archive structure, coordinate math, BSP coordination order, spatial-tree queries) — what to snapshot and where.
- **Whether nightly adds AVX-512** — originally rejected for the fleet (consumer fused-off / Zen4 double-pumps 256), so **probably not gated**; revisit only if a target fleet justifies it.
- **Test-fixture risk**: fixtures can't be golden bitstreams (non-deterministic) — must stay synthetic-generated-in-test or tolerance-compared; a regression here would silently weaken the codec suite.
