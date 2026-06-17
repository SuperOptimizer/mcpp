# mcpp — matter-compressor++

A modern **C++26** volumetric codec + sparse-mmap archive + in-RAM cache +
streaming + sampling + surface rendering, for Vesuvius-Challenge-scale (100 TB+)
masked micro-CT. A from-scratch reimplementation of `matter-compressor`,
replacing the data-plane + surfaces/geometry layer of `volume-cartographer/core`.

See `PLAN.md` for the full design and `PLAN_REVIEW.md` for the adversarial review.

## Architecture

The whole codebase is **Bulk Synchronous Parallel (BSP)**: dumb components with
lock-free frozen reads and a single batched mutation per tick — **no
multithread-safety code anywhere** (no locks/atomics-for-sync). Data races are
designed out, not tested out.

Compute is **f32 everywhere**; `-ffast-math` is an expected build mode.
Correctness is **tolerance-based** (PSNR/MAE/max-error), except where it's exact
by construction (air bit-exactness via integer mask; the integer entropy coder).

## Subsystems (header-only `mcpp::`)

| Module | What |
|--------|------|
| `core/` | fpbits (NaN/Inf under fast-math), dtype vocabulary, mdview, error model |
| `codec/` | f32 separable DCT-16/64, dead-zone quant, range coder, coefficient model, mask-aware air handling, generic `Codec<T,N,Rank,C,Transform>`, max-error pass |
| `os/` | sparse-file mmap seam (Linux/macOS) |
| `archive/` | static-slot sparse-mmap `.mcp`, 3-state occupancy, chunk format, `write_volume`/`read_region` |
| `cache/` | BSP freeze/thaw dumb-store decoded-block cache |
| `coordinator/` | gather → dedup → decode → apply (the BSP mutation phase) |
| `streaming/` | `ByteSource` + write-through remote cache (robust I/O) |
| `sampling/` | `VolumeView` + trilinear/nearest sampler + `sample_quad_volume` |
| `geom/` + `render/` | mcpp-native geometry (Surface/Plane/Quad), pluggable compositors, renderer |
| `mcs/` | `.mcs` segment format (list of 2D parametric surfaces) |
| `ingest/` | volume → masked, multi-LOD `.mcp` |

## Build & test

```sh
cmake --preset clang-debug      # also: gcc-debug, clang-release, gcc-release
cmake --build --preset clang-debug
ctest --preset clang-debug
```

CI (`.github/workflows/ci.yml`) gates merges on all four presets
(`{clang,gcc} × {debug+ASan/UBSan, release+fast-math}`) plus a fuzz smoke.

Fuzzing (Clang/libFuzzer):

```sh
cmake --preset clang-fuzz && cmake --build --preset clang-fuzz
./build/clang-fuzz/fuzz/fuzz_block_decode -max_total_time=60
```

## Status

All subsystems implemented and tested (22 test executables, green on all four
presets, under ASan/UBSan and fast-math). Deferred polish (documented in
headers): SIMD kernels, trained entropy priors, S3/HTTPS sources, Windows mmap
backend, thread-pool fan-out.
