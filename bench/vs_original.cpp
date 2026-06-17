// Rate-distortion comparison: mcpp codec vs the original matter-compressor codec
// (the per-block CABAC codec, NOT the c3g GPU path), on real PHercParis4 scroll
// data. Both are independent lossy codecs (mcpp is greenfield, not bit-compatible)
// so we compare RATE (compression ratio) vs DISTORTION (PSNR/MAE/max-error) at
// matched quality.
//
// Volume: a centered 1024^3 region assembled from raw u8 zarr chunks (128^3).
// We SAMPLE blocks (every Nth 16^3 block) for a fast RD curve.

#include "mcpp/codec/block.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// the original codec's C API
extern "C" {
#include "matter_compressor.h"
}

static constexpr int   N    = 1024;          // region side (voxels)
static constexpr int   CH   = 128;           // zarr chunk side
static constexpr int   NCH  = N / CH;        // 8 chunks/axis
static constexpr int   B    = 16;            // codec block side
static constexpr int   NB   = N / B;         // 64 blocks/axis

// Assemble the 1024^3 volume from /tmp/scroll_chunks/<z>_<y>_<x>.bin.
// Chunk index bases: z 21, y 28, x 28.
static std::vector<std::uint8_t> assemble() {
    std::vector<std::uint8_t> vol(std::size_t(N) * N * N, 0);
    const int z0 = 21, y0 = 28, x0 = 28;
    std::vector<std::uint8_t> cbuf(std::size_t(CH) * CH * CH);
    for (int cz = 0; cz < NCH; ++cz)
    for (int cy = 0; cy < NCH; ++cy)
    for (int cx = 0; cx < NCH; ++cx) {
        char path[256];
        std::snprintf(path, sizeof path, "/tmp/scroll_chunks/%d_%d_%d.bin",
                      z0+cz, y0+cy, x0+cx);
        std::FILE* f = std::fopen(path, "rb");
        if (!f) { std::fprintf(stderr, "MISSING chunk %s\n", path); std::exit(2); }
        std::size_t rd = std::fread(cbuf.data(), 1, cbuf.size(), f);
        std::fclose(f);
        if (rd != cbuf.size()) { std::fprintf(stderr, "SHORT chunk %s\n", path); std::exit(2); }
        // zarr chunk is C-order [z,y,x]; copy into the big volume
        for (int lz = 0; lz < CH; ++lz)
        for (int ly = 0; ly < CH; ++ly) {
            const std::uint8_t* src = &cbuf[(std::size_t(lz)*CH + ly)*CH];
            std::uint8_t* dst = &vol[((std::size_t(cz*CH+lz)*N) + (cy*CH+ly))*N + cx*CH];
            std::memcpy(dst, src, CH);
        }
    }
    return vol;
}

// extract a 16^3 block at block coords into a 4096 u8 buffer (raster z,y,x)
static void get_block(const std::vector<std::uint8_t>& vol, int bz, int by, int bx,
                      std::uint8_t* out) {
    for (int lz = 0; lz < B; ++lz)
    for (int ly = 0; ly < B; ++ly) {
        const std::uint8_t* src = &vol[(((std::size_t)(bz*B+lz))*N + (by*B+ly))*N + bx*B];
        std::memcpy(&out[(std::size_t(lz)*B + ly)*B], src, B);
    }
}

struct Stats {
    double sum_se = 0; double sum_ae = 0; double maxe = 0;
    std::uint64_t nvox = 0; std::uint64_t bytes = 0; std::uint64_t raw = 0;
    void add_voxel(int orig, int dec) {
        double e = double(orig) - double(dec);
        sum_se += e*e; sum_ae += std::fabs(e); if (std::fabs(e) > maxe) maxe = std::fabs(e);
        ++nvox;
    }
    double psnr() const { double mse = sum_se / double(nvox); return mse>0 ? 10*std::log10(255.0*255.0/mse) : 999; }
    double mae()  const { return sum_ae / double(nvox); }
    double ratio() const { return double(raw) / double(bytes); }
};

int main() {
    std::fprintf(stderr, "assembling 1024^3 volume from chunks...\n");
    auto vol = assemble();

    // air fraction
    std::uint64_t air = 0; for (auto v : vol) if (v == 0) ++air;
    std::fprintf(stderr, "volume: %d^3, air fraction = %.1f%%\n", N,
                 100.0 * double(air) / double(vol.size()));

    mc_codec_init();
    mc_codec_ctx* ctx = mc_codec_ctx_new();

    const int STRIDE = 4;  // sample every 4th block per axis -> 1/64 of blocks
    // wide sweep so the two RD curves overlap in ratio and can be compared at
    // matched ratio (q is scaled differently between the codecs).
    const float qualities[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 9.0f, 12.0f, 20.0f, 32.0f};

    std::printf("\n%-6s | %-28s | %-28s\n", "q", "ORIGINAL matter-compressor", "mcpp");
    std::printf("%-6s | %8s %7s %6s %5s | %8s %7s %6s %5s\n",
                "", "ratio", "PSNR", "MAE", "max", "ratio", "PSNR", "MAE", "max");
    std::printf("-------+------------------------------+------------------------------\n");

    std::uint8_t blk[4096], dec_o[4096], dec_m[4096];

    for (float q : qualities) {
        mc_codec_ctx_set_quality(ctx, q);
        Stats so, sm;
        int nblocks = 0;

        for (int bz = 0; bz < NB; bz += STRIDE)
        for (int by = 0; by < NB; by += STRIDE)
        for (int bx = 0; bx < NB; bx += STRIDE) {
            get_block(vol, bz, by, bx, blk);
            ++nblocks;

            // --- ORIGINAL ---
            mc_buf ob{nullptr, 0, 0};
            std::uint32_t olen = 0;
            mc_enc_block(ctx, blk, &ob, &olen);
            mc_dec_block(ctx, ob.p, olen, dec_o);
            so.bytes += olen; so.raw += 4096;
            for (int i = 0; i < 4096; ++i) so.add_voxel(blk[i], dec_o[i]);
            if (ob.p) free(ob.p);

            // --- mcpp ---
            std::vector<float> fblk(4096);
            for (int i = 0; i < 4096; ++i) fblk[i] = float(blk[i]);
            std::vector<std::uint8_t> mb;
            mcpp::codec::encode_block<16,3>(fblk.data(), q, mb);
            std::vector<float> fdec(4096);
            mcpp::codec::decode_block<16,3>(mb.data(), mb.size(), q, fdec.data());
            sm.bytes += mb.size(); sm.raw += 4096;
            for (int i = 0; i < 4096; ++i) {
                int d = int(std::lround(fdec[i])); if (d<0) d=0; if (d>255) d=255;
                dec_m[i] = std::uint8_t(d);
                sm.add_voxel(blk[i], dec_m[i]);
            }
        }

        std::printf("%-6.1f | %7.2fx %7.2f %6.2f %5.0f | %7.2fx %7.2f %6.2f %5.0f   (%d blocks)\n",
                    q, so.ratio(), so.psnr(), so.mae(), so.maxe,
                    sm.ratio(), sm.psnr(), sm.mae(), sm.maxe, nblocks);
    }

    mc_codec_ctx_free(ctx);
    return 0;
}
