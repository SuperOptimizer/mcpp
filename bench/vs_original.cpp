// Rate-distortion comparison: mcpp codec vs the original matter-compressor codec
// (the per-block CABAC codec, NOT the c3g GPU path), on real PHercParis4 scroll
// data. Both are independent lossy codecs (mcpp is greenfield, not bit-compatible)
// so we compare RATE (compression ratio) vs DISTORTION (PSNR/MAE/max-error) at
// matched quality.
//
// Volume: a centered 1024^3 region assembled from raw u8 zarr chunks (128^3).
// We SAMPLE blocks (every Nth 16^3 block) for a fast RD curve.

#include "mcpp/codec/block.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/codec/train.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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

// SSIM constants for 8-bit data (standard).
static constexpr double SSIM_C1 = (0.01*255)*(0.01*255);
static constexpr double SSIM_C2 = (0.03*255)*(0.03*255);

struct Stats {
    double sum_se = 0, sum_ae = 0;
    std::uint64_t nvox = 0, bytes = 0, raw = 0;
    std::vector<std::uint8_t> abs_err;   // per-voxel |error|, for percentiles
    double ssim_sum = 0; std::uint64_t ssim_blocks = 0;
    double enc_ns = 0, dec_ns = 0;       // total encode/decode time
    std::uint64_t enc_blocks = 0;

    void add_voxel(int orig, int dec) {
        double e = double(orig) - double(dec);
        sum_se += e*e; sum_ae += std::fabs(e);
        abs_err.push_back(std::uint8_t(std::min(255, int(std::fabs(e)))));
        ++nvox;
    }
    // block SSIM between two 16^3 blocks (mean/var/cov over the whole block).
    void add_block_ssim(const std::uint8_t* a, const std::uint8_t* b, int n) {
        double ma=0, mb=0;
        for (int i=0;i<n;++i){ma+=a[i];mb+=b[i];} ma/=n; mb/=n;
        double va=0, vb=0, cov=0;
        for (int i=0;i<n;++i){double da=a[i]-ma, db=b[i]-mb; va+=da*da; vb+=db*db; cov+=da*db;}
        va/=(n-1); vb/=(n-1); cov/=(n-1);
        double s = ((2*ma*mb+SSIM_C1)*(2*cov+SSIM_C2)) /
                   ((ma*ma+mb*mb+SSIM_C1)*(va+vb+SSIM_C2));
        ssim_sum += s; ++ssim_blocks;
    }
    double psnr() const { double mse=sum_se/double(nvox); return mse>0?10*std::log10(255.0*255.0/mse):999; }
    double mae()  const { return sum_ae/double(nvox); }
    double ratio()const { return double(raw)/double(bytes); }
    double ssim() const { return ssim_blocks?ssim_sum/double(ssim_blocks):0; }
    double pct(double p) {  // p in [0,1]
        if (abs_err.empty()) return 0;
        std::sort(abs_err.begin(), abs_err.end());
        std::size_t i = std::size_t(p*double(abs_err.size()-1));
        return double(abs_err[i]);
    }
    double enc_mbps() const { return enc_ns>0 ? double(raw)/(enc_ns/1e9)/1e6 : 0; }
    double dec_mbps() const { return dec_ns>0 ? double(raw)/(dec_ns/1e9)/1e6 : 0; }
    double enc_lat_us() const { return enc_blocks?enc_ns/double(enc_blocks)/1e3:0; }
    double dec_lat_us() const { return dec_blocks_?dec_ns/double(dec_blocks_)/1e3:0; }
    std::uint64_t dec_blocks_ = 0;
};

using clk = std::chrono::steady_clock;
static double ns_since(clk::time_point t) {
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now()-t).count());
}

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
    const float qualities[] = {1.0f, 3.0f, 6.0f, 12.0f};

    // gather the sampled blocks once (shared across all q).
    std::vector<std::array<std::uint8_t,4096>> samples;
    for (int bz = 0; bz < NB; bz += STRIDE)
    for (int by = 0; by < NB; by += STRIDE)
    for (int bx = 0; bx < NB; bx += STRIDE) {
        std::array<std::uint8_t,4096> b; get_block(vol, bz, by, bx, b.data());
        samples.push_back(b);
    }
    std::fprintf(stderr, "sampled %zu blocks (1/%d of region)\n", samples.size(), STRIDE*STRIDE*STRIDE);

    auto report = [](const char* tag, float q, Stats& s) {
        std::printf("%-5.1f %-10s | %6.2fx %6.2f %.4f | %5.2f %3.0f %3.0f %3.0f %3.0f | %6.0f %6.0f %5.1f %5.1f\n",
            q, tag, s.ratio(), s.psnr(), s.ssim(), s.mae(),
            s.pct(0.90), s.pct(0.95), s.pct(0.99), s.pct(1.0),
            s.enc_mbps(), s.dec_mbps(), s.enc_lat_us(), s.dec_lat_us());
    };

    std::printf("\n%-16s | %-22s | %-21s | %-25s\n", "",
                "rate / fidelity", "abs-error percentiles", "throughput / latency");
    std::printf("%-5s %-10s | %7s %6s %6s | %5s %3s %3s %3s %3s | %6s %6s %5s %5s\n",
        "q","codec","ratio","PSNR","SSIM","MAE","p90","p95","p99","max",
        "encMB","decMB","encUs","decUs");
    std::printf("---------------------------------------------------------------------------------------------------\n");

    for (float q : qualities) {
        mc_codec_ctx_set_quality(ctx, q);
        Stats so, sm, st;

        // train priors over the corpus (this q) — measurement of the gain.
        mcpp::codec::PriorTally tally;

        // ORIGINAL: encode+decode with timing.
        for (auto& b : samples) {
            mc_buf ob{nullptr,0,0}; std::uint32_t olen=0;
            auto t0=clk::now(); mc_enc_block(ctx, b.data(), &ob, &olen); so.enc_ns+=ns_since(t0); ++so.enc_blocks;
            std::uint8_t dec[4096];
            auto t1=clk::now(); mc_dec_block(ctx, ob.p, olen, dec); so.dec_ns+=ns_since(t1); ++so.dec_blocks_;
            so.bytes+=olen; so.raw+=4096;
            for (int i=0;i<4096;++i) so.add_voxel(b[i], dec[i]);
            so.add_block_ssim(b.data(), dec, 4096);
            if (ob.p) free(ob.p);
        }

        // mcpp UNTRAINED + accumulate training tally.
        for (auto& b : samples) {
            std::array<float,4096> f; for (int i=0;i<4096;++i) f[i]=float(b[i]);
            std::vector<std::uint8_t> mb;
            auto t0=clk::now(); mcpp::codec::encode_block<16,3>(f.data(), q, mb); sm.enc_ns+=ns_since(t0); ++sm.enc_blocks;
            std::array<float,4096> fd;
            auto t1=clk::now(); mcpp::codec::decode_block<16,3>(mb.data(), mb.size(), q, fd.data()); sm.dec_ns+=ns_since(t1); ++sm.dec_blocks_;
            sm.bytes+=mb.size(); sm.raw+=4096;
            std::uint8_t dec[4096];
            for (int i=0;i<4096;++i){int d=int(std::lround(fd[i]));d=d<0?0:(d>255?255:d);dec[i]=std::uint8_t(d);sm.add_voxel(b[i],dec[i]);}
            sm.add_block_ssim(b.data(), dec, 4096);
            std::array<float,4096> co=f; mcpp::codec::forward_dct<16,3>(co.data());
            std::array<std::int32_t,4096> qi; mcpp::codec::quantize_block<16,3>(co.data(), qi.data(), q);
            mcpp::codec::train_block<16,3>(tally, qi.data());
        }

        // mcpp TRAINED (frozen-style: one prior table, one-pass encode).
        mcpp::codec::Priors pr = mcpp::codec::finalize_priors(tally);
        for (auto& b : samples) {
            std::array<float,4096> f; for (int i=0;i<4096;++i) f[i]=float(b[i]);
            std::vector<std::uint8_t> mb;
            auto t0=clk::now(); mcpp::codec::encode_block<16,3>(f.data(), q, mb, pr); st.enc_ns+=ns_since(t0); ++st.enc_blocks;
            std::array<float,4096> fd;
            auto t1=clk::now(); mcpp::codec::decode_block<16,3>(mb.data(), mb.size(), q, fd.data(), pr); st.dec_ns+=ns_since(t1); ++st.dec_blocks_;
            st.bytes+=mb.size(); st.raw+=4096;
            std::uint8_t dec[4096];
            for (int i=0;i<4096;++i){int d=int(std::lround(fd[i]));d=d<0?0:(d>255?255:d);dec[i]=std::uint8_t(d);st.add_voxel(b[i],dec[i]);}
            st.add_block_ssim(b.data(), dec, 4096);
        }

        report("original", q, so);
        report("mcpp",      q, sm);
        report("mcpp+train",q, st);
        std::printf("\n");
    }

    mc_codec_ctx_free(ctx);
    return 0;
}
