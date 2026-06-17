// RD search for the quant constants (dead-zone fraction, band-weight exponent,
// reconstruction offset) on representative scroll data. For each candidate, we
// emulate the quant/dequant with the candidate params, run the full
// transform->quant->entropy(size estimate)->dequant->inverse pipeline on the
// sampled blocks, and score by PSNR at a target rate. The winner is baked into
// quant.hpp as the new constants.
//
// Rate is estimated as the real entropy-coded size (encode_block uses the live
// quant constants, so to search params we recompute quant here with the
// candidate and entropy-code the resulting indices directly).

#include "mcpp/codec/coef_coder.hpp"
#include "mcpp/codec/dct.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/codec/range_coder.hpp"
#include "mcpp/codec/scan.hpp"

#include <cstring>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace mcpp::codec;

static constexpr int N=1024, CH=128, NCH=8, B=16, NB=64;

static std::vector<std::uint8_t> assemble() {
    std::vector<std::uint8_t> vol(std::size_t(N)*N*N,0);
    std::vector<std::uint8_t> cbuf(std::size_t(CH)*CH*CH);
    for (int cz=0;cz<NCH;++cz) for (int cy=0;cy<NCH;++cy) for (int cx=0;cx<NCH;++cx){
        char p[256]; std::snprintf(p,sizeof p,"/tmp/scroll_chunks/%d_%d_%d.bin",21+cz,28+cy,28+cx);
        std::FILE* f=std::fopen(p,"rb"); std::size_t rd=std::fread(cbuf.data(),1,cbuf.size(),f); std::fclose(f); (void)rd;
        for (int lz=0;lz<CH;++lz) for (int ly=0;ly<CH;++ly)
            std::memcpy(&vol[((std::size_t(cz*CH+lz)*N)+(cy*CH+ly))*N+cx*CH], &cbuf[(std::size_t(lz)*CH+ly)*CH], CH);
    }
    return vol;
}

// candidate quant params
struct QP { float dz, hf_exp, recon; };

// quant one coeff with candidate params (mirrors quant.hpp quantize/dequantize)
static std::int32_t cquant(float c, float step, const QP& p) {
    if (step<=0) return 0;
    float a=std::fabs(c)/step;
    if (a < p.dz) return 0;
    std::int32_t mag = std::int32_t((a - p.dz) + 1.0f);
    return c<0 ? -mag : mag;
}
static float cdequant(std::int32_t i, float step, const QP& p) {
    if (i==0) return 0;
    std::int32_t m = i<0?-i:i;
    float r = (float(m-1) + p.dz + p.recon) * step;
    return i<0 ? -r : r;
}
static float cstep(int band, float q, const QP& p) {
    float nf = float(band)/float(max_band<16,3>());
    return kBaseStep * q * (1.0f + std::pow(nf, p.hf_exp)*3.0f);
}

// evaluate (ratio, psnr) for candidate params over the sampled blocks at q
static void eval(const std::vector<std::array<float,4096>>& blocks, float q,
                 const QP& p, double& ratio, double& psnr) {
    std::uint64_t bytes=0; double se=0; std::uint64_t nv=0;
    for (auto& fb : blocks) {
        std::array<float,4096> co = fb;
        forward_dct<16,3>(co.data());
        std::array<std::int32_t,4096> idx;
        for (std::size_t i=0;i<4096;++i){
            int band=coeff_band<16,3>(i);
            idx[i]=cquant(co[i], cstep(band,q,p), p);
        }
        // entropy size (default priors — params don't change the model)
        std::vector<std::uint8_t> out; RangeEncoder enc(out); CoefModel m;
        encode_coeffs<16,3>(enc,m,idx.data()); enc.finish();
        bytes += out.size();
        // dequant + inverse
        std::array<float,4096> rc;
        for (std::size_t i=0;i<4096;++i){int band=coeff_band<16,3>(i); rc[i]=cdequant(idx[i], cstep(band,q,p), p);}
        inverse_dct<16,3>(rc.data());
        for (std::size_t i=0;i<4096;++i){double e=double(fb[i])-double(std::lround(rc[i])); se+=e*e; ++nv;}
    }
    ratio = double(nv)/double(bytes);
    psnr = 10*std::log10(255.0*255.0/(se/double(nv)));
}

int main(){
    auto vol = assemble();
    std::vector<std::array<float,4096>> blocks;
    for (int bz=0;bz<NB;bz+=4) for (int by=0;by<NB;by+=4) for (int bx=0;bx<NB;bx+=4){
        std::array<float,4096> b;
        for (int lz=0;lz<B;++lz) for (int ly=0;ly<B;++ly) for (int lx=0;lx<B;++lx)
            b[(std::size_t(lz)*B+ly)*B+lx]=float(vol[((std::size_t(bz*B+lz)*N)+(by*B+ly))*N+bx*B+lx]);
        blocks.push_back(b);
    }
    std::fprintf(stderr,"%zu blocks\n",blocks.size());

    // baseline (current constants)
    QP base{0.80f, 0.65f, 0.40f};
    std::printf("current: dz=%.2f hf=%.2f recon=%.2f\n", base.dz, base.hf_exp, base.recon);

    // For RD comparison we score each candidate by its PSNR at the rate the
    // BASELINE achieves (interpolate q so ratio matches), summed over a few q.
    // Simpler proxy: at fixed q's, a candidate is better if it gives higher PSNR
    // AND not-worse ratio (Pareto). We aggregate a BD-rate-like score: average
    // PSNR gain at matched-ish q across the sweep, penalizing ratio loss.
    const float qs[] = {1.0f, 3.0f, 6.0f, 12.0f};

    // Proper objective: BD-rate-style PSNR-at-matched-ratio vs the baseline.
    // We evaluate baseline RD points once, then for a candidate compute its PSNR
    // interpolated to each baseline ratio and average the delta. Positive = the
    // candidate's RD curve is above the baseline's (a real improvement).
    std::vector<std::pair<double,double>> base_rd;  // (ratio, psnr)
    for (float q: qs){ double r,ps; eval(blocks,q,base,r,ps); base_rd.push_back({r,ps}); }

    auto psnr_at = [](std::vector<std::pair<double,double>>& c, double r)->double {
        // c sorted by ratio ascending (qs ascending -> ratio ascending)
        if (r<=c.front().first) return c.front().second;
        if (r>=c.back().first)  return c.back().second;
        for (std::size_t i=1;i<c.size();++i) if (r<=c[i].first){
            double t=(r-c[i-1].first)/(c[i].first-c[i-1].first);
            return c[i-1].second + t*(c[i].second-c[i-1].second);
        }
        return c.back().second;
    };
    auto bd_gain = [&](const QP& p)->double {
        std::vector<std::pair<double,double>> rd;
        for (float q: qs){ double r,ps; eval(blocks,q,p,r,ps); rd.push_back({r,ps}); }
        // average PSNR delta at the baseline's ratios
        double s=0; int n=0;
        for (auto& [r,pb] : base_rd){ s += psnr_at(rd, r) - pb; ++n; }
        return s/n;
    };

    std::printf("baseline RD captured (%zu points)\n", base_rd.size());

    QP best = base; double best_g = 0.0;
    // extended grid (push past the prior edges dz=0.5, hf=1.0)
    for (float dz : {0.20f,0.30f,0.40f,0.50f,0.60f,0.70f,0.80f})
    for (float hf : {0.65f,0.85f,1.00f,1.25f,1.50f,2.00f})
    for (float rc : {0.30f,0.40f,0.50f,0.60f}) {
        QP p{dz,hf,rc}; double g=bd_gain(p);
        if (g > best_g) { best_g=g; best=p; }
    }
    double best_s = best_g;  // (reuse var name in the print below)
    std::printf("BEST: dz=%.2f hf=%.2f recon=%.2f  BD-PSNR gain=%+.3f dB\n", best.dz, best.hf_exp, best.recon, best_g);

    // report RD at each q for baseline vs best
    std::printf("\n   q | baseline ratio/psnr | best ratio/psnr\n");
    for (float q: qs){
        double rb,pb,rB,pB; eval(blocks,q,base,rb,pb); eval(blocks,q,best,rB,pB);
        std::printf("%4.0f | %7.2fx %6.2f      | %7.2fx %6.2f\n", q, rb,pb, rB,pB);
    }
    return 0;
}
