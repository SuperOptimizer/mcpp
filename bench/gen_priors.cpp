// Generate the FROZEN global prior table, once, from representative scroll data.
// Pools context tallies across a sweep of q levels (the shipped table is
// q-independent and seeds every block), then emits a constexpr C++ array to paste
// into coef_coder.hpp as the codec default. One-time/build-time; NO runtime
// calibration ever happens in the codec.
#include "mcpp/codec/dct.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/codec/train.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace mcpp::codec;
static constexpr int N=1024, CH=128, NCH=8, B=16, NB=64;

int main(){
    // assemble + sample
    std::vector<std::uint8_t> vol(std::size_t(N)*N*N,0);
    std::vector<std::uint8_t> cbuf(std::size_t(CH)*CH*CH);
    for (int cz=0;cz<NCH;++cz) for (int cy=0;cy<NCH;++cy) for (int cx=0;cx<NCH;++cx){
        char p[256]; std::snprintf(p,sizeof p,"/tmp/scroll_chunks/%d_%d_%d.bin",21+cz,28+cy,28+cx);
        std::FILE* f=std::fopen(p,"rb"); std::size_t rd=std::fread(cbuf.data(),1,cbuf.size(),f); std::fclose(f); (void)rd;
        for (int lz=0;lz<CH;++lz) for (int ly=0;ly<CH;++ly)
            std::memcpy(&vol[((std::size_t(cz*CH+lz)*N)+(cy*CH+ly))*N+cx*CH], &cbuf[(std::size_t(lz)*CH+ly)*CH], CH);
    }

    PriorTally tally;  // pooled across q
    const float qs[] = {1.0f,2.0f,3.0f,4.0f,6.0f,9.0f,12.0f,20.0f};
    int nb=0;
    for (int bz=0;bz<NB;bz+=2) for (int by=0;by<NB;by+=2) for (int bx=0;bx<NB;bx+=2){
        std::array<float,4096> b;
        for (int lz=0;lz<B;++lz) for (int ly=0;ly<B;++ly) for (int lx=0;lx<B;++lx)
            b[(std::size_t(lz)*B+ly)*B+lx]=float(vol[((std::size_t(bz*B+lz)*N)+(by*B+ly))*N+bx*B+lx]);
        std::array<float,4096> co=b; forward_dct<16,3>(co.data());
        for (float q: qs){
            std::array<std::int32_t,4096> idx;
            quantize_block<16,3>(co.data(), idx.data(), q);
            train_block<16,3>(tally, idx.data());
        }
        ++nb;
    }
    std::fprintf(stderr,"trained on %d blocks x %zu q-levels\n", nb, sizeof(qs)/sizeof(qs[0]));

    Priors pr = finalize_priors(tally);

    // emit constexpr table
    std::printf("// Trained on representative PHercParis4 scroll data (bench/gen_priors).\n");
    std::printf("// Pooled across q={1,2,3,4,6,9,12,20}. Layout: [0,8)=sig [8,16)=eob [16,32)=mag.\n");
    std::printf("inline constexpr Priors kTrainedPriors{{{\n   ");
    for (int i=0;i<kNumPriors;++i){
        std::printf(" %4u,", unsigned(pr.p0[std::size_t(i)]));
        if ((i+1)%8==0) std::printf("\n   ");
    }
    std::printf("\n}}};\n");
    return 0;
}
