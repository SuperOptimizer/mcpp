#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/chunk.hpp"
#include "mcpp/codec/mask_block.hpp"
#include "mcpp/test/test.hpp"

#include <cstdint>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace mcpp;
using namespace mcpp::archive;

static std::string tmp_path(const char* tag) {
    return std::string("/tmp/mcpp_arch_") + tag + "_" + std::to_string(::getpid()) + ".mcp";
}

// ---- chunk format ---------------------------------------------------------

MCPP_TEST("chunk: build/view with mix of present and ALL_ZERO blocks") {
    ChunkBuilder b;
    b.set_quality(4.0f);
    // block 0 present, block 1 ALL_ZERO, block 2 present
    std::vector<std::uint8_t> p0{1, 2, 3, 4, 5};
    std::vector<std::uint8_t> p2{9, 8, 7};
    b.set_block(0, p0);
    b.set_block(1, {});       // ALL_ZERO
    b.set_block(2, p2);
    auto bytes = b.finish();

    ChunkView v(bytes.data(), bytes.size());
    MCPP_CHECK(v.verify());                     // xxh3 integrity
    MCPP_CHECK(v.quality() == 4.0f);
    MCPP_CHECK(v.block_present(0));
    MCPP_CHECK(!v.block_present(1));            // sentinel
    MCPP_CHECK(v.block_present(2));
    auto s0 = v.block(0), s2 = v.block(2);
    MCPP_CHECK(s0.size() == 5 && s0[0] == 1 && s0[4] == 5);
    MCPP_CHECK(s2.size() == 3 && s2[0] == 9 && s2[2] == 7);
    MCPP_CHECK(v.block(1).empty());             // ALL_ZERO -> empty span
}

MCPP_TEST("chunk: corrupted bytes fail xxh3 verify") {
    ChunkBuilder b;
    b.set_block(0, std::vector<std::uint8_t>{1, 2, 3});
    auto bytes = b.finish();
    ChunkView good(bytes.data(), bytes.size());
    MCPP_CHECK(good.verify());
    bytes[bytes.size() - 1] ^= 0xFF;  // flip a payload byte
    ChunkView bad(bytes.data(), bytes.size());
    MCPP_CHECK(!bad.verify());
}

// ---- full archive round-trip ---------------------------------------------

// Encode a single mask-aware block payload for use as block 0 of a chunk.
static std::vector<std::uint8_t> encode_one_block(unsigned seed, float q) {
    constexpr std::size_t N = 16;
    std::vector<float> vox(N * N * N);
    std::vector<std::uint8_t> mask(N * N * N, 1);  // all material
    unsigned s = seed;
    for (auto& v : vox) { s = s * 1664525u + 1013904223u; v = float(1 + ((s >> 8) % 200)); }
    std::vector<std::uint8_t> bytes;
    codec::encode_mask_block_framed<16, 3>(vox.data(), mask.data(), q, bytes);
    return bytes;
}

MCPP_TEST("archive: create -> write chunks -> reopen -> read back") {
    const std::string path = tmp_path("rt");
    // a small volume: 512^3 -> 2 chunks/axis at LOD0
    Geometry geom{{512, 512, 512}, 3};

    // write a couple of chunks at LOD0 and mark one ALL_ZERO.
    auto blk = encode_one_block(123, 4.0f);
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        MCPP_CHECK(a.geometry().num_lods == 3);

        ChunkBuilder cb; cb.set_quality(4.0f);
        cb.set_block(0, blk);              // one present block, rest ALL_ZERO
        auto chunk_bytes = cb.finish();

        a.write_chunk(Lod{0}, 0, 0, 0, chunk_bytes, ChunkState::real_data);
        a.write_chunk(Lod{0}, 1, 1, 1, {}, ChunkState::all_zero);  // confirmed empty
        // (0,0,1) left untouched => DONT_KNOW (sparse default)
    }

    // reopen and verify
    {
        Archive a = Archive::open(path, os::Access::read_only);
        MCPP_CHECK(a.geometry().dims[0] == 512);
        MCPP_CHECK(a.dtype() == Dtype::u8);

        MCPP_CHECK(a.state(Lod{0}, 0, 0, 0) == ChunkState::real_data);
        MCPP_CHECK(a.state(Lod{0}, 1, 1, 1) == ChunkState::all_zero);
        MCPP_CHECK(a.state(Lod{0}, 0, 0, 1) == ChunkState::dont_know);  // untouched

        ChunkView v = a.chunk(Lod{0}, 0, 0, 0);
        MCPP_CHECK(v.verify());                       // integrity survived mmap rt
        MCPP_CHECK(v.block_present(0));
        MCPP_CHECK(!v.block_present(1));
        auto payload = v.block(0);
        MCPP_CHECK(payload.size() == blk.size());

        // decode block 0 back to voxels and sanity-check it's coherent
        std::vector<float> recon(16 * 16 * 16);
        codec::decode_mask_block_framed<16, 3>(payload.data(), payload.size(),
                                               v.quality(), recon.data());
        for (float val : recon) MCPP_CHECK(val >= codec::kMaterialFloor);  // all material
    }
    ::unlink(path.c_str());
}

MCPP_TEST("archive: sparse on disk despite large logical size") {
    const std::string path = tmp_path("sparse");
    Geometry geom{{1024, 1024, 1024}, 4};  // logical size is large
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        auto blk = encode_one_block(7, 4.0f);
        ChunkBuilder cb; cb.set_block(0, blk);
        a.write_chunk(Lod{0}, 0, 0, 0, cb.finish(), ChunkState::real_data);
    }
    struct stat st{};
    ::stat(path.c_str(), &st);
    std::uint64_t allocated = std::uint64_t(st.st_blocks) * 512;
    // wrote one chunk into a multi-GiB logical file -> tiny real allocation.
    MCPP_CHECK(allocated < (std::uint64_t(32) << 20));  // < 32 MiB
    MCPP_CHECK(std::uint64_t(st.st_size) > (std::uint64_t(1) << 30));  // logical > 1 GiB
    ::unlink(path.c_str());
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
