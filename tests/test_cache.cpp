#include "mcpp/cache/block_key.hpp"
#include "mcpp/cache/cache.hpp"
#include "mcpp/test/test.hpp"

#include <array>
#include <cstdint>
#include <vector>

using namespace mcpp;
using namespace mcpp::cache;

// ---- BlockKey -------------------------------------------------------------

MCPP_TEST("key: pack/unpack round-trips all fields") {
    BlockKey k = BlockKey::make(Lod{5}, 1000, 2000, 60000);
    MCPP_CHECK(k.lod().v == 5);
    MCPP_CHECK(k.bz() == 1000);
    MCPP_CHECK(k.by() == 2000);
    MCPP_CHECK(k.bx() == 60000);
    MCPP_CHECK(sizeof(BlockKey) == 8);
}

MCPP_TEST("key: distinct keys differ; equal keys compare equal") {
    BlockKey a = BlockKey::make(Lod{0}, 1, 2, 3);
    BlockKey b = BlockKey::make(Lod{0}, 1, 2, 3);
    BlockKey c = BlockKey::make(Lod{0}, 1, 2, 4);
    MCPP_CHECK(a == b);
    MCPP_CHECK(!(a == c));
    MCPP_CHECK(a.hash() != c.hash());  // good avalanche (overwhelmingly likely)
}

// ---- cache helpers --------------------------------------------------------

template <class C>
static std::vector<std::uint8_t> block_bytes(std::uint8_t fill) {
    return std::vector<std::uint8_t>(C::kSamplesPerBlock, fill);
}

// ---- cache BSP semantics --------------------------------------------------

MCPP_TEST("cache: starts frozen; miss returns nullptr") {
    Cache<std::uint8_t, 16> c;
    MCPP_CHECK(c.frozen());
    MCPP_CHECK(c.get(BlockKey::make(Lod{0}, 0, 0, 0)) == nullptr);
}

MCPP_TEST("cache: thaw -> apply -> frozen read returns the block") {
    Cache<std::uint8_t, 16> c;
    auto data = block_bytes<Cache<std::uint8_t, 16>>(42);
    BlockKey k = BlockKey::make(Lod{0}, 3, 4, 5);

    c.thaw();
    MCPP_CHECK(!c.frozen());
    std::array<FreshPair<std::uint8_t>, 1> batch{{ {k, data.data()} }};
    c.apply(batch);
    MCPP_CHECK(c.frozen());  // re-frozen after apply

    const std::uint8_t* p = c.get(k);
    MCPP_CHECK(p != nullptr);
    MCPP_CHECK(p[0] == 42 && p[4095] == 42);
    MCPP_CHECK(c.live_count() == 1);
}

MCPP_TEST("cache: canonical zero block — many keys alias one slot") {
    Cache<std::uint8_t, 16> c;
    c.thaw();
    std::vector<FreshPair<std::uint8_t>> batch;
    for (std::uint32_t i = 0; i < 8; ++i)
        batch.push_back({BlockKey::make(Lod{0}, i, 0, 0), nullptr});  // ALL_ZERO
    c.apply(batch);

    // all 8 keys resolve to the same (zero) pointer, all zeros, no slabs consumed
    const std::uint8_t* first = c.get(BlockKey::make(Lod{0}, 0, 0, 0));
    MCPP_CHECK(first != nullptr);
    for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint8_t* p = c.get(BlockKey::make(Lod{0}, i, 0, 0));
        MCPP_CHECK(p == first);     // aliased to the same canonical slot
        MCPP_CHECK(p[0] == 0 && p[2048] == 0);
    }
}

MCPP_TEST("cache: NRU eviction frees oldest when over capacity") {
    Cache<std::uint8_t, 4> c;  // capacity 4 real blocks

    // tick 1: insert 4 blocks (fills capacity)
    c.thaw();
    std::vector<std::vector<std::uint8_t>> store;
    std::vector<FreshPair<std::uint8_t>> b1;
    for (std::uint32_t i = 0; i < 4; ++i) {
        store.push_back(block_bytes<Cache<std::uint8_t,4>>(std::uint8_t(i + 1)));
        b1.push_back({BlockKey::make(Lod{0}, i, 0, 0), store.back().data()});
    }
    c.apply(b1);
    MCPP_CHECK(c.live_count() == 4);

    // tick 2: re-touch blocks 2,3 (mark hot), then insert 2 new blocks ->
    // must evict the oldest untouched ones (0,1), keep 2,3.
    c.thaw();
    std::array<BlockKey, 2> hot{{ BlockKey::make(Lod{0},2,0,0),
                                  BlockKey::make(Lod{0},3,0,0) }};
    c.touch(hot);
    std::vector<std::vector<std::uint8_t>> store2;
    std::vector<FreshPair<std::uint8_t>> b2;
    for (std::uint32_t i = 4; i < 6; ++i) {
        store2.push_back(block_bytes<Cache<std::uint8_t,4>>(std::uint8_t(i + 1)));
        b2.push_back({BlockKey::make(Lod{0}, i, 0, 0), store2.back().data()});
    }
    c.apply(b2);

    // 2,3 (touched) survive; 4,5 (new) present; 0,1 evicted.
    MCPP_CHECK(c.get(BlockKey::make(Lod{0},2,0,0)) != nullptr);
    MCPP_CHECK(c.get(BlockKey::make(Lod{0},3,0,0)) != nullptr);
    MCPP_CHECK(c.get(BlockKey::make(Lod{0},4,0,0)) != nullptr);
    MCPP_CHECK(c.get(BlockKey::make(Lod{0},5,0,0)) != nullptr);
    MCPP_CHECK(c.get(BlockKey::make(Lod{0},0,0,0)) == nullptr);  // evicted
    MCPP_CHECK(c.get(BlockKey::make(Lod{0},1,0,0)) == nullptr);  // evicted
    MCPP_CHECK(c.live_count() == 4);
}

MCPP_TEST("cache: get_best_lod falls back to coarser LOD") {
    Cache<std::uint8_t, 16> c;
    // populate only LOD 2 for the region covering fine block (0, 8, 8, 8)
    auto data = block_bytes<Cache<std::uint8_t,16>>(99);
    c.thaw();
    // fine key lod0 (bz,by,bx)=(8,8,8) -> lod1 (4,4,4) -> lod2 (2,2,2)
    BlockKey coarse = BlockKey::make(Lod{2}, 2, 2, 2);
    std::array<FreshPair<std::uint8_t>,1> batch{{ {coarse, data.data()} }};
    c.apply(batch);

    BlockKey fine = BlockKey::make(Lod{0}, 8, 8, 8);
    MCPP_CHECK(c.get(fine) == nullptr);            // exact miss
    LodHit<std::uint8_t> h = c.get_best_lod(fine);
    MCPP_CHECK(h.ptr != nullptr);                  // coarse fallback found
    MCPP_CHECK(h.lod.v == 2);                      // reported the LOD it used
    MCPP_CHECK(h.ptr[0] == 99);
}

MCPP_TEST("cache: update existing key replaces its block") {
    Cache<std::uint8_t, 16> c;
    BlockKey k = BlockKey::make(Lod{1}, 7, 7, 7);
    auto d1 = block_bytes<Cache<std::uint8_t,16>>(10);
    auto d2 = block_bytes<Cache<std::uint8_t,16>>(20);

    c.thaw(); { std::array<FreshPair<std::uint8_t>,1> b{{ {k, d1.data()} }}; c.apply(b); }
    MCPP_CHECK(c.get(k)[0] == 10);
    c.thaw(); { std::array<FreshPair<std::uint8_t>,1> b{{ {k, d2.data()} }}; c.apply(b); }
    MCPP_CHECK(c.get(k)[0] == 20);
    MCPP_CHECK(c.live_count() == 1);  // still one entry
}

MCPP_TEST("cache: u16 instantiation (block = 8 KiB)") {
    Cache<std::uint16_t, 8> c;
    std::vector<std::uint16_t> data(Cache<std::uint16_t,8>::kSamplesPerBlock, 0xBEEF);
    BlockKey k = BlockKey::make(Lod{0}, 1, 1, 1);
    c.thaw(); { std::array<FreshPair<std::uint16_t>,1> b{{ {k, data.data()} }}; c.apply(b); }
    const std::uint16_t* p = c.get(k);
    MCPP_CHECK(p != nullptr && p[0] == 0xBEEF && p[4095] == 0xBEEF);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
