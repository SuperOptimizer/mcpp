// Cache<T, Capacity> — in-RAM decoded-block cache (BSP dumb store).
//
// Rationale (design memory mcpp-cache + mcpp-architecture):
//  - 3D SINGLE-CHANNEL ONLY. Generic over dtype T; block = 4096*sizeof(T) bytes
//    (= 16^3 samples, always a whole number of 4 KiB pages).
//  - DUMB STORE: the cache decodes/fetches/coordinates NOTHING. It exposes
//    lock-free reads while frozen + a single batched apply(span<FreshPair>) at
//    thaw. Coordinator/IO/dedup/decode all live OUTSIDE.
//  - BSP single model: freeze() => immutable, readers do plain-load lock-free
//    get(); thaw() opens the one mutation window; apply() inserts fresh pairs,
//    NRU-sweep-evicts, then re-freeze. NO locks/atomics-for-sync.
//  - NO pinning: if data must survive eviction, allocate it outside the cache.
//  - Single canonical ZERO block: many keys map to one shared zeroed slot
//    (mostly-air volumes cost ~one slot).
//  - Fixed compile-time Capacity; open-addressed BlockKey->slot map.
//
// apply() is single-caller (the coordinator), once per tick, asserted.
#ifndef MCPP_CACHE_CACHE_HPP
#define MCPP_CACHE_CACHE_HPP

#include "mcpp/cache/block_key.hpp"
#include "mcpp/core/dtype.hpp"
#include "mcpp/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mcpp::cache {

// Result of get_best_lod: a pointer to the best-available block + which LOD it is.
template <Sample T>
struct LodHit {
    const T* ptr;     // nullptr if nothing available at any coarser LOD
    Lod      lod;     // the LOD actually found
};

// A fresh decoded block handed to apply(): either real data or the zero tag.
template <Sample T>
struct FreshPair {
    BlockKey key;
    const T* data;    // pointer to 4096 decoded samples; nullptr => ALL_ZERO
};

template <Sample T, std::size_t Capacity>
class Cache {
public:
    static constexpr std::size_t kSamplesPerBlock = 4096;  // 16^3
    static constexpr std::size_t kMapSlots = Capacity * 2; // open-addr, ~50% load

    Cache() {
        slabs_.resize((Capacity + 1) * kSamplesPerBlock);  // +1 for canonical zero
        // canonical zero block lives at slab index Capacity (last).
        zero_slot_ = Capacity;
        clear_map();
        frozen_ = true;  // start frozen (read-ready)
    }

    // ---- frozen-phase reads (lock-free; no mutation) ----------------------

    // Exact get: returns pointer to the decoded block, or nullptr on miss.
    const T* get(BlockKey k) const {
        const std::size_t s = find(k);
        if (s == kNoSlot) return nullptr;
        return &slabs_[entries_[s].slot * kSamplesPerBlock];
    }

    // Render-now-refine-later: if k misses, probe coarser LODs (coord/2 per level)
    // and return the best available, reporting which LOD. The fine miss is the
    // caller's to record (the cache stays dumb).
    LodHit<T> get_best_lod(BlockKey k) const {
        // try exact first
        if (const T* p = get(k)) return {p, k.lod()};
        std::uint32_t bz = k.bz(), by = k.by(), bx = k.bx();
        for (int l = int(k.lod().v) + 1; l < kMaxLods; ++l) {
            bz >>= 1; by >>= 1; bx >>= 1;
            BlockKey ck = BlockKey::make(Lod{std::uint8_t(l)}, bz, by, bx);
            if (const T* p = get(ck)) return {p, Lod{std::uint8_t(l)}};
        }
        return {nullptr, k.lod()};
    }

    // ---- thaw / mutation window (single-caller, the coordinator) ----------

    void thaw() { frozen_ = false; }

    // Apply a batch of fresh decoded blocks, NRU-sweep-evict, then re-freeze.
    // Single-caller, once per tick. `data` for a pair may be nullptr => zero block.
    void apply(std::span<const FreshPair<T>> pairs) {
        MCPP_ASSERT(!frozen_, "cache: apply() outside thaw window");
        ++tick_;

        for (const auto& fp : pairs) {
            std::size_t slot;
            if (fp.data == nullptr) {
                slot = zero_slot_;              // canonical zero block (shared)
            } else {
                slot = acquire_slot();          // a real data slot (may evict)
                T* dst = &slabs_[slot * kSamplesPerBlock];
                for (std::size_t i = 0; i < kSamplesPerBlock; ++i) dst[i] = fp.data[i];
            }
            insert(fp.key, slot);
        }
        frozen_ = true;  // re-freeze: reads are valid again
    }

    // Bump the access generation of keys the coordinator knows are "hot" (the
    // re-request-implies-hot NRU signal). Called during the thaw window.
    void touch(std::span<const BlockKey> keys) {
        MCPP_ASSERT(!frozen_, "cache: touch() outside thaw window");
        for (BlockKey k : keys) {
            std::size_t s = find(k);
            if (s != kNoSlot) entries_[s].gen = tick_;
        }
    }

    // ---- introspection (tests) -------------------------------------------
    bool frozen() const { return frozen_; }
    std::uint64_t tick() const { return tick_; }
    std::size_t live_count() const { return live_; }

private:
    static constexpr std::size_t kNoSlot = ~std::size_t(0);

    struct Entry {
        BlockKey key = kEmptyKey;
        std::uint32_t slot = 0;    // index into slabs_ (in blocks)
        std::uint64_t gen = 0;     // last-access tick (NRU)
        bool used = false;
    };

    void clear_map() {
        entries_.assign(kMapSlots, Entry{});
        free_list_.clear();
        for (std::uint32_t i = 0; i < Capacity; ++i) free_list_.push_back(i);
        live_ = 0;
    }

    std::size_t probe(BlockKey k) const {
        return std::size_t(k.hash() % kMapSlots);
    }

    // find an existing key -> map-slot index, or kNoSlot.
    std::size_t find(BlockKey k) const {
        std::size_t i = probe(k);
        for (std::size_t n = 0; n < kMapSlots; ++n) {
            const Entry& e = entries_[i];
            if (!e.used) return kNoSlot;          // empty -> not present
            if (e.key == k) return i;
            i = (i + 1) % kMapSlots;
        }
        return kNoSlot;
    }

    // insert or update key -> slot, stamping current tick.
    void insert(BlockKey k, std::size_t slot) {
        std::size_t i = probe(k);
        for (std::size_t n = 0; n < kMapSlots; ++n) {
            Entry& e = entries_[i];
            if (!e.used) {
                e = Entry{k, std::uint32_t(slot), tick_, true};
                ++live_;
                return;
            }
            if (e.key == k) {  // update existing
                e.slot = std::uint32_t(slot);
                e.gen = tick_;
                return;
            }
            i = (i + 1) % kMapSlots;
        }
        MCPP_ASSERT(false, "cache: map full (capacity/load-factor exceeded)");
    }

    // get a free data slot, evicting the NRU (oldest-gen) real entry if needed.
    std::uint32_t acquire_slot() {
        if (!free_list_.empty()) {
            std::uint32_t s = free_list_.back();
            free_list_.pop_back();
            return s;
        }
        // evict: find the map entry with the oldest gen that owns a real slot.
        std::size_t victim = kNoSlot;
        std::uint64_t oldest = ~std::uint64_t(0);
        for (std::size_t i = 0; i < kMapSlots; ++i) {
            Entry& e = entries_[i];
            if (e.used && e.slot != zero_slot_ && e.gen < oldest) {
                oldest = e.gen; victim = i;
            }
        }
        MCPP_ASSERT(victim != kNoSlot, "cache: nothing to evict");
        std::uint32_t s = entries_[victim].slot;
        entries_[victim] = Entry{};  // remove from map
        --live_;
        return s;
    }

    std::vector<T> slabs_;          // (Capacity+1) blocks, contiguous
    std::vector<Entry> entries_;    // open-addressed map
    std::vector<std::uint32_t> free_list_;
    std::size_t zero_slot_ = 0;
    std::size_t live_ = 0;
    std::uint64_t tick_ = 0;
    bool frozen_ = true;
};

}  // namespace mcpp::cache

#endif  // MCPP_CACHE_CACHE_HPP
