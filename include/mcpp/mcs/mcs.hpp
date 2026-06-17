// .mcs — matter-compressor segments: a TOC of 2D parametric surface blobs.
//
// Rationale (design memory mcpp-everything-else): a .mcs is a LIST of 2D
// parametric surfaces (NOT a single volume). Container = TOC-based:
//   [header][TOC: per-surface {offset,len,gw,gh}][surface blob 0][blob 1]...
// Each surface blob = a codec case-4 compressed grid (DCT-64, C=3 xyz, Identity
// component transform, f32, best-effort error-bounded). Read-mostly / bulk-
// written: produced all at once, editing = rewrite. NOT append-per-surface.
//
// A surface blob here stores the (u,v)->{x,y,z} grid by tiling it into 64^2
// blocks (the 2D codec config) and concatenating block payloads with a small
// per-blob block table. For simplicity + correctness-first, each blob carries its
// own dims and block layout; the fast wire-packing optimization is future work.
#ifndef MCPP_MCS_MCS_HPP
#define MCPP_MCS_MCS_HPP

#include "mcpp/codec/codec.hpp"
#include "mcpp/geom/vec.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mcpp::mcs {

inline constexpr std::uint32_t kMagic   = 0x53434D4Du;  // "MCMS"
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::uint32_t kSurfBlock = 64;  // 2D block side

// One surface = a gw x gh grid of {z,y,x} f32 (codec case 4: C=3). Stored as
// 64^2 tiles; partial edge tiles are zero-padded. The blob is a sequence of
// per-tile [u32 len][payload] using SurfaceCodec<float>.
struct SurfaceDesc {
    std::uint32_t gw = 0, gh = 0;   // grid width (u), height (v)
    float quality = 0.25f;          // tight q for subvoxel coords
};

using SurfacePoints = std::vector<geom::Vec3f>;  // gh*gw, row-major [v*gw+u]

// Build a .mcs byte image from a set of surfaces.
class McsBuilder {
public:
    void add(const SurfaceDesc& desc, const SurfacePoints& pts) {
        surfaces_.push_back({desc, pts});
    }

    std::vector<std::uint8_t> finish() {
        using Cdc = codec::SurfaceCodec<float>;  // 64^2, C=3, Identity
        // encode each surface's tiles into a blob
        struct Blob { std::uint32_t gw, gh; float q; std::vector<std::uint8_t> bytes; };
        std::vector<Blob> blobs;
        for (auto& [d, pts] : surfaces_) {
            Blob b{d.gw, d.gh, d.quality, {}};
            const std::uint32_t tw = (d.gw + kSurfBlock - 1) / kSurfBlock;
            const std::uint32_t th = (d.gh + kSurfBlock - 1) / kSurfBlock;
            for (std::uint32_t ty = 0; ty < th; ++ty)
                for (std::uint32_t tx = 0; tx < tw; ++tx) {
                    // gather a 64x64 tile, interleaved C=3 (z,y,x)
                    std::vector<float> tile(std::size_t(kSurfBlock)*kSurfBlock*3, 0.0f);
                    for (std::uint32_t ly = 0; ly < kSurfBlock; ++ly)
                        for (std::uint32_t lx = 0; lx < kSurfBlock; ++lx) {
                            const std::uint32_t u = tx*kSurfBlock + lx;
                            const std::uint32_t v = ty*kSurfBlock + ly;
                            const std::size_t ti = (std::size_t(ly)*kSurfBlock + lx)*3;
                            if (u < d.gw && v < d.gh) {
                                const geom::Vec3f& p = pts[std::size_t(v)*d.gw + u];
                                tile[ti+0] = p.z; tile[ti+1] = p.y; tile[ti+2] = p.x;
                            }
                        }
                    std::vector<std::uint8_t> pe;
                    Cdc::encode(tile.data(), d.quality, pe);
                    put_u32(b.bytes, std::uint32_t(pe.size()));
                    b.bytes.insert(b.bytes.end(), pe.begin(), pe.end());
                }
            blobs.push_back(std::move(b));
        }

        // assemble: header + TOC + blobs
        const std::uint32_t nsurf = std::uint32_t(blobs.size());
        const std::size_t toc_entry = 4 + 4 + 4 + 4 + 4 + 4;  // off,len,gw,gh,q-bits,pad
        const std::size_t header = 16;
        std::size_t blob_base = header + std::size_t(nsurf) * toc_entry;

        std::vector<std::uint8_t> out;
        // header
        put_u32(out, kMagic); put_u32(out, kVersion);
        put_u32(out, nsurf);  put_u32(out, 0);
        // TOC (compute offsets)
        std::size_t cursor = blob_base;
        for (std::size_t i = 0; i < blobs.size(); ++i) {
            put_u32(out, std::uint32_t(cursor));
            put_u32(out, std::uint32_t(blobs[i].bytes.size()));
            put_u32(out, blobs[i].gw);
            put_u32(out, blobs[i].gh);
            std::uint32_t qb; std::memcpy(&qb, &blobs[i].q, 4);
            put_u32(out, qb);                  // quality bits
            put_u32(out, 0);                   // reserved
            cursor += blobs[i].bytes.size();
        }
        // blobs
        for (auto& b : blobs) out.insert(out.end(), b.bytes.begin(), b.bytes.end());
        return out;
    }

private:
    static void put_u32(std::vector<std::uint8_t>& o, std::uint32_t v) {
        o.push_back(std::uint8_t(v)); o.push_back(std::uint8_t(v>>8));
        o.push_back(std::uint8_t(v>>16)); o.push_back(std::uint8_t(v>>24));
    }
    std::vector<std::pair<SurfaceDesc, SurfacePoints>> surfaces_;
};

// Read a .mcs image.
class McsReader {
public:
    explicit McsReader(const std::uint8_t* data, std::size_t len) : d_(data), len_(len) {
        magic_ = u32(0); version_ = u32(4); nsurf_ = u32(8);
    }
    bool valid() const {
        return len_ >= 16 && magic_ == kMagic && version_ == kVersion;
    }
    std::uint32_t count() const { return nsurf_; }

    struct Entry { std::uint32_t off, len, gw, gh; float q; };
    Entry entry(std::uint32_t i) const {
        const std::size_t base = 16 + std::size_t(i) * 24;
        std::uint32_t qb = u32(base+16); float q; std::memcpy(&q, &qb, 4);
        return Entry{ u32(base), u32(base+4), u32(base+8), u32(base+12), q };
    }

    // Decode surface i back into a points grid.
    SurfacePoints decode(std::uint32_t i) const {
        Entry e = entry(i);
        SurfacePoints pts(std::size_t(e.gw) * e.gh);
        using Cdc = codec::SurfaceCodec<float>;
        const std::uint32_t tw = (e.gw + kSurfBlock - 1) / kSurfBlock;
        const std::uint32_t th = (e.gh + kSurfBlock - 1) / kSurfBlock;
        std::size_t off = e.off;
        for (std::uint32_t ty = 0; ty < th; ++ty)
            for (std::uint32_t tx = 0; tx < tw; ++tx) {
                std::uint32_t plen = u32(off); off += 4;
                std::vector<float> tile(std::size_t(kSurfBlock)*kSurfBlock*3);
                Cdc::decode(d_ + off, plen, e.q, tile.data());
                off += plen;
                for (std::uint32_t ly = 0; ly < kSurfBlock; ++ly)
                    for (std::uint32_t lx = 0; lx < kSurfBlock; ++lx) {
                        const std::uint32_t u = tx*kSurfBlock + lx;
                        const std::uint32_t v = ty*kSurfBlock + ly;
                        if (u < e.gw && v < e.gh) {
                            const std::size_t ti = (std::size_t(ly)*kSurfBlock + lx)*3;
                            pts[std::size_t(v)*e.gw + u] =
                                geom::Vec3f{tile[ti+0], tile[ti+1], tile[ti+2]};
                        }
                    }
            }
        return pts;
    }

private:
    std::uint32_t u32(std::size_t at) const {
        return std::uint32_t(d_[at]) | (std::uint32_t(d_[at+1])<<8) |
               (std::uint32_t(d_[at+2])<<16) | (std::uint32_t(d_[at+3])<<24);
    }
    const std::uint8_t* d_; std::size_t len_;
    std::uint32_t magic_=0, version_=0, nsurf_=0;
};

}  // namespace mcpp::mcs

#endif  // MCPP_MCS_MCS_HPP
