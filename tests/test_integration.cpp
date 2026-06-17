// Full-stack integration: ingest -> archive -> cache(via coordinator) -> sampler
// -> renderer. Proves every subsystem composes end-to-end on one volume.
#include "mcpp/archive/archive.hpp"
#include "mcpp/cache/cache.hpp"
#include "mcpp/coordinator/coordinator.hpp"
#include "mcpp/geom/surface.hpp"
#include "mcpp/ingest/ingest.hpp"
#include "mcpp/render/compositor.hpp"
#include "mcpp/render/renderer.hpp"
#include "mcpp/sampling/sampler.hpp"
#include "mcpp/sampling/volume_view.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <vector>

using namespace mcpp;

static std::string tmp_path() {
    return "/tmp/mcpp_integ_" + std::to_string(::getpid()) + ".mcp";
}

MCPP_TEST("integration: ingest -> archive -> coordinator/cache -> sample -> render") {
    // 1. synthesize a volume: bright slab z in [120,136], value 180; air elsewhere.
    const std::uint32_t n = 256;
    std::vector<std::uint8_t> vol(std::size_t(n)*n*n, 0);
    for (std::uint32_t z = 120; z < 136; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x)
                vol[(std::size_t(z)*n + y)*n + x] = 180;

    // 2. INGEST -> archive (mask, encode, single LOD).
    const std::string path = tmp_path();
    archive::Archive arc =
        ingest::ingest_volume(path, vol, {n,n,n}, /*num_lods*/1, /*thresh*/10, /*q*/2.0f);

    // reopen read-only (the consumer's view)
    archive::Archive a = archive::Archive::open(path, os::Access::read_only);

    // 3. CACHE + COORDINATOR: a consumer samples; misses are resolved by the
    //    coordinator (decode from archive -> apply to cache) in a tick.
    constexpr std::size_t Cap = 8192;
    cache::Cache<std::uint8_t, Cap> cache;
    coordinator::Coordinator<std::uint8_t, Cap> coord(a, cache);

    geom::PlaneSurface plane({128,64,64}, {0,0,1}, {0,1,0}, {1,0,0}, 32, 32);
    render::RenderParams rp{-2.0f, 2.0f, 1.0f};  // march z in [126,130], in slab

    // 4. FRAME 1 (cold): render collects the misses for exactly the blocks it
    //    touches. (render-now-refine-later: image empty this frame.)
    std::vector<cache::BlockKey> misses;
    {
        sampling::CachedView<std::uint8_t, Cap> view(cache, {n,n,n}, Lod{0}, &misses);
        sampling::Sampler<sampling::CachedView<std::uint8_t,Cap>> smp(view);
        std::vector<float> cold;
        mcpp::render::render<geom::PlaneSurface, sampling::CachedView<std::uint8_t,Cap>,
                             render::MaxComposite>(plane, smp, rp, cold);
    }
    MCPP_CHECK(!misses.empty());  // cold cache -> misses recorded

    // 5. coordinator resolves the render's misses (the BSP tick boundary).
    coord.resolve(misses);

    // 6. FRAME 2 (warm): re-render; the blocks the render needs are now resident.
    std::vector<cache::BlockKey> misses2;
    sampling::CachedView<std::uint8_t, Cap> rview(cache, {n,n,n}, Lod{0}, &misses2);
    sampling::Sampler<sampling::CachedView<std::uint8_t,Cap>> rsmp(rview);
    std::vector<float> img;
    mcpp::render::render<geom::PlaneSurface, sampling::CachedView<std::uint8_t,Cap>,
                         render::MaxComposite>(plane, rsmp, rp, img);

    // 7. verify: the rendered plane (inside the bright slab, cache warm) shows the
    //    slab value where blocks are resident.
    MCPP_CHECK(img.size() == 32u*32u);
    float maxpix = 0;
    for (float px : img) maxpix = std::max(maxpix, px);
    MCPP_CHECK(maxpix > 150.0f);  // the bright slab (180) came through the full stack

    ::unlink(path.c_str());
}

MCPP_TEST("integration: cold render misses, warm render hits (render-now-refine-later)") {
    const std::uint32_t n = 256;
    std::vector<std::uint8_t> vol(std::size_t(n)*n*n, 0);
    for (std::uint32_t z = 120; z < 136; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x)
                vol[(std::size_t(z)*n + y)*n + x] = 180;
    const std::string path = tmp_path();
    ingest::ingest_volume(path, vol, {n,n,n}, 1, 10, 2.0f);
    archive::Archive a = archive::Archive::open(path, os::Access::read_only);

    constexpr std::size_t Cap = 8192;
    cache::Cache<std::uint8_t, Cap> cache;
    coordinator::Coordinator<std::uint8_t, Cap> coord(a, cache);

    geom::PlaneSurface plane({128,64,64}, {0,0,1}, {0,1,0}, {1,0,0}, 16, 16);
    render::RenderParams rp{0,0,1};  // single sample at z=128

    // FRAME 1 (cold): render collects misses; image is mostly empty.
    std::vector<cache::BlockKey> m1;
    {
        sampling::CachedView<std::uint8_t,Cap> v(cache, {n,n,n}, Lod{0}, &m1);
        sampling::Sampler<sampling::CachedView<std::uint8_t,Cap>> s(v);
        std::vector<float> img;
        mcpp::render::render<geom::PlaneSurface, sampling::CachedView<std::uint8_t,Cap>,
                             render::MaxComposite>(plane, s, rp, img);
        float mx = 0; for (float p : img) mx = std::max(mx, p);
        MCPP_CHECK(mx == 0.0f);          // cold: nothing resident yet
    }
    MCPP_CHECK(!m1.empty());

    // TICK: resolve.
    coord.resolve(m1);

    // FRAME 2 (warm): same render now shows the slab.
    {
        std::vector<cache::BlockKey> m2;
        sampling::CachedView<std::uint8_t,Cap> v(cache, {n,n,n}, Lod{0}, &m2);
        sampling::Sampler<sampling::CachedView<std::uint8_t,Cap>> s(v);
        std::vector<float> img;
        mcpp::render::render<geom::PlaneSurface, sampling::CachedView<std::uint8_t,Cap>,
                             render::MaxComposite>(plane, s, rp, img);
        float mx = 0; for (float p : img) mx = std::max(mx, p);
        MCPP_CHECK(mx > 150.0f);         // warm: slab visible (refine-later worked)
    }
    ::unlink(path.c_str());
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
