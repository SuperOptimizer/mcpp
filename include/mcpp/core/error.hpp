// mcpp error model.
//
// Rationale (design memory mcpp-review-rulings): the project failure posture is
// ABORT-AND-DIE, happy-path-only — internal-invariant violations abort early
// (in release too); thorough error handling comes later. The ONE exception is
// streaming external I/O, which must be ROBUST (retry / degrade / skip, never
// abort). So std::expected is used at the fallible boundaries that already need
// graceful handling (notably streaming), not forced everywhere yet.
//
// MCPP_ASSERT is the internal-invariant guard: it aborts in ALL build types
// (a silent invariant breach would corrupt the static mapping / cache state).
#ifndef MCPP_CORE_ERROR_HPP
#define MCPP_CORE_ERROR_HPP

#include <cstdio>
#include <cstdlib>
#include <expected>
#include <string_view>

namespace mcpp {

// Error categories for the robust (non-aborting) boundaries — chiefly streaming
// I/O. Internal logic does NOT return these; it asserts.
enum class Error {
    io_unavailable,    // file / object not accessible
    network,           // transport failure (S3/https)
    not_found,         // remote chunk/object missing
    corrupt,           // integrity check failed on untrusted bytes
    format,            // malformed header / unsupported version
    out_of_range,      // request outside volume bounds
};

constexpr std::string_view error_name(Error e) noexcept {
    switch (e) {
        case Error::io_unavailable: return "io_unavailable";
        case Error::network:        return "network";
        case Error::not_found:      return "not_found";
        case Error::corrupt:        return "corrupt";
        case Error::format:         return "format";
        case Error::out_of_range:   return "out_of_range";
    }
    return "unknown";
}

template <class T>
using Result = std::expected<T, Error>;

// Internal-invariant guard: fail-fast in every build (happy-path posture).
[[noreturn]] inline void abort_invariant(std::string_view msg,
                                         std::string_view file, int line) noexcept {
    std::fprintf(stderr, "mcpp invariant violated: %.*s (%.*s:%d)\n",
                 int(msg.size()), msg.data(),
                 int(file.size()), file.data(), line);
    std::abort();
}

}  // namespace mcpp

#define MCPP_ASSERT(COND, MSG)                                          \
    do {                                                                \
        if (!(COND)) ::mcpp::abort_invariant((MSG), __FILE__, __LINE__);\
    } while (0)

#endif  // MCPP_CORE_ERROR_HPP
