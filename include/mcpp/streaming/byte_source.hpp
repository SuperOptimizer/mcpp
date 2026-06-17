// ByteSource — the streaming fetch abstraction.
//
// Rationale (design memory mcpp-streaming): a streaming archive's source of truth
// is remote (S3/https) or another local file. The client fetches WHOLE CHUNKS
// (one ranged read each) into its local sparse .mcp (write-through disk cache).
// ByteSource is the pluggable, concept-constrained read interface:
//
//   read(offset, length) -> Result<bytes>     (robust: returns Error, never abort)
//
// Streaming I/O is the ONE exception to abort-and-die (design memory
// mcpp-review-rulings): transport failure / missing object / OOB returns an
// Error and the caller degrades (retry / leave DONT_KNOW / serve coarse), never
// crashes.
//
// Concrete sources: MemorySource (tests), LocalFileSource (a complete local .mcp
// acting as remote). S3Source / HttpsSource layer libcurl behind this same
// concept later — the streaming logic is source-agnostic.
#ifndef MCPP_STREAMING_BYTE_SOURCE_HPP
#define MCPP_STREAMING_BYTE_SOURCE_HPP

#include "mcpp/core/error.hpp"

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace mcpp::streaming {

using Bytes = std::vector<std::uint8_t>;

// A ByteSource provides ranged reads that fail softly (Result, not abort).
template <class S>
concept ByteSource = requires(S& s, std::uint64_t off, std::uint64_t len) {
    { s.size() } -> std::convertible_to<std::uint64_t>;
    { s.read(off, len) } -> std::same_as<Result<Bytes>>;
};

// In-memory source (tests, and a model for the robust contract).
class MemorySource {
public:
    explicit MemorySource(Bytes data) : data_(std::move(data)) {}
    std::uint64_t size() const { return data_.size(); }
    Result<Bytes> read(std::uint64_t off, std::uint64_t len) {
        if (fail_next_) { fail_next_ = false; return std::unexpected(Error::network); }
        if (off + len > data_.size()) return std::unexpected(Error::out_of_range);
        return Bytes(data_.begin() + std::ptrdiff_t(off),
                     data_.begin() + std::ptrdiff_t(off + len));
    }
    // test hook: simulate a transient failure on the next read
    void fail_next() { fail_next_ = true; }
private:
    Bytes data_;
    bool fail_next_ = false;
};

// Local-file source: a complete .mcp file standing in as the "remote". Robust —
// open/read failures return Error, never abort (this is the streaming path).
class LocalFileSource {
public:
    static Result<LocalFileSource> open(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return std::unexpected(Error::io_unavailable);
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        if (sz < 0) { std::fclose(f); return std::unexpected(Error::io_unavailable); }
        LocalFileSource s;
        s.f_ = f;
        s.size_ = std::uint64_t(sz);
        return s;
    }
    LocalFileSource() = default;
    LocalFileSource(LocalFileSource&& o) noexcept { swap(o); }
    LocalFileSource& operator=(LocalFileSource&& o) noexcept { close(); swap(o); return *this; }
    LocalFileSource(const LocalFileSource&) = delete;
    LocalFileSource& operator=(const LocalFileSource&) = delete;
    ~LocalFileSource() { close(); }

    std::uint64_t size() const { return size_; }

    Result<Bytes> read(std::uint64_t off, std::uint64_t len) {
        if (off + len > size_) return std::unexpected(Error::out_of_range);
        if (std::fseek(f_, long(off), SEEK_SET) != 0)
            return std::unexpected(Error::io_unavailable);
        Bytes out(len);
        std::size_t got = std::fread(out.data(), 1, len, f_);
        if (got != len) return std::unexpected(Error::io_unavailable);
        return out;
    }

private:
    void close() { if (f_) { std::fclose(f_); f_ = nullptr; } }
    void swap(LocalFileSource& o) { std::swap(f_, o.f_); std::swap(size_, o.size_); }
    std::FILE* f_ = nullptr;
    std::uint64_t size_ = 0;
};

static_assert(ByteSource<LocalFileSource>);
static_assert(ByteSource<MemorySource>);

}  // namespace mcpp::streaming

#endif  // MCPP_STREAMING_BYTE_SOURCE_HPP
