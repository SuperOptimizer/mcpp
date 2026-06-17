// mcpp::os — platform seam for sparse-file mmap.
//
// Rationale (design memory mcpp-archive): static-mapping + sparse-file mmap is
// THE archive design. Platform priority Linux > macOS > Windows; the archive
// logic above this seam is platform-agnostic and assumes sparse-mmap semantics.
// This is the Linux reference implementation (POSIX mmap + ftruncate + madvise);
// macOS is a near-identical POSIX shim, Windows a CreateFileMapping backend, both
// added later behind this same interface.
//
// Failure posture (design memory mcpp-review-rulings): happy-path; open/map
// failures abort (internal-invariant) — this is local-file I/O, not the robust
// streaming path.
#ifndef MCPP_OS_MMAP_HPP
#define MCPP_OS_MMAP_HPP

#include "mcpp/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define MCPP_OS_POSIX 1
#else
#  define MCPP_OS_POSIX 0
#endif

namespace mcpp::os {

inline std::size_t page_size() {
#if MCPP_OS_POSIX
    static const std::size_t ps = std::size_t(::sysconf(_SC_PAGESIZE));
    return ps;
#else
    return 4096;
#endif
}

enum class Access { read_only, read_write };

// A sparse, memory-mapped file. Created at a fixed LOGICAL size via ftruncate;
// untouched pages are holes (zero disk, read as zero). RAII: unmaps + closes on
// destruction. Move-only.
class MappedFile {
public:
    MappedFile() = default;

    // Open `path`, set its logical size to `logical_size` (ftruncate => sparse),
    // and map the whole thing. Creates the file if it doesn't exist.
    static MappedFile open(const std::string& path, std::uint64_t logical_size,
                           Access access) {
#if MCPP_OS_POSIX
        const int oflag = (access == Access::read_write) ? (O_RDWR | O_CREAT) : O_RDONLY;
        int fd = ::open(path.c_str(), oflag, 0644);
        MCPP_ASSERT(fd >= 0, "mcpp::os: open failed");

        if (access == Access::read_write) {
            // ftruncate to the logical size => sparse file. Existing data kept;
            // gap is holes, no blocks allocated, no zeroing.
            int rc = ::ftruncate(fd, off_t(logical_size));
            MCPP_ASSERT(rc == 0, "mcpp::os: ftruncate failed");
        } else {
            struct stat st{};
            int rc = ::fstat(fd, &st);
            MCPP_ASSERT(rc == 0, "mcpp::os: fstat failed");
            logical_size = std::uint64_t(st.st_size);
        }

        const int prot = (access == Access::read_write) ? (PROT_READ | PROT_WRITE) : PROT_READ;
        void* p = ::mmap(nullptr, std::size_t(logical_size), prot, MAP_SHARED, fd, 0);
        MCPP_ASSERT(p != MAP_FAILED, "mcpp::os: mmap failed");

        MappedFile m;
        m.fd_ = fd;
        m.base_ = static_cast<std::uint8_t*>(p);
        m.size_ = logical_size;
        return m;
#else
        (void)path; (void)logical_size; (void)access;
        MCPP_ASSERT(false, "mcpp::os: non-POSIX backend not yet implemented");
        return MappedFile{};
#endif
    }

    ~MappedFile() { reset(); }

    MappedFile(MappedFile&& o) noexcept { swap(o); }
    MappedFile& operator=(MappedFile&& o) noexcept { reset(); swap(o); return *this; }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    std::uint8_t* data() noexcept { return base_; }
    const std::uint8_t* data() const noexcept { return base_; }
    std::uint64_t size() const noexcept { return size_; }
    bool valid() const noexcept { return base_ != nullptr; }

    // Advise the kernel about an access pattern over a byte range.
    enum class Advice { normal, random, sequential, willneed, dontneed };
    void advise(std::uint64_t offset, std::uint64_t length, Advice a) noexcept {
#if MCPP_OS_POSIX
        if (!base_) return;
        int adv = POSIX_MADV_NORMAL;
        switch (a) {
            case Advice::normal:     adv = POSIX_MADV_NORMAL; break;
            case Advice::random:     adv = POSIX_MADV_RANDOM; break;
            case Advice::sequential: adv = POSIX_MADV_SEQUENTIAL; break;
            case Advice::willneed:   adv = POSIX_MADV_WILLNEED; break;
            case Advice::dontneed:   adv = POSIX_MADV_DONTNEED; break;
        }
        ::posix_madvise(base_ + offset, std::size_t(length), adv);
#else
        (void)offset; (void)length; (void)a;
#endif
    }

    // Flush a byte range to disk (best-effort; crash-safety is lowest priority).
    void sync(std::uint64_t offset, std::uint64_t length) noexcept {
#if MCPP_OS_POSIX
        if (base_) ::msync(base_ + offset, std::size_t(length), MS_ASYNC);
#else
        (void)offset; (void)length;
#endif
    }

private:
    void reset() noexcept {
#if MCPP_OS_POSIX
        if (base_) { ::munmap(base_, std::size_t(size_)); base_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        size_ = 0;
    }
    void swap(MappedFile& o) noexcept {
        std::swap(fd_, o.fd_); std::swap(base_, o.base_); std::swap(size_, o.size_);
    }

    int fd_ = -1;
    std::uint8_t* base_ = nullptr;
    std::uint64_t size_ = 0;
};

}  // namespace mcpp::os

#endif  // MCPP_OS_MMAP_HPP
