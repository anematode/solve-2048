#include <iostream>
#include <chrono>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/parallel_for.h>
#include <zstd.h>
#include <zstd_errors.h>
#include <omp.h>
#include <execution>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "StupidHashMap.h"
#include "Position.h"

thread_local std::vector<uint64_t> next_tl;
thread_local std::vector<uint64_t> next_tl2;

#define SHOW_TIMINGS 1

template <typename Func>
void timed_run(const std::string& label, Func&& f) {
#if SHOW_TIMINGS
    auto start = std::chrono::high_resolution_clock::now();
#endif
    f();  // call the function
#if SHOW_TIMINGS
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << label << " took " << elapsed.count() << " seconds.\n";
#endif
}

bool compress_data(const char* data, size_t bytes, const std::string& filename) {
    // Estimate max compressed size
    size_t maxCompressedSize = ZSTD_compressBound(bytes);

    // Open and truncate file to maxCompressedSize
    int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return false;
    }
    if (ftruncate(fd, maxCompressedSize) != 0) {
        perror("ftruncate");
        close(fd);
        return false;
    }

    // mmap the output file
    void* mmap_ptr = mmap(nullptr, maxCompressedSize, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return false;
    }

    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) {
        std::cerr << "Failed to create ZSTD_CCtx\n";
        munmap(mmap_ptr, maxCompressedSize);
        close(fd);
        return false;
    }

    // Fastest compression level
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, 1);

    // Use all hardware threads
    unsigned threads = std::thread::hardware_concurrency();
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, threads);

    std::cout << "Compressing " << bytes / (1024 * 1024) << " MB using " << threads << " threads...\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Perform compression directly into the mmap'd buffer
    size_t compressedSize = ZSTD_compress2(
        cctx,
        mmap_ptr, maxCompressedSize,
        data, bytes
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (ZSTD_isError(compressedSize)) {
        std::cerr << "Compression failed: " << ZSTD_getErrorName(compressedSize) << "\n";
        ZSTD_freeCCtx(cctx);
        munmap(mmap_ptr, maxCompressedSize);
        close(fd);
        return false;
    }

    // Truncate file to actual compressed size
    if (ftruncate(fd, compressedSize) != 0) {
        perror("ftruncate to final size");
    }

    munmap(mmap_ptr, maxCompressedSize);
    close(fd);
    ZSTD_freeCCtx(cctx);

    std::cout << "Compression done in " << elapsed.count() << " seconds\n";
    std::cout << "Original size: " << bytes << " bytes\n";
    std::cout << "Compressed size: " << compressedSize << " bytes\n";
    std::cout << "Compression ratio: " << static_cast<double>(bytes) / compressedSize << "\n";

    return true;
}


struct Vectorish {
    virtual uint64_t& operator[] (size_t index) = 0;
    virtual size_t size();
    virtual ~Vectorish();
};

struct StdVectorWrapper : public Vectorish {
    std::vector<uint64_t> contents;

    uint64_t& operator[] (size_t index) override {
        return contents[index];
    }
    size_t size() override {
        return contents.size();
    }
    void push_back(uint64_t data) {
        contents.push_back(data);
    }
};

struct FixedSizeBackedWrapper : public Vectorish {
    std::string filename; // Stores the name of the temporary file
    uint64_t *mapped;     // Pointer to the memory-mapped region
    size_t size_;         // Logical size (number of uint64_t elements)
    int fd;               // File descriptor for the temporary file

    // Constructor: Initializes the wrapper with a specified size,
    // creating and mapping a temporary file.
    FixedSizeBackedWrapper(size_t size) : mapped(nullptr), size_(0), fd(-1) {
        if (size == 0) {
            size_ = 0;
            return;
        }
        size_t bytes_to_map;
        if (__builtin_mul_overflow(size, sizeof(uint64_t), &bytes_to_map)) {
            throw std::runtime_error("FixedSizeBackedWrapper: Size calculation overflow, requested size too large.");
        }

        char temp_filename_template[] = "/tmp/fixed_backed_XXXXXX"; // Template for mkstemp
        fd = mkstemp(temp_filename_template);
        if (fd == -1) {
            throw std::runtime_error("FixedSizeBackedWrapper: Failed to create temporary file.");
        }
        filename = temp_filename_template;
        if (ftruncate(fd, bytes_to_map) == -1) {
            close(fd);
            unlink(filename.c_str());
            throw std::runtime_error("FixedSizeBackedWrapper: Failed to set file size with ftruncate.");
        }

        mapped = static_cast<uint64_t*>(mmap(nullptr, bytes_to_map, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (mapped == MAP_FAILED) {
            close(fd);
            unlink(filename.c_str());
            throw std::runtime_error("FixedSizeBackedWrapper: Failed to mmap file.");
        }

        size_ = size;
    }

    uint64_t *begin() {
        return mapped;
    }

    uint64_t *end() {
        return mapped + size_;
    }

    uint64_t& operator[] (size_t index) override {
        if (index >= size_) {
            throw std::out_of_range("FixedSizeBackedWrapper: Index out of bounds");
        }
        return mapped[index];
    }

    size_t size() override {
        return size_;
    }

    ~FixedSizeBackedWrapper() override {
        if (mapped != nullptr && mapped != MAP_FAILED) {
            size_t bytes_to_unmap = 0;
            if (size_ > 0) {
                 bytes_to_unmap = size_ * sizeof(uint64_t);
            }
            if (bytes_to_unmap > 0) {
                if (munmap(mapped, bytes_to_unmap) == -1) {
                    std::cerr << "Warning: Failed to munmap memory in FixedSizeBackedWrapper destructor." << std::endl;
                }
            }
        }

        // Close the file descriptor if it's valid.
        if (fd != -1) {
            if (close(fd) == -1) {
                 std::cerr << "Warning: Failed to close file descriptor in FixedSizeBackedWrapper destructor." << std::endl;
            }
        }

        // Unlink (delete) the temporary file if its name is known.
        if (!filename.empty()) {
            if (unlink(filename.c_str()) == -1) {
                std::cerr << "Warning: Failed to unlink temporary file in FixedSizeBackedWrapper destructor." << std::endl;
            }
        }
    }
};

void census(const std::vector<uint64_t> & h2, int tile_sum) {
    uint64_t max_2 = 0, max_4 = 0, max_8 = 0, max_16 = 0, max_32 = 0, max_64 = 0, max_128 = 0, max_256 = 0, max_512 = 0;
#pragma omp parallel for reduction(+:max_2,max_4,max_8,max_16,max_32,max_64,max_128,max_256,max_512)
    for (size_t i = 0; i < h2.size(); ++i) {
        auto tile = max_tile(h2[i]);
        max_2 += tile == 1;
        max_4 += tile == 2;
        max_8 += tile == 3;
        max_16 += tile == 4;
        max_32 += tile == 5;
        max_64 += tile == 6;
        max_128 += tile == 7;
        max_256 += tile == 8;
        max_512 += tile == 9;
    }
    std::cout << "Census for tile sum " << tile_sum << ": " << max_2 << ',' << max_4 << ',' <<
         max_8 << ',' << max_16 << ',' << max_32 << ',' << max_64 << ',' << max_128 << ',' << max_256 << ',' << max_512 << '\n';
}

int main()
{
    uint32_t h1_tile_sum = 4;  // 4, 6, 8
omp_set_num_threads(omp_get_max_threads());

    std::vector<uint64_t>  h1, h2, h3;
    std::vector<uint64_t> all = starting_positions();

    for (auto b : all) {
        auto sum = tile_sum(b);
        (sum == 4 ? h1 : sum == 6 ? h2 : h3).push_back(b);
    }

    StupidHashMap c3(h3);

    // Build additional elements of c2 from c1
    std::vector<uint64_t> next;
    std::for_each(h1.begin(), h1.end(), [&] (uint64_t pos) {
        list_successors(next, pos, 1);
        for (auto succ : next) {
            if (std::find(h2.begin(), h2.end(), succ) == h2.end()) {
                h2.push_back(succ);
            }
        }
    });

    std::unordered_map<uint32_t /* tile sum */, size_t /* micros */> compute_time;
    std::unordered_map<uint32_t, size_t> count;

    auto print_stats = [&] () {
        std::cout << "Tile sum " << (h1_tile_sum + 2) << ": " << count[h1_tile_sum + 2] << '\n';
    };

    while (true) {
        h1_tile_sum += 2;

        auto start = std::chrono::steady_clock::now();

        // Build c3 from c1, c2
#pragma omp parallel for
        for (size_t i = 0; i < h1.size(); ++i) {
            list_successors(next_tl, h1[i], 2);
            std::sort(next_tl.begin(), next_tl.end());
            next_tl.erase(std::unique(next_tl.begin(), next_tl.end()), next_tl.end());
            for (auto succ : next_tl) {
                c3.insert(succ);
            }
        }

#pragma omp parallel for
        for (size_t i = 0; i < h2.size(); ++i) {
            list_successors(next_tl, h2[i], 1);
            std::sort(next_tl.begin(), next_tl.end());
            next_tl.erase(std::unique(next_tl.begin(), next_tl.end()), next_tl.end());
            for (auto succ : next_tl) {
                c3.insert(succ);
            }
        }

        // c1 = c2, c2 = c3, allocate new c3
        std::swap(h1, h2);
        timed_run("parallel copy",
        [&] { c3.parallel_copy_into(h2); });

        census(h2, h1_tile_sum + 2);

        auto next = std::max({ std::min((2 * h2.size()), 69793218560UL), (uint64_t)(1.2 * h2.size()), 10000000UL });
        std::cout << "Allocating " << next << " for tile sum " << (h1_tile_sum + 6) << '\n';

        if (c3.capacity() < next || next < 100000) {
            c3.~StupidHashMap();
            new (&c3) StupidHashMap(next);
        }

        count[h1_tile_sum + 2] = h2.size();
        auto end = std::chrono::steady_clock::now();
        compute_time[h1_tile_sum + 2] = (size_t)((end - start).count() / 1000);

        print_stats();
        std::cout << "Generation rate: " << (count[h1_tile_sum + 2] / (double)compute_time[h1_tile_sum + 2]) << "M positions/sec" << '\n';
    }
}
