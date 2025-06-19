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

int main()
{
    uint32_t current_tile_sum = 4;  // 4, 6, 8
    omp_set_num_threads(omp_get_max_threads());

    std::vector<uint64_t> four, six, eight;
    std::vector<uint64_t> all = starting_positions();

    for (auto b : all) {
        auto sum = tile_sum(b);
        (sum == 4 ? four : sum == 6 ? six : eight).push_back(b);
    }

    StupidHashMap c3(eight);

    // Build additional elements of c2 from c1
    std::vector<uint64_t> next;
    std::for_each(four.begin(), four.end(), [&] (uint64_t pos) {
        list_successors(next, pos, 1);
        for (auto succ : next) {
            if (std::find(six.begin(), six.end(), succ) == six.end()) {
                six.push_back(succ);
            }
        }
    });

    std::unordered_map<uint32_t /* tile sum */, size_t /* micros */> compute_time;
    std::unordered_map<uint32_t, size_t> count;

    auto print_stats = [&] () {
        std::cout << "Tile sum " << (current_tile_sum + 4) << ": " << count[current_tile_sum + 4] << '\n';
    };

    while (true) {
        current_tile_sum += 2;

        auto start = std::chrono::steady_clock::now();

        // Build c3 from c1, c2
#pragma omp parallel for
        for (size_t i = 0; i < four.size(); ++i) {
            list_successors(next_tl, four[i], 2);
            std::sort(next_tl.begin(), next_tl.end());
            next_tl.erase(std::unique(next_tl.begin(), next_tl.end()), next_tl.end());
            for (auto succ : next_tl) {
                c3.insert(succ);
            }
        }

#pragma omp parallel for
        for (size_t i = 0; i < six.size(); ++i) {
            list_successors(next_tl, six[i], 1);
            std::sort(next_tl.begin(), next_tl.end());
            next_tl.erase(std::unique(next_tl.begin(), next_tl.end()), next_tl.end());
            for (auto succ : next_tl) {
                c3.insert(succ);
            }
        }

        // c1 = c2, c2 = c3, allocate new c3
        std::swap(four, six);
        timed_run("parallel copy",
        [&] { c3.parallel_copy_into(six); });

        if ((current_tile_sum + 2) % 50 == 0) {
            timed_run("parallel sort", [&] {
                std::sort(std::execution::par_unseq, six.begin(), six.end());
            });
            timed_run("compressing data", [&] {
                compress_data((const char*)six.data(), six.size() * sizeof(six[0]),
        "./lists/" + std::to_string(current_tile_sum + 2) + ".zst");
            });
        }

        auto next = std::max((uint64_t)(1.25 * six.size()), 10000000UL);
        std::cout << "Allocating " << next << " for tile sum " << (current_tile_sum + 6) << '\n';

        if (c3.capacity() < next || next < 100000) {
            c3.~StupidHashMap();
            new (&c3) StupidHashMap(next);
        }

        count[current_tile_sum + 4] = six.size();
        auto end = std::chrono::steady_clock::now();
        compute_time[current_tile_sum + 4] = (size_t)((end - start).count() / 1000);

        print_stats();
        std::cout << "Generation rate: " << (count[current_tile_sum + 4] / (double)compute_time[current_tile_sum + 4]) << "M positions/sec" << '\n';

    }

    /*
    StupidHashMap set(1'000'000'000);
    auto start = std::chrono::steady_clock::now();

    omp_set_num_threads(96);

    // Insert 1 billion numbers
#pragma omp parallel for
    for (int i = 1; i < 1'000'000'000; ++i) {
        auto b = i % 100'000'000;
        if (b) {
            set.insert(b);
        }
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed: " << (double)elapsed.count() / 1e9 << '\n';

    start = std::chrono::steady_clock::now();
    std::cout << "Count: " << set.parallel_count() << '\n';
    elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "Elapsed: " << (double)elapsed.count() / 1e9 << '\n';

    return 0;
    */
}
