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
#include "AdvancedHashSet.h"

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
    uint32_t h1_tile_sum = 4;  // 4, 6, 8
    omp_set_num_threads(omp_get_max_threads());

    AdvancedHashSet::Config config = {
        .tile_sum = 4,
        .initial_size = 100,
        .load_factor = 1.0
    };

    AdvancedHashSet h1(config);
    config.tile_sum = 6;
    AdvancedHashSet h2(config);
    config.tile_sum = 8;
    AdvancedHashSet h3(config);

    std::vector<uint64_t> all = starting_positions();

    for (auto b : all) {
        auto sum = tile_sum(b);
        (sum == 4 ? h1 : sum == 6 ? h2 : h3).insert(Position { b });
    }

    // Build additional elements of c2 from c1
    std::vector<uint64_t> next;
    h1.for_each_position_parallel([&] (Position pos) {
        list_successors(next, pos.bits, 1);
        for (auto succ : next) {
            h2.insert(Position { succ });
        }
    }, 1);

    h1.gorge();
    h2.gorge();

    std::unordered_map<uint32_t /* tile sum */, size_t /* micros */> compute_time;
    std::unordered_map<uint32_t, size_t> count;

    auto print_stats = [&] () {
        std::cout << "Tile sum " << (h1_tile_sum + 2) << ": " << count[h1_tile_sum + 2] << '\n';
    };

    while (true) {
        h1_tile_sum += 2;

        auto start = std::chrono::steady_clock::now();

        // Build c3 from c1, c2
        timed_run("insert h1", [&] {
            h1.for_each_position_parallel([&] (Position p) {
                list_successors(next_tl, p.bits, 2);
                for (auto succ : next_tl) {
                    h3.insert(Position { succ } );
                }
            });
        });
        std::vector<std::vector<uint64_t>> per_thread_census;
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            per_thread_census.push_back(std::vector<uint64_t> (12));
        }
        timed_run("insert h2", [&] {
            h2.for_each_position_parallel([&] (Position p) {
                per_thread_census[omp_get_thread_num()][p.max_tile()]++;
                list_successors(next_tl, p.bits, 1);
                for (auto succ : next_tl) {
                    h3.insert(Position { succ } );
                }
            });
        });

        std::cout << "Max tile census: ";
        for (int tile_i = 1; tile_i < 12; ++tile_i) {
            uint64_t total = 0;
            for (int j = 0; j < per_thread_census.size(); ++j) {
                total += per_thread_census[j][tile_i];
            }
            std::cout << (1 << tile_i) << ": " << total << "; ";
        }
        std::cout << std::endl;

        // c1 = c2, c2 = c3, allocate new c3
        h1 = std::move(h2);
        timed_run("h3 gorge", [&] {
            h3.gorge();
        });
        h2 = std::move(h3);

        // census(h2, h1_tile_sum + 2);
        auto next = (uint64_t)(h3.capacity * 1.2);

        std::cout << "Allocating " << next << " for tile sum " << (h1_tile_sum + 4) << '\n';

        config.tile_sum = h1_tile_sum + 4;
        config.initial_size = std::max(next, 10000000UL);
        new (&h3) AdvancedHashSet(config);

        count[h2.tile_sum] = h2.parallel_count();
        auto end = std::chrono::steady_clock::now();
        compute_time[h2.tile_sum] = (size_t)((end - start).count() / 1000);

        print_stats();
        std::cout << "Generation rate: " << (count[h2.tile_sum] / (double)compute_time[h2.tile_sum]) << "M positions/sec" << '\n';
    }
}
