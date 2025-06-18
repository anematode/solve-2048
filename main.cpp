#include <iostream>
#include <chrono>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/parallel_for.h>
#include <omp.h>

#include "StupidHashMap.h"
#include "Position.h"

thread_local std::vector<uint64_t> next_tl;
thread_local std::vector<uint64_t> next_tl2;

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
            for (auto succ : next_tl) {
                c3.insert(succ);
            }
        }

        auto end = std::chrono::steady_clock::now();
        compute_time[current_tile_sum + 4] = (size_t)((end - start).count() / 1000);

        // c1 = c2, c2 = c3, allocate new c3
        std::swap(four, six);
        c3.parallel_copy_into(six);

        count[current_tile_sum + 4] = six.size();
        print_stats();
        std::cout << "Generation rate: " << (count[current_tile_sum + 4] / (double)compute_time[current_tile_sum + 4]) << "M positions/sec" << '\n';

        auto next = (uint64_t)(1.5 * six.size());
        std::cout << "Allocating " << next << " for tile sum " << (current_tile_sum + 6) << '\n';
        c3.~StupidHashMap();
        new (&c3) StupidHashMap(next);
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
