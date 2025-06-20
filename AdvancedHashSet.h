//
// Created by root on 6/20/25.
//

#ifndef ADVANCEDHASHSET_H
#define ADVANCEDHASHSET_H

#include <cstdint>
#include <atomic>
#include <pthread.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <omp.h>
#include <functional>
#include <sys/mman.h>

#include "libdivide.h"
#include "Position.h"

static __m256i goose = _mm256_setr_epi8(
                2, 1, 0, -1,
                2, 0, 1, -1,   // 2,0,1
                0, 2, 1, -1, // 0,2,1
                0, 1, 2, -1, // 0,1,2
                1, 2, 0, -1,   // 1,2,0
                1, 0, 2, -1,  // 1,0,2
                -1, -1, -1, -1,  // ignore
                -1, -1, -1, -1  // ignore
                );

static uint16_t sort_lower_3_lut[1 << 12];

__attribute__((constructor))
void initialize_lut();

inline std::pair<int, Position> sort_lower_3(Position position) {
    uint16_t m = sort_lower_3_lut[position.nth_row(0) & 0xfff];
    constexpr int t1 = 0, t2 = 1, t3 = 2;
    // Sort the bottom three nibbles to determine the storage location.
    uint8_t tile_0 = position[t1], tile_1 = position[t2], tile_2 = position[t3];

    if (tile_1 < tile_2) {
        std::swap(tile_1, tile_2);
    }
    if (tile_0 < tile_1) {
        std::swap(tile_0, tile_1);
    }
    if (tile_1 < tile_2) {
        std::swap(tile_1, tile_2);
    }

    Position sorted = position.set_tile(t1, tile_0).set_tile(t2, tile_1).set_tile(t3, tile_2);

    // Find the number between 0 and 5, inclusive, which corresponds to the lexicographically minimal permutation
    // of the sorted position that yields the original position.

    // splat to bytes
    uint32_t original_three = position[t3] << 16 | position[t2] << 8 | position[t1];
    uint32_t sorted_three = sorted[t3] << 16 | sorted[t2] << 8 | sorted[t1];

    __m256i shuffled = _mm256_shuffle_epi8(
        _mm256_set1_epi32(sorted_three), goose
    );

    // Try all six shuffles and record the first matching index
    int matches = _mm256_cmpeq_epi32_mask(
        shuffled, _mm256_set1_epi32(original_three));
    matches = __builtin_ctz(matches);
    return { matches, sorted };
    //return { m >> 12, Position((position.bits & ~0xfffULL) | (m & 0xfffULL))};
}

// Stores canonical positions with a tile sum fixed at creation. Supports concurrent insertions
// and lookups, but not deletions.
//
// TODO: support resizing
struct AdvancedHashSet {
    int tile_sum;
    uint64_t *data;

    constexpr static int POSITION_BITS = 58;

    // Capacity, in 64-bit words.
    libdivide::divider<uint64_t> divider;
    size_t capacity;

    struct Config {
        int tile_sum;
        size_t initial_size;
        double load_factor;
    };

    AdvancedHashSet(Config config) {
        auto huge = config.initial_size > (1 << 20) ? (MAP_HUGETLB | (30 << MAP_HUGE_SHIFT)) : 0;
        try_again:
        tile_sum = config.tile_sum;
        auto ptr = mmap(NULL, std::max(config.initial_size * sizeof(uint64_t), 4096UL), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | huge, -1, 0);
        if (ptr == MAP_FAILED || !ptr) {
            if (huge) {
                huge = 0;
                goto try_again;
            }
            throw std::runtime_error("map failed");
        }
        data = (uint64_t*)ptr;
        capacity = config.initial_size;
        divider = libdivide::divider(capacity);
    }

    ~AdvancedHashSet() {
        munmap(data, capacity * sizeof(uint64_t));
    }

    bool insert(Position position);

    bool contains(Position position) const {
        auto [ index, sorted ] = sort_lower_3(position);
        size_t hash_index = sorted.hash() / divider;

        uint64_t the_bit = 1ULL << (POSITION_BITS + index);
        while (true) {
            if ((data[hash_index] << 6 >> 6) == (sorted.bits >> 4)) {
                return (data[hash_index] & the_bit);
            }
            hash_index = (hash_index + 1) % capacity;
        }
    }

    void gorge();

    AdvancedHashSet& operator=(AdvancedHashSet&& rhs) noexcept {
        tile_sum = rhs.tile_sum;
        if (data) {
            munmap(data, capacity * sizeof(uint64_t));
        }
        data = rhs.data;
        rhs.data = nullptr;
        capacity = rhs.capacity;
        divider = rhs.divider;
        return *this;
    }


    template <typename F>
    void for_each_position_parallel(F&& f, int threads=omp_get_max_threads() ) const {
#pragma omp parallel for num_threads(threads)
        for (size_t i = 0; i < capacity; ++i) {
            uint64_t d = data[i];
            if (d) {
                uint64_t low_bits = d & ((1ULL << POSITION_BITS) - 1);

                uint32_t recovered_tile = tile_sum - Position(low_bits).tile_sum();
                assert(recovered_tile == 0 || __builtin_popcount(recovered_tile) == 1);
                uint64_t recovered_position = (low_bits << 4) | (recovered_tile == 0 ? 0 : __builtin_ctz(recovered_tile));
                assert(Position(recovered_position).tile_sum() == tile_sum);

                int perms[6] = { 0x012, 0x102, 0x120, 0x210, 0x021, 0x201 };
                uint64_t perm_list[6];
                size_t perm_count = 0;
#pragma GCC unroll 6
                for (int perm_i = 0; perm_i < 6; ++perm_i) {
                    if (((d >> POSITION_BITS) & (1 << perm_i))) {
                        int perm = perms[perm_i];
                        Position permed { recovered_position };
                        for (int j = 0; j < 3; ++j) {
                            permed = permed.set_tile(j, Position(recovered_position)[(perm >> (4*j)) & 0xf]);
                        }
                        perm_list[perm_count++] = permed.bits;
                    }
                }
                for (int i = 0; i < perm_count; ++i) {
                    f(Position { perm_list[i] });
                }
            }
        }
    }

    size_t parallel_count() const {
        size_t count = 0;
#pragma omp parallel for reduction(+:count)
        for (size_t i = 0; i < capacity; ++i) {
            count += __builtin_popcount(data[i] >> POSITION_BITS);
        }
        return count;
    }
};

#endif //ADVANCEDHASHSET_H
