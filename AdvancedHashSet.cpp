#include "AdvancedHashSet.h"

#include <iostream>
#include <mutex>
#include <algorithm>
#include <string.h>

void initialize_lut() {
    for (int i = 0; i < 1 << 12; ++i) {
        Position position(i);
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

        sort_lower_3_lut[i] = (matches << 12) | (sorted.nth_row(0) & 0xfff);
    }
}

bool AdvancedHashSet::insert(Position position) {
    assert(position.is_canonical());
    assert(position.tile_sum() == tile_sum);

    auto [ index, sorted ] = sort_lower_3(position);

    auto hash = sorted.hash();
try_again:
    size_t hash_index = hash % capacity;

    uint64_t the_bit = 1ULL << (POSITION_BITS + index);
    while (true) {
        if (data[hash_index] == 0) {
            uint64_t expected = 0;
            bool success = __atomic_compare_exchange_n(&data[hash_index],
                                                       &expected, the_bit | (sorted.bits >> 4), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
            if (!success)
                goto try_again;
            return true;
        }
        if ((data[hash_index] & ((1ULL << POSITION_BITS) - 1)) == (sorted.bits >> 4)) {
            if (data[hash_index] & the_bit) {
                return false;  // already in there
            }
            uint64_t expected = data[hash_index];
            bool success = __atomic_compare_exchange_n(&data[hash_index],
                                                       &expected, the_bit | expected, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
            if (!success)
                goto try_again;
            return true;
        }
        hash_index = (hash_index + 1) % capacity;
    }
}

void parallel_memcpy(uint64_t *p, uint64_t *p1, uint64_t *p2) {
    if (p + (p2 - p1) <= p2 - 8 || (p2 - p1 < 1000000)) {
        // do ST
        memmove(p, p1, (p2 - p1) * sizeof(uint64_t));
        return;
    }

    // First handle non-aligned beginning
    while (p1 < p2 && ((uintptr_t)p % 64 != 0)) {
        *p++ = *p1++;
    }

    // Then copy with cache line granularity so that we can use the full memory bandwidth
#pragma omp parallel for
    for (size_t i = 0; i < p2 - p1; i += 8) {
        _mm512_store_si512(&p[i], _mm512_loadu_si512(&p1[i]));
    }
}

void AdvancedHashSet::gorge() {
    size_t old_capacity = capacity;
    // Remove all zero entries, place at the beginning, and truncate capacity
    if (capacity < 10000) {
        size_t j = 0;
        for (size_t i = 0; i < capacity; ++i) {
            if (data[i] != 0) {
                data[j++] = data[i];
            }
        }
        capacity = j;
    } else {
        int num_threads = omp_get_max_threads();
        size_t chunk_size = capacity / num_threads;
        chunk_size -= chunk_size % 8;  // for alignment purposes

        std::vector<uint64_t> ranges;
        for (size_t i = 0; i < num_threads; ++i) {
            ranges.push_back(i * chunk_size);
        }
        ranges.push_back(capacity);
        std::vector<uint64_t> count(num_threads);

        std::mutex mtx;
        std::vector<std::pair<int, int>> live_ranges;

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t j = ranges[tid];
            for (size_t i = ranges[tid]; i < ranges[tid + 1]; ++i) {
                if (data[i] != 0) {
                    data[j++] = data[i];
                }
            }
            {
                std::lock_guard lg(mtx);
                live_ranges.push_back({ ranges[tid], j });
            }
        }

        std::sort(live_ranges.begin(), live_ranges.end(), [&] (auto a, auto b) {
            return a.first < b.first;
        });

        size_t offset = 0;
        for (auto [low, high] : live_ranges) {
            parallel_memcpy(&data[offset], &data[low], &data[high]);
            offset += high - low;
        }
        capacity = offset;
    }
    divider = libdivide::divider(capacity);
    data = (uint64_t*)mremap(data, old_capacity * sizeof(uint64_t), capacity * sizeof(uint64_t), MREMAP_MAYMOVE);
    if (data == MAP_FAILED || !data) {
        throw std::runtime_error("Failed to remap");
    }
}
