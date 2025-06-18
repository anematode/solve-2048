//
// Created by root on 6/17/25.
//

#ifndef STUPIDHASHMAP_H
#define STUPIDHASHMAP_H

#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <atomic>

inline uint64_t get_hash_index(uint64_t a) {
    uint8_t key_bytes[16] = {
        0x42, 0x7a, 0x13, 0x9d, 0xfe, 0x5c, 0x88, 0x21,
        0xde, 0xad, 0xbe, 0xef, 0x77, 0x66, 0x55, 0x44
    };
    __m128i key = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key_bytes));

    // Zero-extend or duplicate the 64-bit input to 128 bits
    alignas(16) uint64_t input_block[2] = {a, a};  // or {input, input} if you prefer
    __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_block));

    // Perform one AESENC round
    __m128i result = _mm_aesenc_si128(_mm_aesenc_si128(_mm_aesenc_si128(data, key), key), key);

    // Extract and return the lower 64 bits of the result
    alignas(16) uint64_t output[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), result);
    return output[0];
}

inline size_t count_nonzero(const uint64_t *begin, const uint64_t *end) {
    uint64_t sum = 0;
    auto C = end - begin;
    if (C%8) {
        throw std::runtime_error("oops");
    }
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < C ; i += 8) {
        __m512i v = _mm512_loadu_epi64(&begin[i]);
        sum += __builtin_popcount(_mm512_test_epi64_mask(v, v));
    }
    return sum;
}

constexpr int MIN_CAP_LG2 = 12; /* 4096 */

struct StupidHashMap {
    uint64_t *data;
    uint32_t cap_lg2;

    std::atomic<int> count;  // lazily updated by threads

    StupidHashMap(uint64_t needed_capacity) : cap_lg2(std::max(64 - __builtin_clzll(needed_capacity - 1), MIN_CAP_LG2)) {
        auto huge = 0; //cap_lg2 > 20 ? (MAP_HUGETLB | (30 << MAP_HUGE_SHIFT)) : 0;
        data = (uint64_t*)mmap(NULL, std::max(capacity() * sizeof(uint64_t), 4096UL), PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | huge, -1, 0);
        if (data == MAP_FAILED) {
            std::cerr << "Failed to allocate " << capacity() << " words (huge TLB working?)\n";
            throw std::bad_alloc();
        }
        madvise(data, capacity(), MADV_WILLNEED);

#pragma omp parallel for
        for (size_t i = 0; i < capacity(); ++i) {
            data[i] = 0;  // force the memory to be allocated
        }
    }

    StupidHashMap(const std::vector<uint64_t>& v) : StupidHashMap(v.size()) {
        for (auto p : v) {
            insert(p);
        }
    }

    StupidHashMap& operator=(StupidHashMap&& rhs) noexcept {
        if (data) {
            munmap(data, capacity() * sizeof(uint64_t));
        }

        data = rhs.data;
        rhs.data = nullptr;
        cap_lg2 = rhs.cap_lg2;
        count = rhs.count.load();
        return *this;
    }

    template <typename F>
    void serial_for_each(F&& f) {
        for (size_t i = 0; i < capacity(); ++i) {
            if (data[i]) {
                f(data[i]);
            }
        }
    }

    template <typename F>
    void parallel_for_each(F&& f) {
        auto C = capacity();
#pragma omp parallel for
        for (size_t i = 0; i < C; ++i) {
            if (data[i]) {
                f(data[i]);
            }
        }
    }

    size_t capacity() const {
        return 1ULL << cap_lg2;
    }

    bool contains(uint64_t entry) const {
        assert(entry && "SHM can't store a 0");
        uint64_t index = get_hash_index(entry) & (capacity() - 1);
        while (data[index] != 0) {
            if (data[index] == entry) {
                return true;
            }
            index = (index + 1) % capacity();
        }
        return false;
    }

    // Returns true iff something new was inserted
    bool insert(uint64_t entry) {
        assert(entry && "SHM can't store a 0");
        try_again:
        uint64_t index = get_hash_index(entry) % capacity();
        while (data[index] != 0) {
            if (data[index] == entry) {
                return false;
            }
            index = (index + 1) % capacity();
        }
        uint64_t expected = 0;
        bool success = __atomic_compare_exchange_n(&data[index], &expected, entry, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        if (!success)
            goto try_again;
        return true;
    }

    ~StupidHashMap() {
        if (data) {
            munmap(data, capacity() * sizeof(uint64_t));
        }
        data = nullptr;
    }

    // Get the number of set elements
    size_t parallel_count() const {
        return count_nonzero(data, data + capacity());
    }

    void parallel_copy_into(std::vector<uint64_t>& six) {
        if (capacity() < 1000000) {
            six.resize(0);
            serial_for_each([&] (uint64_t a) {
                six.push_back(a);
            });
            return;
        }
        int num_threads = omp_get_max_threads();
        size_t chunk_size = capacity() / num_threads;
        chunk_size -= chunk_size % 8;  // for alignment purposes

        std::vector<uint64_t> ranges;
        for (size_t i = 0; i < num_threads; ++i) {
            ranges.push_back(i * chunk_size);
        }
        ranges.push_back(capacity());
        std::vector<uint64_t> count(num_threads);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            count[tid] = count_nonzero(data + ranges[tid], data + ranges[tid + 1]);
        }


        // Now prefix sum so that each thread knows where to start writing
        uint64_t S = 0;
        for (int i = 0; i < count.size(); ++i) {
            auto tmp = count[i];
            count[i] = S;
            S += tmp;
        }

        six.resize(0); // so that old elements don't need to be preserved
        six.resize(S);  // TODO skip zero-fill

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t write_i = count[tid];
            for (size_t offs = ranges[tid]; offs < ranges[tid + 1]; ++offs) {
                if (data[offs]) {
                    six[write_i++] = data[offs];
                }
            }
            std::sort(six.begin() + count[tid], six.begin() + write_i);
        }
    }
};

#endif //STUPIDHASHMAP_H
