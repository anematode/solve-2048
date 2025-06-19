#include "Position.h"
#include <assert.h>
#include "MoveLUT.h"

void split_nibble_shuffle(__m256i shuf, __m256i* hi, __m256i* lo) {
	const __m256i lo_nibble_msk = _mm256_set1_epi8(0xf);
	__m256i indices_lo = _mm256_and_si256(lo_nibble_msk, shuf);
	__m256i indices_hi = _mm256_andnot_si256(lo_nibble_msk, shuf);
	*lo = _mm256_slli_epi32(indices_lo, 2);
	*hi = _mm256_srli_epi32(indices_hi, 2);
}

__m256i shuffle_nibbles(__m256i data, __m256i idx) {
	__m256i lo_nibble_msk = _mm256_set1_epi8(0x0f);

	__m256i shuf_lo, shuf_hi;
	split_nibble_shuffle(idx, &shuf_hi, &shuf_lo);

	__m256i shuffled_lo = _mm256_multishift_epi64_epi8(shuf_lo, data);
	__m256i shuffled_hi = _mm256_multishift_epi64_epi8(shuf_hi, data);

	shuffled_hi = _mm256_slli_epi32(shuffled_hi, 4);
	return _mm256_ternarylogic_epi32(lo_nibble_msk, shuffled_lo, shuffled_hi, 202);
}

__attribute__((always_inline))
void split_nibble_shuffle(__m512i shuf, __m512i* hi, __m512i* lo) {
	const __m512i lo_nibble_msk = _mm512_set1_epi8(0xf);
	__m512i indices_lo = _mm512_and_si512(lo_nibble_msk, shuf);
	__m512i indices_hi = _mm512_andnot_si512(lo_nibble_msk, shuf);
	*lo = _mm512_slli_epi32(indices_lo, 2);
	*hi = _mm512_srli_epi32(indices_hi, 2);
}

__attribute__((always_inline))
__m512i shuffle_nibbles(__m512i data, __m512i idx) {
	__m512i lo_nibble_msk = _mm512_set1_epi8(0x0f);

	__m512i shuf_lo, shuf_hi;
	split_nibble_shuffle(idx, &shuf_hi, &shuf_lo);

	__m512i shuffled_lo = _mm512_multishift_epi64_epi8(shuf_lo, data);
	__m512i shuffled_hi = _mm512_multishift_epi64_epi8(shuf_hi, data);

	shuffled_hi = _mm512_slli_epi32(shuffled_hi, 4);
	return _mm512_ternarylogic_epi32(lo_nibble_msk, shuffled_lo, shuffled_hi, 202);
}

__attribute__((always_inline))
__m512i shuffle_nibbles_same(__m512i data, uint64_t idx) {
	return shuffle_nibbles(data, _mm512_set1_epi64(idx));
}

__attribute__((always_inline))
void canonicalize_positions(uint64_t positions[8]) {
	// Compute the lexicographic minimum of every rotation/reflection
	__m512i p = _mm512_loadu_si512(&positions[0]);

	__m512i a1 = _mm512_min_epu64(
		p,
		shuffle_nibbles_same(p, constants::rotate_90)
	);
	__m512i a2 = _mm512_min_epu64(
		shuffle_nibbles_same(p, constants::rotate_180),
		shuffle_nibbles_same(p, constants::rotate_270)
	);
	__m512i a3 = _mm512_min_epu64(
		shuffle_nibbles_same(p, constants::reflect_h),
		shuffle_nibbles_same(p, constants::reflect_v)
	);
	__m512i a4 = _mm512_min_epu64(
		shuffle_nibbles_same(p, constants::reflect_tl),
		shuffle_nibbles_same(p, constants::reflect_tr)
	);

	__m512i b1 = _mm512_min_epu64(a1, a2);
	__m512i b2 = _mm512_min_epu64(a3, a4);

	__m512i c = _mm512_min_epu64(b1, b2);
	_mm512_storeu_si512(&positions[0], c);
}

uint64_t canonicalize_position(uint64_t position) {
	__m512i shuffled = shuffle_nibbles(
		_mm512_set1_epi64(position),
		_mm512_set_epi64(constants::identity, constants::rotate_90,
			constants::rotate_180, constants::rotate_270, constants::reflect_h, constants::reflect_v,
			constants::reflect_tl, constants::reflect_tr)
	);
	return _mm512_reduce_min_epu64(shuffled);
}

uint64_t set_tile(uint64_t tiles, uint8_t tile, int idx) {
	assert(idx >= 0 && idx < 16);

	uint64_t msk = 0xfULL << (4 * idx);
	return (tiles & ~msk) | ((uint64_t)(tile & 0xf) << (4 * idx));
}

uint8_t get_tile(uint64_t tiles, int idx) {
	assert(idx >= 0 && idx < 16);

	idx *= 4;
	return (tiles & (0xfULL << idx)) >> idx;
}

uint32_t repr_to_tile(uint8_t repr) {
	return (repr == 0) ? 0 : (1ULL  << repr);
}

std::string position_to_string(uint64_t tiles) {
	char out[400];
	char* end = out;

	for (int i = 0; i < 16; ++i) {
		end += sprintf(end, "%d", repr_to_tile(get_tile(tiles, i)));
		*end++ = (i % 4 == 3) ? '\n' : '\t';
	}

	*end++ = '\0';
	return { out };
}

uint32_t tile_sum(uint64_t tiles) {
	uint32_t s = 0;
	for (int i = 0; i < 16; ++i) {
		s += repr_to_tile(get_tile(tiles, i));
	}
	return s;
}

std::vector<uint64_t> starting_positions() {
	std::unordered_set<uint64_t> result;
	for (int i = 0; i < 16; ++i) {
		for (int j = i + 1; j < 16; ++j) {
			for (int tile1 = 1; tile1 <= 2; ++tile1) {
				for (int tile2 = 1; tile2 <= 2; ++tile2) {
					uint64_t a = 0;

					a = set_tile(a, tile1, i);
					a = set_tile(a, tile2, j);

					result.insert(canonicalize_position(a) );
				}
			}
		}
	}
	std::vector<uint64_t> v;
	std::copy(result.begin(), result.end(), std::back_inserter(v));
	return v;
}

void get_rotations(uint64_t tiles, uint64_t arr[4]) {
    __m256i data = _mm256_set1_epi64x((int64_t) tiles);
	__m256i rotations = _mm256_set_epi64x(constants::rotate_90, constants::rotate_180, constants::rotate_270, constants::identity);
	__m256i goose = shuffle_nibbles(data, rotations);
	_mm256_storeu_si256((__m256i*)arr, goose);
}

uint8_t max_tile(uint64_t tile) {
	uint8_t max = 0;
	for (int i = 0; i < 16; ++i) {
		max = std::max(max, (uint8_t)((tile >> (4 * i)) & 0xf));
	}
	return max;
}

void list_successors(std::vector<uint64_t> &vec, uint64_t tiles, int tile) {
	// Try all four rotations, rotate right. Then canonicalize all. Don't attempt to deduplicate
	// as that will be done implicitly by the set insertion (which should be fairly fast due to caching)

	vec.resize(0);

	uint64_t rotations[4];
	get_rotations(tiles, rotations);

	uint64_t moved_right[4];
	for (int i = 0; i < 4; ++i) {
		moved_right[i] = move_right(rotations[i]);
	}

	for (int rot_i = 0; rot_i < 4; ++rot_i) {
		auto start = rotations[rot_i];
		auto moved = moved_right[rot_i];
		bool necessarily_valid = start != moved;
		for (int i = 0; i < 4; ++i) {
			bool goose = true;
			for (int j = 0; j < 4; ++j) {
				bool empty = !(start & (0xfULL  << (16 * i + 4 * j)));
				if (empty && goose) {
					uint16_t row = (uint16_t)(start >> (16 * i)) | (tile << (4 * j));
					uint16_t replace = move_right(row);
					bool valid = replace != row || necessarily_valid;
					if (valid) {
						vec.push_back(moved & ~(0xffffULL  << (16 * i)) | ((uint64_t)replace << (16*i)));
					}
					goose = false;
				}
				goose = goose || !empty;
			}
		}
	}

	const size_t n = vec.size();
	const uint64_t* ptr = vec.data();

	for (size_t i = 0; i < n; i += 8) {
		size_t remaining = n - i;
		__mmask8 mask = __builtin_expect_with_probability(remaining >= 8, 1, 0.5)  ? 0xFF : (1 << remaining) - 1;
		__m512i v = _mm512_maskz_loadu_epi64(mask, ptr + i);
		uint64_t p[8];
		_mm512_storeu_si512(p, v);
		canonicalize_positions(p);
		_mm512_mask_storeu_epi64((void*)(ptr + i), mask, _mm512_loadu_si512(p));
	}
}

