//
// Created by root on 6/17/25.
//

#ifndef POSITION_H
#define POSITION_H

#include <immintrin.h>
#include <stdint.h>
#include <string>
#include <unordered_set>
#include <vector>

namespace constants {
    constexpr inline uint64_t identity = 0xfedcba9876543210,
        rotate_90 = 0xc840d951ea62fb73,
        rotate_180 = 0x0123456789abcdef,
        rotate_270 = 0x37bf26ae159d048c,
        reflect_h = 0xcdef89ab45670123,
        reflect_v = 0x32107654ba98fedc,
        reflect_tl = 0xfb73ea62d951c840,
        reflect_tr = 0x048c159d26ae37bf;
}

// High 6 bits: bitset of permutations. Low six bits: canonical position, shifted down by 4 (reconstruct
// final nibble using tile sum)
using Packed6Perm = uint64_t;

void canonicalize_positions(uint64_t positions[8]);
uint64_t set_tile(uint64_t tiles, uint8_t tile, int idx);
uint8_t get_tile(uint64_t tiles, int idx);
uint32_t repr_to_tile(uint8_t repr);
std::string position_to_string(uint64_t tiles);
uint32_t tile_sum(uint64_t tiles);
std::vector<uint64_t> starting_positions();
void get_rotations(uint64_t tiles, uint64_t arr[4]);
int count_symmetries(uint64_t position);
// Get the (representation of) the maximum tile in the position.
uint8_t max_tile(uint64_t tile);
// Compress position containing at most the tile 1024 (2^11) using the encoding t1*
uint64_t compress_small_position(uint64_t pos, int tile_sum);

void list_successors(std::vector<uint64_t>& vec, uint64_t tiles, int tile);

#endif //POSITION_H
