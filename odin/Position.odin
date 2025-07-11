package solve_2048
import "core:fmt"
import "core:simd"
import "core:strconv"
import "core:strings"

foreign import modern "Position.asm"
foreign modern {
	__test_add    :: proc "c" (a: #simd [8]f32, p: #simd [8]f32) -> #simd [8]f32 ---
	__tile_values :: proc "c" (tiles: u64) -> #simd [16] u16 ---
}

// Constants
identity   :: 0xfedcba9876543210;
rotate_90  :: 0xc840d951ea62fb73;
rotate_180 :: 0x0123456789abcdef;
rotate_270 :: 0x37bf26ae159d048c;
reflect_h  :: 0xcdef89ab45670123;
reflect_v  :: 0x32107654ba98fedc;
reflect_tl :: 0xfb73ea62d951c840;
reflect_tr :: 0x048c159d26ae37bf;

// Each consecutive 4 bits is a tile. The largest tile we can store in this fashion is 32768.
Position :: distinct u64;

RawPosition :: #simd [16] u16;

set_tile :: proc(tiles: Position, tile: u8, index: int) -> Position
{
	assert(index >= 0 && index < 16);

	mask: Position = 0xF << u32(4 * index);
	return (tiles & ~mask) | (Position(tile & 0xF) << u32(4 * index));
}

get_tile :: proc(tiles: Position, index: int) -> u8
{
	assert(index >= 0 && index < 16);
	return cast(u8) (tiles >> u32(index*4)) & 0xF;
}

tile_value :: proc(repr: u8) -> u32
{
	return (repr == 0) ? 0 : (1 << repr);
}

to_string :: proc(tiles: Position) -> string
{
	builder : strings.Builder;
	strings.builder_init(&builder, context.temp_allocator);
	for i in 0..<16 {
		strings.write_u64(&builder, auto_cast tile_value(get_tile(tiles, i)), 10);
		strings.write_rune(&builder, '\n' if i % 4 == 3 else '\t');
	}
	return strings.to_string(builder);
}

tile_sum :: proc(tiles: Position) -> u32
{
	when false {
		sum: u32;
		for i in 0..<16 {
			sum += tile_value(get_tile(tiles, i));
		}
		return sum;
	}
	else {
		return simd.reduce_add_pairs(__tile_values(u64(tiles)));
	}
}

/*

    explicit Position(uint64_t bits);

    static uint16_t move_row_right(uint16_t row);

	// Move an array of rows right (rows is an inout parameter)
    static void move_rows_right(uint16_t *rows, int count);

    // Move the position right. Returns the same position if the move is illegal.
    Position move_right() const;
    uint64_t hash() const;

    auto operator<=>(const Position& right) const = default;

    // Permute this position according to one of the shuffle constants
    Position permute(uint64_t shuffle) const;

    // Find the lexicographically smallest position of all rotations and reflections. The resulting
    // position is guaranteed to be smaller than 3*2^60 because all reachable positions have a corner
    // that is a 0, 2, or 4.
    Position canonical_form() const;

    // Return whether the position is canonical. Useful for debugging.
    bool is_canonical() const {
        return canonical_form() == *this;
    }
    // Return the tile sum.
    uint32_t tile_sum() const;

    // Get the nth tile, 0 <= index <= 15. 0 -> empty, 1 -> 2, etc.
    uint8_t operator[] (int index) const {
        return (bits >> (4 * index)) & 0xF;
    }
    // Get the nth row, 0 <= row <= 3.
    uint16_t nth_row(int index) const {
        return bits >> (16 * index);
    }
    // Get the maximum tile.
    uint8_t max_tile() const;

    // Set the tile at the given index to the given value.
    Position set_tile(int index, uint8_t tile) const;

    // List all rotations/reflections of this position. May include duplicates.
    void list_equivalent(Position out[8]) const;
    // Count the number of distinct rotations/reflections of this position (8 for most position, fewer for symmetric positions)
    int count_equivalent() const;
    // Convert the position to a readable string.
    std::string to_string() const;

    // Returns true iff the position has no legal moves.
    bool is_lost() const;
    // List the possible successors of this position, formed by placing the given tile in an empty square, then making
    // a legal move.
    void list_successors(std::vector<uint64_t>& out, int tile /* 1 or 2 */);

    static std::vector<Position> starting_positions();
};

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

*/