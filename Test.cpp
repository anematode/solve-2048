#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "AdvancedHashSet.h"
#include "doctest.h"
#include "Position.h"

uint64_t positions[] = {
    0x002,
    0x0002015,
    0x000219a,
    0xaaaa52521100011
};

TEST_CASE("AdvancedHashSet") {
    AdvancedHashSet set({ .tile_sum = 98, .initial_size = (uint64_t)100000000, .load_factor = 1.0 });

    // Open the file in binary mode and position the file pointer at the end.
    std::ifstream file("/home/mitchell/compression/350", std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    // Get the size of the file.
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint64_t> data(size / sizeof(uint64_t));

    // Read the entire file into the vector.
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Error reading file");
    }

    #pragma omp parallel for
    for (auto goose : data) {
        set.insert(Position { goose });
    }

    size_t count = set.parallel_count();
    set.gorge();

    std::vector<uint64_t> v;
    set.for_each_position_parallel([&] (Position f) {
        v.push_back(f.bits);
    }, /*threads=*/1);
    std::sort(v.begin(), v.end());

    for (int i = 0; i < v.size() ; ++i) {
        CHECK(v[i] == data[i]);
    }
}