#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Position.h"

uint64_t positions[] = {
    0x002,
    0x0002015,
    0x000219a,
    0xaaaa52521100011
};

TEST_CASE("max_tile") {
    auto reference = [&] (uint64_t p) {
        uint8_t max = 0;
        for (int i = 0; i < 16; ++i) {
            max = std::max(get_tile(p, i), max);
        }
        return max;
    };

    for (auto position : positions) {

    }
}