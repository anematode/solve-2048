#include "MoveLUT.h"

#include <stdlib.h>
static uint16_t* move_right_lut16 = nullptr;

__attribute__((constructor))
int generate_move_right_luts() {
    static bool called = false;
    if (called) return 0;
    called = true;

    const int SZ = 1 << 16;

    move_right_lut16 = (uint16_t*)malloc(SZ * sizeof(uint16_t));

    for (uint32_t a = 0; a < (1 << 16); ++a) {
        uint16_t v16 = 0;

        uint8_t tt[4] = { (uint8_t)(a & 0xf), (uint8_t)((a & 0xf0) >> 4), (uint8_t)((a & 0xf00) >> 8), (uint8_t)((a & 0xf000) >> 12)  };

        auto collapse_right = [&] () -> void {
            for (int i = 2; i >= 0; --i) {
                if (tt[i+1] == 0) {
                    tt[i+1] = tt[i];
                    tt[i] = 0;
                }
            }
        };

        collapse_right();
        collapse_right();
        collapse_right();
        for (int i = 2; i >= 0; --i) {
            if (tt[i] == tt[i+1] && tt[i]) {
                tt[i+1] = 1 + tt[i];
                tt[i] = 0;
            }
        }
        collapse_right();
        collapse_right();

        v16 = tt[0] + (tt[1] << 4) + (tt[2] << 8) + (tt[3] << 12);

        move_right_lut16[a] = v16;
    }

    return 0;
}

uint64_t move_right(uint64_t tiles) {
    uint16_t msk = -1;

    tiles = (uint64_t)move_right_lut16[tiles & msk] |
        ((uint64_t)move_right_lut16[(tiles >> 16) & msk] << 16) |
        ((uint64_t)move_right_lut16[(tiles >> 32) & msk] << 32) |
        ((uint64_t)move_right_lut16[(tiles >> 48) & msk] << 48);

    return tiles;
}

uint16_t move_row_right(uint16_t a) {
    return move_right_lut16[a];
}
