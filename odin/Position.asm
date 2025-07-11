bits 64

.LCPI0_1:
        .short  0
        .short  2
        .short  4
        .short  8
        .short  16
        .short  32
        .short  64
        .short  128
        .short  256
        .short  512
        .short  1024
        .short  2048
        .short  4096
        .short  8192
        .short  16384
        .short  32768
.LCPI0_2:
        .byte   60
        .byte   56
        .byte   52
        .byte   48
        .byte   44
        .byte   40
        .byte   36
        .byte   32
        .byte   28
        .byte   24
        .byte   20
        .byte   16
        .byte   12
        .byte   8
        .byte   4
        .byte   0

global __test_add
global __tile_values

section .text
__test_add:
    vaddps  ymm0, ymm0, ymm1
	ret

__tile_values:
        vpbroadcastq    ymm0, rdi
        vpmovsxbw       ymm1, [rip + .LCPI0_2]
        vpmultishiftqb  ymm0, ymm1, ymm0
        vpermw          ymm0, ymm0, [rip + .LCPI0_1]
        ret