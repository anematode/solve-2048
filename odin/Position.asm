bits 64
section .text


global __test_add
__test_add:
   	vaddps		ymm0, ymm1, ymm1
	ret


global __tile_sum
__tile_sum:
        vmovdqa 		ymm0, [rel .shifts]
        vpbroadcastq    ymm1, rdi
        vpmultishiftqb  ymm0, ymm0, ymm1
        vpermw  		ymm0, ymm0, [rel .powers_of_two]
        vmovdqa 		xmm1, xmm0
        vextracti32x4   xmm0, ymm0, 0x1
        vpaddw  		xmm0, xmm0, xmm1
        vpunpckhqdq     xmm1, xmm0, xmm0
        vpaddw  		xmm0, xmm0, xmm1
        vpshuflw        xmm1, xmm0, 238
        vpaddw  		xmm0, xmm1, xmm0
        vpshuflw        xmm1, xmm0, 229
        vpaddw  		xmm0, xmm1, xmm0
        vpextrw			eax, xmm0, 0
        cwde
        vzeroupper
        ret
align 32
.shifts:
    db	60, 0, 56, 0, 52, 0, 48, 0, 44, 0, 40, 0, 36, 0, 32, 0, 28, 0, 24, 0, 20, 0, 16, 0, 12, 0, 8, 0, 4, 0, 0, 0
align 32
.powers_of_two:
	dw	0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, -32768


global __canonical
__canonical:
        vmovdqa64       zmm2, [rel .LC2]
        vpbroadcastq    zmm1, rdi
        vmovdqa32       zmm0, [rel .LC4]
        vpmultishiftqb  zmm2, zmm2, zmm1
        vpmultishiftqb  zmm0, zmm0, zmm1
        vmovdqa32       zmm1, [rel .LC0]
        vpslld  		zmm0, zmm0, 4
        vpternlogd      zmm1, zmm2, zmm0, 202
        vshufi64x2      zmm0, zmm1, zmm1, 78
        vpminuq 		zmm0, zmm0, zmm1
        vshufi64x2      zmm1, zmm0, zmm0, 177
        vpminuq 		zmm0, zmm0, zmm1
        vpermq  		zmm1, zmm0, 177
        vpminuq 		zmm0, zmm0, zmm1
        vmovq   		rax, xmm0
        vzeroupper
        ret
align 64
.LC0:
	dd 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135, 252645135
align 64
.LC2:
	db 60, 28, 56, 24, 52, 20, 48, 16,  0, 32,  4, 36,  8, 40, 12, 44, 48, 56, 32, 40, 16, 24,  0,  8, 12,  4, 28, 20, 44, 36, 60, 52, 48, 16, 52, 20, 56, 24, 60, 28, 60, 52, 44, 36, 28, 20, 12,  4, 12, 44,  8, 40,  4, 36,  0, 32,  0,  8, 16, 24, 32, 40, 48, 56
align 64
.LC4:
	dd 136842284,   2098212, 873738256, 1008482328, 740572212, 201595924, 270008328,  808984616,  69468192, 204212264, 539504696,     528408, 941112348, 806368276, 471075844, 1010052132