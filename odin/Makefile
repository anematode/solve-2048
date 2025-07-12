FLAGS=-microarch:x86-64-v4 -target-features:avx,avx2,avx512f,avx512vl -debug

test: solve-2048
	@./solve-2048
solve-2048: *.odin *.asm Makefile
	odin build . -define:TEST=true $(FLAGS) -out:solve-2048
asm:
	odin build . -define:TEST=true -build-mode:asm $(FLAGS)
