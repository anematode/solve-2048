package solve_2048
import "core:fmt"
import "core:os"
import "core:simd"

main :: proc()
{
	TEST :: #config(TEST, false)
	when TEST { test(); }
}

test :: proc() -> !
{
	//p: Position = 0x0123456789ABCDEF;
	p: Position = 0x010301000;

	fmt.println(to_string(p));

	fmt.println(tile_sum(p))

	a, b : #simd [8] f32 =
		{1, 2, 3, 4, 5, 6, 7, 8},
		{1, 2, 3, 4, 5, 6, 7, 8};
	fmt.println(a)
	fmt.println(b)
	fmt.println(__test_add(a, b))

	os.exit(0);
}
