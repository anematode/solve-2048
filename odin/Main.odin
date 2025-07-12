package solve_2048
import "core:fmt"
import "core:testing"

main :: proc()
{
	p: Position = 0xE89022A71C848C61;
	fmt.println(to_string(p))
	fmt.println(tile_sum(p))
	fmt.println(tile_sum_real(p))
}



	//fmt.println(to_string(make_canonical(p)))
//
	//a, b : #simd [8] f32 =
	//	{1, 2, 3, 4, 5, 6, 7, 8},
	//	{2, 4, 6, 8, 10, 12, 14, 16};
	//	//{1, 2, 3, 4},
	//	//{2, 2, 3, 4};
//
	//fmt.println("input 1  ", a)
	//fmt.println("input 2  ", b)
//
	////intrinsics.debug_trap();
//
	//c := __test_add(a, b)
	//fmt.println("result   ", c)
	//
	//d := a + b;
	//fmt.println("real     ", d)
//
	//os.exit(0);
