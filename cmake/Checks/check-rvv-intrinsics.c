#if (__riscv_v_intrinsic >= 1000000 || __clang_major__ >= 18 || __GNUC__ >= 14)
int main() { return 0; }
#else
#error "rvv intrinsics aren't supported"
#endif
