#include <riscv_vector.h>
#include <stdio.h>

int main() {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t v = __riscv_vfmv_v_f_f32m2(0.5f, vlmax);
    
    // Test vfmsac
    vfloat32m2_t result = __riscv_vfmsac(cf1, v, v, vlmax);
    
    printf("Test completed\n");
    return 0;
}
