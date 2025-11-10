#include <riscv_vector.h>

int main() {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t v = __riscv_vfmv_v_f_f32m2(0.5f, vlmax);
    
    // Compute 1 - v*v using vfnmsac: vd = -(vs1 * vs2) + vd
    vfloat32m2_t result = __riscv_vfnmsac(cf1, v, v, vlmax);
    
    return 0;
}
