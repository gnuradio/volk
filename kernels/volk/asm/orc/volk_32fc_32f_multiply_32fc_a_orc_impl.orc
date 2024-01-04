.function volk_32fc_32f_multiply_32fc_a_orc_impl
.source 8 src1 lv_32fc_t
.source 4 src2 float
.dest 8 dst lv_32fc_t
.temp 8 tmp
mergelq tmp, src2, src2
x2 mulf dst, src1, tmp
