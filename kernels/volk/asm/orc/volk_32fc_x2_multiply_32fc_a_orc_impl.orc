.function volk_32fc_x2_multiply_32fc_a_orc_impl
.source 8 src1 lv_32fc_t
.source 8 src2 lv_32fc_t
.dest 8 dst lv_32fc_t
.temp 8 iqprod
.temp 4 real
.temp 4 imag
.temp 4 ac
.temp 4 bd
.temp 8 swapped
x2 mulf iqprod, src1, src2
splitql bd, ac, iqprod
subf real, ac, bd
swaplq swapped, src1
x2 mulf iqprod, swapped, src2
splitql bd, ac, iqprod
addf imag, ac, bd
mergelq dst, real, imag
