.function volk_16ic_deinterleave_16i_x2_a_orc_impl
.dest 2 idst int16_t
.dest 2 qdst int16_t
.source 4 src lv_16sc_t
splitlw qdst, idst, src
