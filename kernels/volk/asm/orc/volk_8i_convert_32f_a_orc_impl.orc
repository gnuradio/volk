.function volk_8i_convert_32f_a_orc_impl
.dest 4 dst
.source 1 src
.temp 4 ftmp
.temp 4 lsrc
.temp 2 ssrc
convsbw ssrc, src
convswl lsrc, ssrc
convlf ftmp, lsrc
mulf dst, ftmp, 0.0078125
