.function volk_8u_convert_32f_a_orc_impl
.dest 4 dst
.source 1 src
.temp 4 ftmp
.temp 4 lsrc
.temp 2 ssrc
convubw ssrc, src
convuwl lsrc, ssrc
convlf ftmp, lsrc
addf ftmp, ftmp, -127.5
mulf dst, ftmp, 0.00784313725490196
