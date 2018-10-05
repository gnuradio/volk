.function volk_8u_convert_64f_a_orc_impl
.dest 8 dst
.source 1 src
.temp 8 dtmp
.temp 4 lsrc
.temp 2 ssrc
convubw ssrc, src
convuwl lsrc, ssrc
convld dtmp, lsrc
subd dtmp, dtmp, 127.5L
muld dst, dtmp, 0.00784313725490196L
