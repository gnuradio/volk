.function volk_8i_convert_64f_a_orc_impl
.dest 8 dst
.source 1 src
.temp 8 dtmp
.temp 4 lsrc
.temp 2 ssrc
convsbw ssrc, src
convswl lsrc, ssrc
convld dtmp, lsrc
muld dst, dtmp, 0.0078125L
