        .text
        .align 2
        .type volk_32f_s32f_multiply_32f_sifive_u74, @function
        .global volk_32f_s32f_multiply_32f_sifive_u74

volk_32f_s32f_multiply_32f_sifive_u74:
        # Input:
        # a0  out
        # a1  in
        # fa0 scalar
        # a2  size

        # Main loop in 8x unrolled.

        # Split counter into main and final loop.
        # a5  main loop counter
        # a2  closing loop counter
        srli    a5,a2,3
        andi    a2,a2,7
        slli    a5,a5,5
        beqz    a5,.dolastloop
        add     a5,a0,a5

        .align 2
.loop:
        flw     fa1,0(a1)
        addi    a0,a0,32      # increment output (free, running on pipeline A)

        flw     fa2,4(a1)
        flw     fa3,8(a1)
        flw     fa4,12(a1)
        flw     fa5,16(a1)
        flw     fa6,20(a1)
        flw     fa7,24(a1)
        flw     ft8,28(a1)
        addi    a1,a1,32      # increment input (free, running on pipeline A)

        fmul.s  fa1,fa1,fa0
        fmul.s  fa2,fa2,fa0
        fmul.s  fa3,fa3,fa0
        fmul.s  fa4,fa4,fa0
        fmul.s  fa5,fa5,fa0
        fmul.s  fa6,fa6,fa0
        fmul.s  fa7,fa7,fa0
        fmul.s  ft8,ft8,fa0

        fsw     fa1,-32(a0)
        fsw     fa2,-28(a0)
        fsw     fa3,-24(a0)
        fsw     fa4,-20(a0)
        fsw     fa5,-16(a0)
        fsw     fa6,-12(a0)
        fsw     fa7,-8(a0)
        fsw     ft8,-4(a0)

        bne    a5,a0,.loop

        .align 2
.dolastloop:
        # TODO: is branch assumed to be taken or not?
        beqz    a2,.done

        # Everything below is less optimized. In theory we could split
        # this into more partial unrolled loops, but it's at most 7
        # iterations, so not clear that it's worth it.

        # make a2 a pointer to the last entry.
        slli    a2,a2,2
        add     a2,a0,a2   # Stall!

        .align 2
.lastloop:
        flw     fa5,0(a1)     # Latency: 2
        addi    a0,a0,4       # Increment out
        fmul.s  fa5,fa5,fa0   # Stalled for a cycle or two. Latency: 5
        addi    a1,a1,4       # Increment in
        fsw     fa5,-4(a0)    # Stalled for a couple of cycles waiting for mul.
        bne     a2,a0,.lastloop

	.align 2
.done:
        ret
	.size	volk_32f_s32f_multiply_32f_sifive_u74, .-volk_32f_s32f_multiply_32f_sifive_u74
