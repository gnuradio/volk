        .text
        .align 2
        .type   volk_32fc_x2_dot_prod_32fc_sifive_u74, @function
        .global volk_32fc_x2_dot_prod_32fc_sifive_u74

        #
        # RISC-V implementation using only I and F sets.
        # About 41% less CPU use than GCC, measured with volk_profile,
        # and a test gnuradio graph using Freq XLAT FIR filter.
        #
        # The generic C code is also 2x unrolled, but its main flaw
        # seems to be not properly fusing into fmadd and fnmsub.
        #
        # Focus of this hand coded assembly:
        # * Better use of fused multiply.
        # * Try to maximize space between write and read.
        #
        # Instruction order has been done manually and benchmarked,
        # and may not be optimal.
        #
volk_32fc_x2_dot_prod_32fc_sifive_u74:
        # a0: out
        # a1: in
        # a2: taps
        # a3: number of points

        # Calculate end of main loop.
        and     a4,a3,1
        xor     a4,a3,a4
        slli    a5,a4,3
        add     a5,a5,a1

        # Output regs.
        fmv.w.x ft0,zero
        fmv.w.x ft1,zero
        fmv.w.x ft2,zero
        fmv.w.x ft3,zero
        fmv.w.x ft4,zero
        fmv.w.x ft5,zero
        fmv.w.x ft6,zero
        fmv.w.x ft7,zero
        beq     a1,a5,.endloop

        # Main loop two complexes at a time.
.loop:
        # Load input in order of when it'll be used.
        # flw has 2 cycle latency, 1 cycle repeat.
        flw     ft8,0(a1)               # in0
        flw     ft9,0(a2)               # tp0
        flw     ft10,4(a2)              # tp1
        flw     ft11,4(a1)              # in1

        # None of the fused multiple-adds have a write-read stall.
        # FMA, like mul and add, have 5 cycle latency, 1 cycle repeat.
        fmadd.s  ft0,ft8, ft9, ft0      # in0*tp0
        flw      fa0,8(a1)              # in0
        fmadd.s  ft1,ft8, ft10,ft1      # in0*tp1
        flw      fa1,8(a2)              # tp0
        fnmsub.s ft2,ft11,ft10,ft2      # -in1*tp1
        flw      fa2,12(a2)             # tp1
        fmadd.s  ft3,ft11,ft9, ft3      # in1*tp0
        flw      fa3,12(a1)             # in1

        fmadd.s  ft4,fa0,fa1,ft4        # in0*tp0
        addi     a1,a1,16               # free ride in pipeline A.
        fmadd.s  ft5,fa0,fa2,ft5        # in0*tp1
        addi     a2,a2,16               # free ride in pipeline A.
        fnmsub.s ft6,fa3,fa2,ft6        # -in1*tp1
        fmadd.s  ft7,fa3,fa1,ft7        # in1*tp0
        bne      a1,a5,.loop

.endloop:
        # Check if odd number of inputs.
        andi    a3,a3,1
        beqz    a3,.done

        # Do odd one complex.
        flw     fa0,0(a1) # in0
        flw     fa1,0(a2) # tp0
        flw     fa2,4(a2) # tp1
        flw     fa3,4(a1) # in1

        fmadd.s  ft4,fa0,fa1,ft4   # in0*tp0
        fmadd.s  ft5,fa0,fa2,ft5   # in0*tp1
        fnmsub.s ft6,fa3,fa2,ft6   # -in1*tp1
        fmadd.s  ft7,fa3,fa1,ft7   # in1*tp0
.done:
        # Some one-time stalling here.
        # Latency 5, repeat 1.
        fadd.s  ft0,ft0,ft2
        fadd.s  ft1,ft1,ft3
        fadd.s  ft0,ft0,ft4
        fadd.s  ft1,ft1,ft5
        fadd.s  ft0,ft0,ft6
        fadd.s  ft1,ft1,ft7
        # fsw has latency 4, repeat 1.
        fsw     ft0,0(a0)
        fsw     ft1,4(a0)
        ret

        .size volk_32fc_x2_dot_prod_32fc_sifive_u74, .-volk_32fc_x2_dot_prod_32fc_sifive_u74
