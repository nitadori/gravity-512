..B1.42:                        # Preds ..B1.42 ..B1.41
                                # Execution count [2.50e+01]
        vsubps    (%rcx,%r10){1to16}, %zmm10, %zmm18            #196.3 c1
        vmovaps   %zmm4, %zmm14                                 #196.3 c1
        vsubps    4(%rcx,%r10){1to16}, %zmm9, %zmm19            #196.3 c1
        addq      $1, %r11                                      #196.3 c1
        vfmadd231ps %zmm18, %zmm18, %zmm14                      #196.3 c7 stall 2
        vsubps    8(%rcx,%r10){1to16}, %zmm8, %zmm21            #196.3 c7
        vfmadd231ps %zmm19, %zmm19, %zmm14                      #196.3 c13 stall 2
        vfmadd231ps %zmm21, %zmm21, %zmm14                      #196.3 c19 stall 2
        vrsqrt28ps %zmm14, %zmm15{%k1}{z}                       #196.3 c25 stall 2
        vmulps    12(%rcx,%r10){1to16}, %zmm15, %zmm16          #196.3 c33 stall 3
        vmulps    %zmm15, %zmm15, %zmm17                        #196.3 c33
        addq      $16, %r10                                     #196.3 c33
        vmulps    %zmm17, %zmm16, %zmm20                        #196.3 c39 stall 2
        vfnmadd231ps %zmm18, %zmm20, %zmm13                     #196.3 c45 stall 2
        vfnmadd231ps %zmm19, %zmm20, %zmm12                     #196.3 c45
        vfnmadd231ps %zmm21, %zmm20, %zmm11                     #196.3 c51 stall 2
        cmpq      %r15, %r11                                    #196.3 c51
        jb        ..B1.42       # Prob 82%                      #196.3 c53
