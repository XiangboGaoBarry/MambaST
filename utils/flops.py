def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops




if __name__ == "__main__":
    
    B = 1
    verbose = False
    verbose_harsh = True
    layer_hw = {0: 80, 1: 40, 2: 20}
    layer_dim = {0: 256, 1: 512, 2: 1024}
    
    # K1, K2, K3, dual_scan
    setups = [
        (1, 1, 1, False),
        (2, 1, 1, False),
        (2, 2, 1, False),
        (4, 1, 1, False),
        (4, 2, 1, False),
        (4, 2, 2, False),
        (4, 4, 2, False),
        (4, 2, 1, True),
    ]
    for setup in setups:
        print("Setup", setup)
        K1, K2, K3, dual_scan = setup
        Ks = (K1, K2, K3)
        total_flops = 0
        for layer in range(3):
            K = Ks[layer]
            hw = layer_hw[layer]
            dim_in = layer_dim[layer]
            dim = dim_in // K
            total_flops_layer = 0
            for i in range(K):
                hw_ = hw // (2 ** i)
                L = hw_ * hw_ * 2
                
                if dual_scan:
                    L = L * 2
                    D = dim // 2
                else:
                    D = dim
                
                flops_extra = 0
                
                patch_embeding = 2 * B * L * dim_in * D * (2**i) * (2**i)
                pos_embeding = B * L * D
                modality_embeding = pos_embeding
                residual = pos_embeding * 8
                upsample = 5 * 2 * D * hw * hw 
                norm = L * D * 9 * 7
                flops_extra += patch_embeding + pos_embeding + modality_embeding + residual + upsample + norm
                
                
                
                

                if verbose:
                    print("\t\tLayer\t\t", layer, "K", K, "L", L, "D", D)
                    print("\t\tPatch\t\t", patch_embeding / 1e9, "GFLOPS")
                    print("\t\tPos\t\t", pos_embeding / 1e9, "GFLOPS")
                    print("\t\tModality\t", modality_embeding / 1e9, "GFLOPS")
                    print("\t\tResidual\t", residual / 1e9, "GFLOPS")
                    print("\t\tUpsample\t", upsample / 1e9, "GFLOPS")
                    print("\t\tNorm\t\t", norm / 1e9, "GFLOPS")
                    print("\t\tFLOPs Extra\t", flops_extra / 1e9, "GFLOPS")
                flops = flops_selective_scan_ref(B=1, L=L, D=D, N=16, with_D=True, with_Z=True, with_Group=False, with_complex=False) * 8
                if verbose:
                    print("\t\tK", K, "FLOPS", flops / 1e9, "GFLOPS")
                total_flops_layer += flops + flops_extra
            if verbose_harsh:
                print("\tTotal FLOPS Layer", layer, "FLOPS", total_flops_layer / 1e9, "GFLOPS")
            total_flops += total_flops_layer
        print("Total FLOPS", total_flops / 1e9, "GFLOPS")