using LinearAlgebra

##----------- Type utilities for generic Bunch-Kaufman implementation ------------
# Generic real type. Any real number type should able to approximate
# real numbers, and thus be closed under arithmetic operations.
# Therefore so Int, Complex{Int}, etc. are excluded.
ClosedReal = T where T <: Union{AbstractFloat, Rational}
# Similarly, we also use a closed scalar type
ClosedScalar = Union{T, Complex{T}} where T <: ClosedReal
##--------------------------------------------------------------------------------


"""
inertia(B::BunchKaufman; atol::Real=0, rtol::Real=atol>0 ? 0 : n*ϵ) ->
    np::Union{Nothing,Integer}, nn::Union{Nothing,Integer}, nz::Integer

`inertia` computes the numerical inertia (the number of positive,
negative and zero eigenvalues, given by `np`, `nn` and `nz`,
respectively) of a real symmetric of Hermitian matrix `B` that has been
factored using the Bunch-Kaufman algorithm. For complex symmetric
matrices the inertia is not defined. in that case `np` and `nn` are set
to `nothing`, but the function still returns the number of zero
eigenvalues. The inertia is computed by counting the eigenvalues signs
of `B.D`. The number of zero eigenvalues is computed as the number of
estimated eigenvalues with complex 1-norm (defined as `|re(.)|+|im(.)|`)
less or equal than `max(atol, rtol*s₁)`, where `s₁` is an upper bound of
the largest singular value of `B.D`, `σ₁` (more specifically,
`0.5*s₁ <= σ₁ <= s₁` for real matrices and `0.35*s₁ <= σ₁ <= s₁` for
complex matrices). `atol` and `rtol` are the absolute and relative
tolerances, respectively. The default relative tolerance is `n*ϵ`, where
`n` is the size of  of `A`, and `ϵ` is the [`eps`](@ref) of the number
type of `A`, if this type is a subtype of `AbstractFloat`. In any other
case (if the number type of `A` is `Rational`, for example) `ϵ` is set
to `0`.

!!! note
    Numerical inertia can be a sensitive and imprecise characterization of
    ill-conditioned matrices with eigenvalues that are close in magnitude to the
    threshold tolerance `max(atol, rtol*s₁)`. In such cases, slight perturbations
    to the Bunch-Kaufman computation or to the matrix can change the result of
    `rank` by pushing one or more eigenvalues across the threshold. These
    variations can even occur due to changes in floating-point errors between
    different Julia versions, architectures, compilers, or operating systems.
    In particular, the size of the entries of the tringular factor directly
    influende the scale of the eigenvalues of the diagonal factor, so it is
    strongly recommended to use rook pivoting is the inertia is going to be
    computed.
    On the other hand, if the matrix has rational entries, the inertia
    computation is guaranteed is to be exact, as long as there is no
    under/overflow in the underlying integer type (and in such cases Julia itself
    throws an error), or a positive tolerance (absolute or relative) is
    specified.
"""
function inertia!(B::BunchKaufman{TS}, d::Vector{TS}, e::Vector{TS};
    atol::TR = TR(0),
    rtol::TR = TR(0)
    ) where TS <: ClosedScalar{TR} where TR <: ClosedReal

    # Check if matrix is complex symmetric
    get_inertia = !(issymmetric(B) && TS <: Complex)

    # Initialize outputs
    np, nn, nz = get_inertia ? (0, 0, 0) : (nothing, nothing, 0)

    # Compute matrix size
    N = size(B, 1)

    # Quick return if possible
    if N == 0; return np, nn, nz; end

    # Compute default relative tolerance
    if rtol <= 0 && atol <= 0
        rtol = TR <: AbstractFloat ? (N * eps(TR)) : TR(0)
    end

    # We use the complex 1-norm for complex matrices
    real_matrix = (TS <: Real)
    abs1_fun = real_matrix ? abs : cabs1
    real_fun = real_matrix ? identity : real

    # Check if we must track the largest singular value
    get_s1 = (rtol > 0)

    # Constant for lower bound estimation of the smallest eigenvalue in 2x2 blocks.
    # The best (largest) value for complex matrices is 1/sqrt(2), but for rational
    # matrices we use the small denominator approximation 12/17, in order to not
    # increase the denominator size too much in computations. The error of this
    # approximation is ≤0.2%, and we still get a valid lower bound.
    c = real_matrix ? TR(1) : (TR <: AbstractFloat ? 1/sqrt(TR(2)) : TR(12//17))

    # First pass, estimate largest singular value and group together size-1 blocks
    D = get_D!(B, d, e)
    s1 = TR(0)
    i = 1
    while i <= N; @inbounds begin
        if i < N && D[i,i+1] != 0
            # 2x2 block
            # The largest singular value of a 2x2 matrix is between [1, 2] times
            # its complex max-norm, which is between [c, 1] times the largest
            # complex 1-norm among the entries of the 2x2 matrix. See "Roger
            # Horn and Charles Johnson. Matrix Analysis, 2nd Edition, 5.6.P23".
            abs_Dii = abs1_fun(D[i,i])
            abs_Dxx = abs1_fun(D[i+1,i+1])
            s1_block = 2 * max(abs_Dii, abs1_fun(D[i,i+1]), abs_Dxx)
            if get_s1; s1 = max(s1, s1_block); end
            # Lower bound on the smallest eigenvalue complex 2-norm is
            # abs(λ₂) ≥ abs(det(block)) / s1_block
            # so the bound in terms of the complex 1-norm becomes
            # abs1_fun(λ₂) ≥ c * abs1_fun(det(block)) / s1_block
            # For rational matrices, if λ₂=0 then det(block)=0 and then the bound
            # becomes zero too. If λ₁=0 too then the block has all zero entries
            # and 's1_block'=0, but 'D[i,i+1]' != 0 and so 's1_block' > 0. However, we
            # may still have that 'smin_block'≈0, then the value of 'smin_block' may not
            # be accurate. In that case the counting routine will detect that both
            # eigenvalues are zero without using 'smin_block', so it doesn't matter.
            # TODO: is this the most numerically stable way to compute the determinant?
            # TODO: is this the best way to avoid under/overflow?
            if abs_Dii >= abs_Dxx
                smin_block = c * abs1_fun((D[i,i]/s1_block)*D[i+1,i+1] -
                    (D[i,i+1]/s1_block)*D[i+1,i])
            else
                smin_block = c * abs1_fun(D[i,i]*(D[i+1,i+1]/s1_block) -
                    (D[i,i+1]/s1_block)*D[i+1,i])
            end
            # Store lower bound in-place in the lower off-diagonal and upper bound
            # in-place in the upper off-diagonal. The trace is stored in the first
            # diagonal entry block, but only if the full inertia is needed.
            D[i,i+1] = s1_block
            D[i+1,i] = smin_block
            if get_inertia; D[i,i] += D[i+1,i+1]; end
            i += 2
        else
            # 1x1 block
            if get_s1; s1 = max(s1, abs1_fun(D[i,i])); end
            i += 1
        end
    end; end

    # Second pass, count eigenvalue signs
    tol = max(atol, rtol * s1)
    i = 1
    while i <= N; @inbounds begin
        if i < N && D[i,i+1] != 0
            # 2x2 block. For the counting of zero eigenvalues we use the lower bound on the
            # eigenvalues' magnitude. This way, if an eigenvalue is deemed non-zero, then
            # it is guaranteed that its magnitude is greater than the tolerance.
            s1_block = real_fun(D[i,i+1])
            if (c / 2) * s1_block <= tol
                # Lower bound of largest eigenvalue is smaller than the tolerance,
                # we consider the both eigenvalues of this block to be zero.
                nz += 2
                i += 2
                continue
            end
            # Reaching this part of the lopp implies that 's1_block' != 0.
            smin_block = real_fun(D[i+1,i])
            trace_block = real_fun(D[i,i])
            if smin_block > tol || trace_block == 0
                # If first condition holds then the lower bound of the smallest eigenvalue
                # is larger than the tolerance. If the second condition holds then the trace
                # is exactly zero, so both eigenvalues have the same magnitude, and we
                # already know that the largest one is non-zero. In any case we conclude
                # that both eigenvalues are non-zero.
                if get_inertia
                    # The eigenvalues of a 2x2 block are guaranteed to be a
                    # positive-negative pair.
                    np += 1
                    nn += 1
                end
            else
                # The lower bound of smallest eigenvalue is smaller than the tolerance and
                # the trace is non-zero, so we consider the smallest eigenvalues of this
                # block to be zero.
                nz += 1
                if get_inertia
                    # The trace is non-zero, and its sign is the same of the largest
                    # eigenvalue.
                    if trace_block >= 0
                        np += 1
                    else
                        nn += 1
                    end
                end
            end
            i += 2
        else
            # 1x1 block
            if get_inertia
                eig = real_fun(D[i,i])
                if eig > tol
                    np += 1
                elseif eig < -tol
                    nn += 1
                else
                    nz += 1
                end
            elseif abs1_fun(D[i,i]) <= tol
                nz += 1
            end
            i += 1
        end
    end; end

    return np, nn, nz
end

function get_D!(F::BunchKaufman{TS}, d::Vector{TS}, e::Vector{TS},
            )  where TS <: ClosedScalar{TR} where TR <: ClosedReal
    # Inputs must be 1-indexed; bounds may not be checked.
    Base.require_one_based_indexing(F.LD, F.ipiv)

    # Extract necessary variables
    A, ipiv, rook = F.LD, F.ipiv, F.rook

    # Get size of matrix
    N = size(A)[1]

    #   Quick return if possible
    if N == 0; return nothing, e, d; end
    
    # d .= @views diag(A, 0)
    for i in 1:N
        d[i] = A[i, i]
    end

    # Main loops
    upper = (F.uplo == 'U')
    @inline icond_d = upper ? i -> i > 1 : i -> i < N
    @inline icond_T = upper ? i -> i >= 1 : i -> i <= N
    @inline inext = upper ? i -> i - 1 : i -> i + 1
    #   Convert VALUE
    i = upper ? N : 1
    e[N+1-i] = 0
    while icond_d(i); @inbounds begin
        if ipiv[i] < 0
            ix = inext(i)
            e[i] = A[ix,i]
            e[ix] = 0
            if upper; i -= 1; else; i += 1; end
        else
            e[i] = 0
        end
        if upper; i -= 1; else; i += 1; end
    end; end
    
    if getfield(F, :uplo) == 'L'
        odl = @views e[1:N - 1]
        md = @views d[1:N]
        return Tridiagonal(odl, md, odl)
    else # 'U'
        odu = @views e[2:N]
        md = @views d[1:N]
        return Tridiagonal(odu, md, odu)
    end
end

function inertia_correction!(bk_ws::BunchKaufmanWs{T}, kkt_mat::Matrix{T}, D_cache::Pair{Vector{T}},
                num_controls::Int64, μ::T, reg::T, reg_last::T, options::Options) where T
    status = 0
    δ_c = 0.0
    Ap, ipiv, info = LAPACK.sytrf_rook!(bk_ws, 'U', kkt_mat)
    bk = LinearAlgebra.BunchKaufman(Ap, ipiv, 'U', true, true, info)
    if info > 0
        δ_c = options.δ_c * μ^options.κ_c
    end
    np, _, _ = inertia!(bk, D_cache[1], D_cache[2]; atol=T(options.linsolve_tol))
    if np != num_controls || info != 0
        if iszero(reg) # initial setting of regularisation
            reg = (reg_last == 0.0) ? options.reg_1 : max(options.reg_min, options.κ_w_m * reg_last)
        else
            reg = (reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
        end
        status = 1
    end
    return bk, status, reg, δ_c
end
