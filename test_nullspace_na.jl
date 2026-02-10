using LinearAlgebra
using FastLapackInterface
using BenchmarkTools

nc, nu, nx = 2, 8, 5

# data matrices
H = rand(nu,  nu)
H .+= H'
H = H * H'
AT = rand(nc, nu) # jacobian
params_eq = rand(nu + nc, nx + 1)
α_β = @views params_eq[1:nu, :]
ψ_ω = @views params_eq[nu+1:end, :]

H1 = copy(H)
AT1 = copy(AT)
params_eq1 = copy(params_eq)

# workspace nullspace
ZY = zeros(nu, nu)
Z = @views ZY[:, nc+1:end]
Y = @views ZY[:, 1:nc]
AY = zeros(nc, nc)
ws_ZY = Workspace(LAPACK.geqrf!, ZY)
ws_AY = Workspace(LAPACK.getrf!, AY)
y_nx1_tmp = zeros(nu, nx+1)
y_nx1_tmp1 = zeros(nu, nx+1)
z_nx1_tmp = zeros(nu-nc, nx+1)
HZ_tmp = zeros(nu, nu-nc)
ZHZ_tmp = zeros(nu-nc, nu-nc)
ws_ZHZ = Workspace(LAPACK.pstrf!, ZHZ_tmp)

# make struct

function test(H, AT, ZY, Z, Y, AY, ws_AY, ws_ZY, ψ_ω, α_β, ws_ZHZ, ZHZ_tmp, HZ_tmp, y_nx1_tmp, y_nx1_tmp1, z_nx1_tmp, nc)
    ZY .= 0.
    @views ZY[1:end, 1:nc] .= AT'
    LAPACK.geqrf!(ws_ZY, ZY)
    LAPACK.orgqr!(ws_ZY, ZY)

    # compute α_y, β_y, i.e., the component of α, β in Y basis
    mul!(AY, AT, Y)
    LAPACK.getrf!(ws_AY, AY)
    LAPACK.getrs!(ws_AY, 'N', AY, ψ_ω)

    # compute Z' (RHS_top - H * Y * [α_y β_y])
    mul!(y_nx1_tmp, Y, ψ_ω)
    mul!(α_β, H, y_nx1_tmp, -1.0, 1.0)

    mul!(z_nx1_tmp, Z', α_β)

    # compute reduced Hessian Z'HZ
    mul!(HZ_tmp, H, Z)
    mul!(ZHZ_tmp, Z', HZ_tmp)
    ZHZ_tmp, piv, rank_, info = LAPACK.pstrf!(ws_ZHZ, 'U', ZHZ_tmp, 1e-12)
    ch_ = CholeskyPivoted(ZHZ_tmp, 'U', piv, rank_, 1e-12, info)
    if info > 0
        throw("asdasda")
    end
    ldiv!(ch_, z_nx1_tmp) # allocates because permute! call allocates

    # recover [ψ, ω]
    # compute RHS_top - H * Y * [α β] (i.e., RHS of eqn to get ψ, ω) by adding Z[α_z β_z]
    mul!(y_nx1_tmp1, Z, z_nx1_tmp)  # Y[α_y β_y] + Z[α_z β_z]
    mul!(α_β, H, y_nx1_tmp1, -1.0, 1.0)
    mul!(ψ_ω, Y', α_β)
    LAPACK.getrs!(ws_AY, 'T', AY, ψ_ω)

    # recover [α β]
    α_β .= y_nx1_tmp
    α_β .+= y_nx1_tmp1
end

b = @benchmark test($H, $AT, $ZY, $Z, $Y, $AY, $ws_AY, $ws_ZY, $ψ_ω, $α_β, $ws_ZHZ, $ZHZ_tmp, $HZ_tmp, $y_nx1_tmp, $y_nx1_tmp1, $z_nx1_tmp, $nc)
display(b)
# test(H, AT, ZY, Z, Y, AY, ws_AY, ws_ZY, ψ_ω, α_β, ws_ZHZ, ZHZ_tmp, HZ_tmp, y_nx1_tmp, y_nx1_tmp1, z_nx1_tmp, nc)

K = zeros(nu+nc, nu+nc)
K_bl = @views K[nu+1:end, 1:nu]
K_tr = @views K[1:nu, nu+1:end]
K_tl = @views K[1:nu, 1:nu]
ws_K = Workspace(LAPACK.sytrf_rook!, K)

function test_ldlt(H, AT, K, K_tl, K_bl, K_tr, ws_K, params_eq)
    K .= 0
    K_bl .= AT
    K_tr .= AT'
    K_tl .= H
    Ap, ipiv, info = LAPACK.sytrf_rook!(ws_K, 'L', K)
    bk_ = LinearAlgebra.BunchKaufman(Ap, ipiv, 'L', true, true, info)
    ldiv!(bk_, params_eq)
end

b1 = @benchmark test_ldlt($H, $AT, $K, $K_tl, $K_bl, $K_tr, $ws_K, $params_eq)
display(b1)
# test_ldlt(H1, AT1, K, K_tl, K_bl, K_tr, ws_K, params_eq1);
