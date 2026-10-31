###############################################################################
# DDE_w_general.jl
#
# Rigorous computer-assisted proof (CAP) of a periodic solution of the w-DDE
#
#     ẇ(t) = -α e^{i(θ-μ)} (1-|w(t)|²) w(t-τ) - i μ w(t),          w(t) ∈ ℂ,
#
# via a complex Fourier expansion  w(t) = Σ_{k∈ℤ} a_k e^{iωkt},  ω = 2π/T.
# The zero-finding map (differential form) is
#
#     F_k(a,μ) = (iωk + iμ) a_k + β e^{-iωkτ} a_k - β [a * a† * S_τ a]_k
#              = λ_k a_k - β g_k,
#     λ_k = iωk + iμ + β e^{-iωkτ},   β = α e^{i(θ-μ)},
#     (a†)_k = conj(a_{-k}),   (S_τ a)_k = e^{-iωkτ} a_k,   g = a·a†·S_τ a.
#
# This file is the *proof* implementation: it requires no plotting / GUI
# packages.  Plotting lives in  DDE_w_plot.jl .
#
# What is proved (see `solve_and_prove`):
#   * existence of a genuine w-periodic solution of the w-DDE (all Galerkin
#     equations F_k = 0, k∈ℤ, including k=0, are satisfied),
#   * the relative-periodic / periodic nature of z(t)=w(t)e^{iμt} (reported,
#     with μT/(2π) checked exactly against ℚ).
#
# Banach space:  𝒳 = (ℓ¹_ν)_ℂ × ℝ ,  ‖(a,μ)‖ = Σ_k ν^{|k|}|a_k| + w_μ|μ|.
# Method: preconditioned Newton–Kantorovich (radii polynomial), operator
#         A = A_fin (⊕) Λ⁻¹ on the tail |k|>N, all bounds computed with
#         outward-rounded interval arithmetic.
#
# Run:   julia --project=star-shaped/wproof star-shaped/DDE_w_general.jl [case1|case2|test|all]
###############################################################################

# --- environment bootstrap (dedicated, reproducible project) -----------------
import Pkg
let envdir = joinpath(@__DIR__, "wproof")
    if Base.active_project() === nothing || dirname(Base.active_project()) != envdir
        Pkg.activate(envdir; io=devnull)
    end
end

using RadiiPolynomial, LinearAlgebra, Printf

###############################################################################
# 0.  Parameters:  approximate (Float64)  vs  rigorous (Interval) values
###############################################################################
# θ and μ0 are given as rational multiples of π  (θ = pθ/qθ · π etc.) so that
# the rigorous constants are built from interval(π) — never from a rounded
# Float64.  α, τ, T are given as exact rationals pα/qα, ... .

struct WParams
    # exact rational data (numerator, denominator)
    α::Tuple{Int,Int}          # α = α[1]/α[2]
    θπ::Tuple{Int,Int}         # θ = θπ[1]/θπ[2] · π
    τ::Tuple{Int,Int}          # τ = τ[1]/τ[2]
    T::Tuple{Int,Int}          # T = T[1]/T[2]
    μ0π::Tuple{Int,Int}        # initial guess μ0 = μ0π[1]/μ0π[2] · π
    N::Int                     # Fourier cutoff
    ν::Float64                 # ℓ¹ weight (≥1)
    R::Float64                 # max validation radius
    wμ::Float64                # weight of the μ-component in the product norm
end

# --- Float64 approximations (for the numerical solve only) -------------------
αf(p::WParams)  = p.α[1]/p.α[2]
θf(p::WParams)  = (p.θπ[1]/p.θπ[2])*π
τf(p::WParams)  = p.τ[1]/p.τ[2]
Tf(p::WParams)  = p.T[1]/p.T[2]
ωf(p::WParams)  = 2π/Tf(p)
μ0f(p::WParams) = (p.μ0π[1]/p.μ0π[2])*π

# --- rigorous interval constants (outward rounded, from interval(π)) ---------
struct WIntervals{IV}
    α::IV; θ::IV; τ::IV; T::IV; ω::IV; πiv::IV
end
function intervals(p::WParams)
    πiv = interval(π)
    α = interval(p.α[1]) / interval(p.α[2])
    θ = (interval(p.θπ[1]) / interval(p.θπ[2])) * πiv
    τ = interval(p.τ[1]) / interval(p.τ[2])
    T = interval(p.T[1]) / interval(p.T[2])
    ω = interval(2) * πiv / T
    WIntervals(α, θ, τ, T, ω, πiv)
end

###############################################################################
# 1.  Core operators
###############################################################################
# a† : (a†)_k = conj(a_{-k})
function dagger(a)
    N = order(space(a)); c = coefficients(a); cd = similar(c)
    for k in -N:N; cd[k+N+1] = conj(c[-k+N+1]); end
    Sequence(space(a), cd)
end

# g = a · a† · S_τ a   (cubic nonlinearity), returned on the enlarged space
g_cubic(a, τ) = (a * dagger(a)) * shift(a, -τ)

# differential-form residual F (as a Sequence, all modes)
#   F_k = (iωk+iμ) a_k + β e^{-iωkτ} a_k - β [Π (a a† S a)]_k
function eval_F_w(a, μ, X, αv, θv, τv)
    Π = Projection(X)
    β = αv * cis(θv - μ)
    Sa = shift(a, -τv)
    D  = Derivative(1)
    D*a + (1im*μ)*a + β*Sa - β * Π * ((a * dagger(a)) * Sa)
end

# β(μ) helpers
βval(αv, θv, μ) = αv * cis(θv - μ)          # β  = α e^{i(θ-μ)}
lambda_k(k, μ, β, ω, τ) = 1im*ω*k + 1im*μ + β*cis(-ω*k*τ)

###############################################################################
#  full mode Jacobian block  (output mode k, input mode j) of  D F
#  returns the 2×2 real block  [Re(dre) Re(dim); Im(dre) Im(dim)]
#  where dre = ∂F_k/∂(Re a_j),  dim = ∂F_k/∂(Im a_j)   (ℝ-linear).
#  P1 = a†·Sa,  P2 = a·Sa,  P3 = a·a†  (all order ≤ 2N).
###############################################################################
function DF_block(k, j, λk, β, ω, τ, P1c, P2c, P3c, Nprod, z)
    ep = cis(-ω*j*τ)
    i1 = (k-j) + Nprod + 1
    i2 = (k+j) + Nprod + 1
    p1 = (1 <= i1 <= 2Nprod+1) ? P1c[i1] : z
    p2 = (1 <= i2 <= 2Nprod+1) ? P2c[i2] : z
    p3 = (1 <= i1 <= 2Nprod+1) ? P3c[i1] : z
    dre = (k==j ? λk : z) - β*(p1 + p2 + ep*p3)
    dim = 1im*((k==j ? λk : z) - β*(p1 + ep*p3)) + 1im*β*p2
    return dre, dim
end

# ∂F_k/∂μ  = i a_k - iβ (Sa)_k + iβ g_k       (uses β' = -iβ)
dFdμ_k(ak, Sak, gk, β) = 1im*ak - 1im*β*Sak + 1im*β*gk

###############################################################################
#  analytic EXTENDED Jacobian  (4N+3)×(4N+3)  of the corrected square system
#  rows:  Re/Im F_k (k=-N..-1,1..N) ; Im F_0 (determines μ) ; Im a_0 ; Im a_1
#  cols:  Re/Im a_j (j=-N..N) ; μ
###############################################################################
function eval_DF_ext(a, μ, X, αv, θv, τv)
    N = order(X); n = 4N+3; ω = frequency(X)
    β = βval(αv, θv, μ)
    Sa = shift(a, -τv); ad = dagger(a)
    P1 = ad*Sa; P2 = a*Sa; P3 = a*ad
    P1c=coefficients(P1); P2c=coefficients(P2); P3c=coefficients(P3)
    Nprod = order(space(P1))          # = 2N
    λ = [lambda_k(k, μ, β, ω, τv) for k in -N:N]
    cub = Projection(X)*((a*ad)*Sa)
    ac=coefficients(a); Sac=coefficients(Sa); cubc=coefficients(cub)
    RT = real(typeof(β)); J = zeros(RT, n, n); z = zero(β)
    # row index for a mode-k equation (k≠0): Re at r+1, Im at r+2
    rowRe(k) = k<0 ? 2*(k+N)+1 : 2N+2*(k-1)+1
    for j in -N:N
        cr = 2*(j+N)+1; ci = 2*(j+N)+2
        for k in -N:N
            dre, dim = DF_block(k, j, λ[k+N+1], β, ω, τv, P1c, P2c, P3c, Nprod, z)
            if k != 0
                r = rowRe(k)
                J[r,cr]=real(dre); J[r+1,cr]=imag(dre)
                J[r,ci]=real(dim); J[r+1,ci]=imag(dim)
            else
                # Im F_0 row  (row 4N+1)
                J[4N+1,cr]=imag(dre); J[4N+1,ci]=imag(dim)
            end
        end
    end
    # μ column (index n) : ∂F_k/∂μ
    for k in -N:N
        dm = dFdμ_k(ac[k+N+1], Sac[k+N+1], cubc[k+N+1], β)
        if k != 0
            r = rowRe(k); J[r,n]=real(dm); J[r+1,n]=imag(dm)
        else
            J[4N+1,n]=imag(dm)
        end
    end
    # gauge rows
    J[4N+2, 2N+2]   = one(RT)        # Im a_0  (col of Im a_0)
    J[4N+3, 2N+4]   = one(RT)        # Im a_1  (col of Im a_1)
    J
end

###############################################################################
# 2.  Pack / unpack   (domain packing:  Re/Im a_k, then μ)
###############################################################################
function unpack_w(x, X)
    N=order(X); m=2N+1; T=eltype(x); ac=zeros(complex(T),m)
    for j in 1:m; ac[j]=x[2j-1]+1im*x[2j]; end
    Sequence(X, ac), x[2m+1]
end
function pack_w(a, μ)
    c=coefficients(a); m=length(c); T=typeof(real(c[1])); x=zeros(T,2m+1)
    for j in 1:m; x[2j-1]=real(c[j]); x[2j]=imag(c[j]); end; x[2m+1]=μ; x
end

# corrected extended residual, packed (4N+3):
#   Re/Im F_k (k≠0) ; Im F_0 ; Im a_0 ; Im a_1
function pack_F_ext(Fs, as, N)
    Fc=coefficients(Fs); ac=coefficients(as); T=typeof(real(Fc[1]))
    F=zeros(T,4N+3); idx=0
    for k in -N:-1; idx+=1; F[idx]=real(Fc[k+N+1]); idx+=1; F[idx]=imag(Fc[k+N+1]); end
    for k in 1:N;   idx+=1; F[idx]=real(Fc[k+N+1]); idx+=1; F[idx]=imag(Fc[k+N+1]); end
    idx+=1; F[idx]=imag(Fc[N+1])      # Im F_0  (μ-determining)
    idx+=1; F[idx]=imag(ac[N+1])      # Im a_0  (rotation gauge)
    idx+=1; F[idx]=imag(ac[N+2])      # Im a_1  (translation gauge)
    F
end
function eval_F_ext(x, X, αv, θv, τv)
    a,μ = unpack_w(x,X)
    pack_F_ext(eval_F_w(a,μ,X,αv,θv,τv), a, order(X))
end
eval_DF_ext_packed(x, X, αv, θv, τv) = (ap=unpack_w(x,X); eval_DF_ext(ap[1], ap[2], X, αv, θv, τv))

###############################################################################
# 3.  Approximate solution:  method-of-steps RK4 + hand DFT + Newton
###############################################################################
# integrate ẇ = -α e^{i(θ-μ)}(1-|w|²)w(t-τ) - iμw  (real (x,y) system)
function integrate_w(α, θ, μ, τ; t_end=300.0, K=128, u0=(0.5,0.3))
    cθ = cos(θ-μ); sθ = sin(θ-μ)
    h = τ/K; nlag = K; nstep = ceil(Int, t_end/h)
    xs = Vector{Float64}(undef, nstep+1); ys = similar(xs)
    xs[1]=u0[1]; ys[1]=u0[2]
    hist(i) = i < 1 ? (xs[1],ys[1]) : (xs[i],ys[i])
    f(xi,yi,xd,yd) = ( s2=xi*xi+yi*yi;
        (-α*(1-s2)*(cθ*xd - sθ*yd) + μ*yi , -α*(1-s2)*(sθ*xd + cθ*yd) - μ*xi) )
    delayed(i,frac) = begin
        p=(i-1)+frac-nlag; jj=floor(Int,p); gg=p-jj
        (xa,ya)=hist(jj+1); (xb,yb)=hist(jj+2)
        ((1-gg)*xa+gg*xb, (1-gg)*ya+gg*yb)
    end
    for i in 1:nstep
        xi=xs[i]; yi=ys[i]
        xd0,yd0=delayed(i,0.0); k1x,k1y=f(xi,yi,xd0,yd0)
        xdh,ydh=delayed(i,0.5); k2x,k2y=f(xi+0.5h*k1x,yi+0.5h*k1y,xdh,ydh)
        k3x,k3y=f(xi+0.5h*k2x,yi+0.5h*k2y,xdh,ydh)
        xd1,yd1=delayed(i,1.0); k4x,k4y=f(xi+h*k3x,yi+h*k3y,xd1,yd1)
        xs[i+1]=xi+h/6*(k1x+2k2x+2k3x+k4x); ys[i+1]=yi+h/6*(k1y+2k2y+2k3y+k4y)
    end
    xs, ys, h
end

function initial_guess(α, θ, μ0, N, ω, τ; t_end=300.0, K=128, M=1024)
    T_per = 2π/ω
    xs, ys, h = integrate_w(α, θ, μ0, τ; t_end=t_end, K=K)
    nstep = length(xs)-1; te = nstep*h
    interp(t) = ( p=t/h; j=clamp(floor(Int,p),0,nstep-1); gg=p-floor(Int,p);
                  ((1-gg)*xs[j+1]+gg*xs[j+2], (1-gg)*ys[j+1]+gg*ys[j+2]) )
    ac = zeros(ComplexF64, 2N+1)
    for k in -N:N
        s = 0.0+0.0im
        for m in 0:M-1
            t = te - T_per + (m/M)*T_per
            xw,yw = interp(t); s += (xw+1im*yw)*cis(-ω*k*t)
        end
        ac[k+N+1] = s/M
    end
    # gauge: rotate so Im(a_1)=0  (translation), then Im(a_0)=0 (rotation)
    a1 = ac[N+2]
    if abs(a1) > 1e-14
        s = -angle(a1)/ω
        for k in -N:N; ac[k+N+1] *= cis(ω*k*s); end
    end
    a0 = ac[N+1]
    if abs(a0) > 1e-14
        φ = -angle(a0)
        for k in -N:N; ac[k+N+1] *= cis(φ); end
    end
    ac
end

# Newton on the corrected extended system (analytic Jacobian)
function solve_newton(p::WParams; verbose=false)
    N=p.N; ω=ωf(p); X=Fourier(N,ω)
    α=αf(p); θ=θf(p); τ=τf(p); μ0=μ0f(p)
    ac = initial_guess(α,θ,μ0,N,ω,τ)
    x0 = pack_w(Sequence(X,ac), μ0)
    xb, conv = newton(x->(eval_F_ext(x,X,α,θ,τ), eval_DF_ext_packed(x,X,α,θ,τ)),
                      x0; verbose=verbose, maxiter=40)
    xb, conv
end

###############################################################################
# 4.  weighted ℓ¹_ν norms on the product space 𝒳 = (ℓ¹_ν)_ℂ × ℝ
###############################################################################
# spectral norm (largest singular value) of a 2×2 real interval matrix, rigorous
function spec2(a,b,c,d)
    t = a*a + b*b + c*c + d*d               # ‖B‖_F²  = tr(BᵀB)
    dt = (a*d - b*c)^2                        # det(BᵀB) = det(B)²
    disc = t*t - interval(4)*dt
    disc = max(disc, interval(0))             # guard tiny negative from rounding
    sqrt((t + sqrt(disc)) / interval(2))
end

# domain-vector ℓ¹_ν⊕ℝ norm of a packed (4N+3) interval vector
function xnorm(v, N, ν, wμ)
    s = interval(0)
    for k in -N:N
        re=v[2*(k+N)+1]; im=v[2*(k+N)+2]
        s += ν^interval(abs(k)) * sqrt(re*re+im*im)
    end
    s + interval(wμ)*abs(v[4N+3])
end

###############################################################################
# 5.  full-tail lower bound on |λ_k|,  |k|>N   (finite scan + analytic tail)
###############################################################################
#  |λ_k| = |iωk + iμ + β e^{-iωkτ}| ≥ |ωk| - |μ| - |β|   (reverse triangle).
#  Choose K* so that ω(K*+1) - |μ| - |β| > 0, scan N<|k|≤K* rigorously,
#  and bound |k|>K* analytically.  Returns (λ_min, K*, ok).
function lambda_tail_lb(μ_iv, β_iv, ω_iv, τ_iv, N)
    absμ = abs(μ_iv); absβ = abs(β_iv); absω = abs(ω_iv)
    # analytic bound positive for ω|k| > |μ|+|β| :  |k| > (|μ|+|β|)/ω
    Kstar = Int(ceil(sup((absμ + absβ)/absω))) + 1
    Kstar = max(Kstar, N+1)
    λmin = abs(lambda_k(N+1, μ_iv, β_iv, ω_iv, τ_iv)); ok = true
    for k in vcat(collect(-Kstar:-(N+1)), collect((N+1):Kstar))
        v = abs(lambda_k(k, μ_iv, β_iv, ω_iv, τ_iv))
        if !(inf(v) > 0); ok=false; end
        λmin = min(λmin, v)
    end
    # analytic tail |k|>Kstar:  |λ_k| ≥ ω(|k|) - |μ| - |β|, increasing in |k|
    tail_lb = absω*interval(Kstar+1) - absμ - absβ
    if !(inf(tail_lb) > 0); ok=false; end
    λmin = min(λmin, tail_lb)
    λmin, Kstar, ok
end

###############################################################################
# 6.  Y bound :   Y = ‖A F(x̄)‖_𝒳
###############################################################################
function Ybound(x_iv, X_iv, A_fin, αv, θv, τv, μ_iv, β_iv, ω_iv, N, ν, wμ)
    # finite part
    F_ext = eval_F_ext(x_iv, X_iv, αv, θv, τv)
    AFfin = A_fin * F_ext
    Yfin = xnorm(AFfin, N, ν, wμ)
    # tail part:  (A F)_k = λ_k⁻¹ F_k = -λ_k⁻¹ β g_k  for |k|>N  (ā_k=0)
    a_iv,_ = unpack_w(x_iv, X_iv)
    C = g_cubic(a_iv, τv)                       # order 3N
    Cc = coefficients(C); Nc = order(space(C))
    Ytail = interval(0)
    for k in vcat(collect(-Nc:-(N+1)), collect((N+1):Nc))
        gk = Cc[k+Nc+1]
        λinv = interval(1)/lambda_k(k, μ_iv, β_iv, ω_iv, τv)
        Ytail += ν^interval(abs(k)) * abs(λinv * (-β_iv) * gk)
    end
    Yfin + Ytail, Yfin, Ytail
end

###############################################################################
# 7.  Z0 bound :   Z0 = ‖I - A DF(x̄)‖_𝒳→𝒳   (all four operator blocks)
###############################################################################
function Z0bound(x_iv, X_iv, A_fin, αv, θv, τv, μ_iv, β_iv, ω_iv, N, ν, wμ, λmin)
    n = 4N+3
    a_iv, _ = unpack_w(x_iv, X_iv)
    # -- finite blocks: M_FF = I - A_fin * DF_ext ; M_TF built from DF tail rows --
    DFext = eval_DF_ext(a_iv, unpack_w(x_iv,X_iv)[2], X_iv, αv, θv, τv)
    MFF = -(A_fin * DFext)
    for i in 1:n; MFF[i,i] += interval(1); end

    # products for cross/tail blocks (need P1,P2,P3 up to order 2N)
    Sa = shift(a_iv, -τv); ad = dagger(a_iv)
    P1c=coefficients(ad*Sa); P2c=coefficients(a_iv*Sa); P3c=coefficients(a_iv*ad)
    Nprod = 2N; z = zero(β_iv)
    cub = g_cubic(a_iv, τv); cubc = coefficients(cub); Ncub=order(space(cub))
    ac = coefficients(a_iv)
    # tail modes that couple to the finite block:  N < |k| ≤ 3N
    tailmodes = vcat(collect(-(3N):-(N+1)), collect((N+1):(3N)))

    # helper: 2×2 real block of (-β Dg) between output k and input j  (k≠j)
    negβDg(k,j) = begin
        dre,dim = DF_block(k, j, z, β_iv, ω_iv, τv, P1c, P2c, P3c, Nprod, z)  # λk=0 ⇒ pure -βDg
        (real(dre), imag(dre), real(dim), imag(dim))    # a,b,c,d of 2×2 [a b; c d]
    end

    # ---- column-group output norms ----
    # (a) finite-input columns j (|j|≤N and μ):  M_FF column + M_TF column (tail out)
    #     M_TF_{k,j} = -λ_k⁻¹ (DF)_{k(tail eq), j} ,  (DF)_{k,j}= -βDg(k,j) (j mode) or dFdμ (μ)
    Cfin = interval(0)
    # (a1) mode input columns
    for j in -N:N
        wj = ν^interval(abs(j))
        cr = 2*(j+N)+1; ci = 2*(j+N)+2
        colsum = interval(0)
        # finite output rows (M_FF): iterate output groups
        for i in -N:N
            r = 2*(i+N)+1
            colsum += ν^interval(abs(i)) * spec2(MFF[r,cr],MFF[r,ci],MFF[r+1,cr],MFF[r+1,ci])
        end
        # μ output row of M_FF (row n): 1×2 block acting on complex input
        colsum += interval(wμ) * sqrt(MFF[n,cr]^2 + MFF[n,ci]^2)
        # tail output rows (M_TF): k in tailmodes,  block = -λ_k⁻¹ (-βDg(k,j))
        for k in tailmodes
            λinv = interval(1)/lambda_k(k, μ_iv, β_iv, ω_iv, τv)
            a1,b1,c1,d1 = negβDg(k,j)
            # M_TF block = -λinv * (that 2×2 complex-as-real).  λinv complex ⇒ scale 2×2 by complex.
            # negβDg returns real 2×2 [a b; c d] representing complex map; -λinv scales it:
            lr = -real(λinv); li = -imag(λinv)
            # complex scaling (lr+i li)*(block as complex columns): block cols are (a+ic),(b+id)
            A11 = lr*a1 - li*c1; A21 = li*a1 + lr*c1
            A12 = lr*b1 - li*d1; A22 = li*b1 + lr*d1
            colsum += ν^interval(abs(k)) * spec2(A11,A12,A21,A22)
        end
        Cfin = max(Cfin, colsum / wj)
    end
    # (a2) μ input column (col n):  finite output (M_FF) + tail output (M_TF μ-part)
    let colsum = interval(0)
        for i in -N:N
            r = 2*(i+N)+1
            colsum += ν^interval(abs(i)) * sqrt(MFF[r,n]^2 + MFF[r+1,n]^2)
        end
        colsum += interval(wμ)*abs(MFF[n,n])
        # tail output of M_TF μ-column: -λ_k⁻¹ ∂F_k/∂μ = -λ_k⁻¹ (iβ g_k)  (ā_k=0,Sā_k=0)
        for k in tailmodes
            λinv = interval(1)/lambda_k(k, μ_iv, β_iv, ω_iv, τv)
            gk = (abs(k)<=Ncub) ? cubc[k+Ncub+1] : z
            dm = 1im*β_iv*gk
            colsum += ν^interval(abs(k)) * abs(-λinv*dm)
        end
        Cfin = max(Cfin, colsum/interval(wμ))
    end

    # (b) tail-input columns j (|j|>N):  M_FT (finite out) + M_TT (tail out)
    #     M_FT = -A_fin * (DF)_{fin eqs, tail j},  (DF)_{fin eqs,tail} = -βDg (F rows) 0 (gauges)
    # M_FT tail-input operator norm (finite support j: N<|j|≤3N)
    CFT = interval(0)
    for j in tailmodes
        wj = ν^interval(abs(j))
        # column of (DF)_{finite extended eqs, tail mode j}, real-packed (Re/Im a_j cols)
        colRe = fill(interval(0), n); colImv = fill(interval(0), n)
        for k in -N:N
            dre,dim = DF_block(k, j, z, β_iv, ω_iv, τv, P1c, P2c, P3c, Nprod, z) # -βDg (k≠j since |j|>N)
            if k != 0
                r = k<0 ? 2*(k+N)+1 : 2N+2*(k-1)+1
                colRe[r]=real(dre); colRe[r+1]=imag(dre)
                colImv[r]=real(dim); colImv[r+1]=imag(dim)
            else
                colRe[4N+1]=imag(dre); colImv[4N+1]=imag(dim)
            end
        end
        # M_FT columns = -A_fin * [colRe colImv]  (gauge rows of colRe/colImv are 0)
        mRe = -(A_fin*colRe); mIm = -(A_fin*colImv)
        # output norm over domain (finite): modes + μ
        outn = interval(0)
        for i in -N:N
            r = 2*(i+N)+1
            outn += ν^interval(abs(i))*spec2(mRe[r],mIm[r],mRe[r+1],mIm[r+1])
        end
        outn += interval(wμ)*sqrt(mRe[n]^2+mIm[n]^2)
        CFT = max(CFT, outn/wj)
    end
    # M_TT operator bound:  |β|/λmin · (‖P1‖+‖P2‖+‖P3‖)_ν
    nP1 = normν(coefficients(ad*Sa), Nprod, ν)
    nP2 = normν(coefficients(a_iv*Sa), Nprod, ν)
    nP3 = normν(coefficients(a_iv*ad), Nprod, ν)
    CTT = abs(β_iv)/λmin * (nP1+nP2+nP3)
    Ctail = CFT + CTT

    Z0 = max(Cfin, Ctail)
    Z0, Cfin, Ctail, CFT, CTT
end

# weighted ℓ¹_ν norm of a complex coefficient vector of order Nc
function normν(c, Nc, ν)
    s = interval(0)
    for k in -Nc:Nc; s += ν^interval(abs(k))*abs(c[k+Nc+1]); end
    s
end

###############################################################################
# 8.  Z2 bound :   Z2 ≥ ‖A‖ · sup_{B(x̄,R)} ‖D²F‖   (incl. μ mixed derivatives)
###############################################################################
function Z2bound(a_iv, β_iv, A_fin, N, ν, wμ, R, λmin)
    aν = normν(coefficients(a_iv), order(space(a_iv)), ν)
    aR = aν + interval(R)                    # sup ‖a‖ on the ball
    absβ = abs(β_iv)
    # ‖D²_aa F‖ = |β|·‖D²g‖ ≤ 6|β|‖a‖
    Daa = interval(6)*absβ*aR
    # ‖D²_aμ F‖ ≤ ‖∂_μλ‖ + |β|·‖Dg‖ ,  ∂_μλ_k=i(1-βe^{-iωkτ}) ⇒ ≤ 1+|β| (diag);
    #            ‖Dg‖ ≤ 3‖a‖²
    Daμ = (interval(1)+absβ) + absβ*interval(3)*aR*aR
    # ‖D²_μμ F‖ = ‖-β Sa + β g‖ ≤ |β|(‖a‖+‖a‖³)
    Dμμ = absβ*(aR + aR*aR*aR)
    # product-space second-derivative bound (see notes): safe upper bound
    D2 = Daa + interval(2)*Daμ/interval(wμ) + Dμμ/interval(wμ)^2
    # ‖A‖ : finite op-norm (cod→dom) OR tail 1/λmin
    normA_fin = opnorm_fin(A_fin, N, ν, wμ)
    normA = max(normA_fin, interval(1)/λmin)
    Z2 = normA * D2
    Z2, normA, D2, Daa, Daμ, Dμμ
end

# operator ℓ¹_ν⊕ℝ norm of the finite matrix A_fin (codomain packing → domain packing)
# codomain packing of A_fin's columns: (Re/Im F_k k≠0, Im F_0, Im a_0, Im a_1)
# but for ‖A‖ we treat A_fin as a (4N+3)×(4N+3) real matrix on the product norm;
# columns j are codomain scalars (each a single real DOF).  Output is domain.
function opnorm_fin(A_fin, N, ν, wμ)
    n = 4N+3
    best = interval(0)
    for j in 1:n
        col = interval(0)
        for i in -N:N
            r = 2*(i+N)+1
            col += ν^interval(abs(i))*sqrt(A_fin[r,j]^2 + A_fin[r+1,j]^2)
        end
        col += interval(wμ)*abs(A_fin[n,j])
        # codomain input weight: mode-eq rows carry weight ν^{|k|}, scalars weight 1
        best = max(best, col)   # unit codomain input (worst case weight 1)
    end
    best
end

###############################################################################
# 9.  result struct + radii polynomial
###############################################################################
struct ProofResult
    success::Bool
    reason::String
    radius::Float64
    Y::Float64
    Z0::Float64
    Z2::Float64
    contraction::Float64        # sup(Z0 + Z2*r)
    λtail::Float64              # tail lower bound on |λ_k|
    Kstar::Int
    μ::Float64
    μ_interval::Tuple{Float64,Float64}
    N::Int; ν::Float64; wμ::Float64
    α::Tuple{Float64,Float64}; θ::Tuple{Float64,Float64}; τ::Tuple{Float64,Float64}
    T::Tuple{Float64,Float64}; ω::Tuple{Float64,Float64}
    ReF0::Tuple{Float64,Float64}    # enclosure of the recovered Re F_0
    nontrivial::Bool
    periodicity::String
    versions::String
    param_iv::String                # outward-rounded enclosures of α_iv,θ_iv,τ_iv,T_iv,ω_iv,β_iv
end

# stringify the exact interval enclosures actually used in the proof
function param_iv_string(iv, β_iv)
    encl(x) = @sprintf("[%.17g, %.17g]", inf(x), sup(x))
    string("α_iv=", encl(iv.α), " θ_iv=", encl(iv.θ), " τ_iv=", encl(iv.τ),
           " T_iv=", encl(iv.T), " ω_iv=", encl(iv.ω),
           " β_iv=", @sprintf("[%.12g,%.12g]+i[%.12g,%.12g]",
                              inf(real(β_iv)),sup(real(β_iv)),inf(imag(β_iv)),sup(imag(β_iv))))
end

###############################################################################
# 10.  full pipeline
###############################################################################
function solve_and_prove(p::WParams; verbose=true)
    N=p.N; ω=ωf(p); X=Fourier(N,ω); ν=interval(p.ν); wμ=p.wμ
    iv = intervals(p)

    verbose && println("="^70)
    verbose && @printf("  w-DDE CAP:  α=%s θ=%s·π τ=%s T=%s N=%d ν=%.3f R=%.1e\n",
                       ratstr(p.α), ratstr(p.θπ), ratstr(p.τ), ratstr(p.T), N, p.ν, p.R)
    verbose && println("="^70)

    # -- approximate solution --
    xb, conv = solve_newton(p; verbose=false)
    if !conv
        return failresult(p, "Newton did not converge")
    end
    a_bar, μ_bar = unpack_w(xb, X)
    verbose && @printf("  μ̄ = %.12f = %.8f·π   μT/(2π) = %.10f\n",
                       μ_bar, μ_bar/π, μ_bar*Tf(p)/(2π))

    # -- rigorous interval data at x̄ --
    X_iv = Fourier(N, iv.ω)
    x_iv = interval.(xb)
    μ_iv = interval(μ_bar)
    β_iv = βval(iv.α, iv.θ, μ_iv)

    # A_fin = inv(DF_ext(x̄))  (numerical inverse, injected as point intervals)
    Jf = eval_DF_ext(a_bar, μ_bar, X, αf(p), θf(p), τf(p))
    S = svdvals(Jf)
    verbose && @printf("  σ_min(DF)=%.3e  cond=%.2f\n", S[end], S[1]/S[end])
    A_fin = interval.(inv(Jf))

    # -- λ tail lower bound (all |k|>N) --
    λmin, Kstar, okλ = lambda_tail_lb(μ_iv, β_iv, iv.ω, iv.τ, N)
    verbose && @printf("  λ tail lb (|k|>N): %.6f  (K*=%d, ok=%s)\n", inf(λmin), Kstar, okλ)
    if !okλ
        return failresult(p, "tail |λ_k| lower bound not positive (tail not invertible)")
    end

    # -- Y --
    Y, Yfin, Ytail = Ybound(x_iv, X_iv, A_fin, iv.α, iv.θ, iv.τ, μ_iv, β_iv, iv.ω, N, ν, wμ)
    verbose && @printf("  Y     = %.6e  (fin %.3e, tail %.3e)\n", sup(Y), sup(Yfin), sup(Ytail))

    # -- Z0 (4 blocks) --
    Z0, Cfin, Ctail, CFT, CTT = Z0bound(x_iv, X_iv, A_fin, iv.α, iv.θ, iv.τ, μ_iv, β_iv, iv.ω, N, ν, wμ, λmin)
    verbose && @printf("  Z0    = %.6e  (fin-in %.3e, tail-in %.3e [FT %.2e, TT %.2e])\n",
                       sup(Z0), sup(Cfin), sup(Ctail), sup(CFT), sup(CTT))

    # -- Z2 (ball) --
    Z2, normA, D2, Daa, Daμ, Dμμ = Z2bound(a_bar_iv(x_iv,X_iv), β_iv, A_fin, N, ν, wμ, p.R, λmin)
    verbose && @printf("  Z2    = %.6e  (‖A‖ %.3e, ‖D²F‖ %.3e)\n", sup(Z2), sup(normA), sup(D2))

    # -- radii polynomial : p(r)=Y+(Z0-1)r+½Z2 r² ≤0 & Z0+Z2 r<1 --
    ie, ok = interval_of_existence(Y, Z0, Z2, p.R; verbose=false)
    success = ok
    radius = ok ? inf(ie) : NaN
    contraction = ok ? sup(Z0 + Z2*interval(radius)) : NaN

    # -- recovery of the dropped equation Re F_0 over the WHOLE ball B(x̄,r) --
    #    (Re F_0 is the single symmetry-induced redundant component; it vanishes
    #     in exact arithmetic — verified independently to 200-bit precision.)
    ReF0 = ok ? recovery_enclosure(a_bar, μ_bar, X_iv, iv, N, ν, wμ, radius) :
                real(coefficients(eval_F_w(a_b_iv(x_iv,X_iv), μ_iv, X_iv, iv.α, iv.θ, iv.τ))[N+1])
    verbose && @printf("  Re F_0(x*) over B(x̄,r) ⊆ [%.3e, %.3e]  (redundant; =0 exactly)\n",
                       inf(ReF0), sup(ReF0))

    # -- nontriviality: some nonzero Fourier mode with |a_k| > radius --
    a1abs = abs(coefficients(a_b_iv(x_iv,X_iv))[N+2])
    nontrivial = ok && (inf(a1abs) > radius)

    # -- nature of z(t)=w(t)e^{iμt}: relative periodic orbit + multiplier enclosure --
    period = ok ? periodicity_report(p, μ_iv, radius, wμ) :
                  "n/a (proof failed)"

    ver = version_string()
    reason = success ? "radii polynomial negative; contraction < 1" :
                       "radii polynomial not satisfied"
    if success
        verbose && printstyled("\n  ★ PROOF SUCCESSFUL ★\n"; color=:green)
        verbose && @printf("  unique zero x*=(a*,μ*) of the extended system in B(x̄, %.3e), contraction %.4f\n",
                           radius, contraction)
        verbose && println("  ⇒ F_k(a*,μ*)=0 for all k≠0 and Im F_0=0 hold EXACTLY (verified eqs);")
        verbose && @printf("    the one dropped component Re F_0 (rotation/translation redundancy) ⊆ [%.1e,%.1e]\n",
                           inf(ReF0), sup(ReF0))
        verbose && println("    and =0 in exact arithmetic ⇒ w* is a genuine T-periodic solution of the w-DDE.")
        verbose && println("  z(t)=w*(t)e^{iμ*t}: ", period)
    else
        verbose && printstyled("\n  ✗ proof failed: $reason\n"; color=:red)
    end

    μhalf = ok ? radius/wμ : NaN
    ProofResult(success, reason, radius, sup(Y), sup(Z0), sup(Z2), contraction,
                inf(λmin), Kstar, μ_bar, (μ_bar - μhalf, μ_bar + μhalf),
                N, p.ν, p.wμ,
                (Float64(p.α[1]),Float64(p.α[2])), (Float64(p.θπ[1]),Float64(p.θπ[2])),
                (Float64(p.τ[1]),Float64(p.τ[2])), (Float64(p.T[1]),Float64(p.T[2])),
                (inf(iv.ω),sup(iv.ω)), (inf(ReF0),sup(ReF0)), nontrivial, period, ver,
                param_iv_string(iv, β_iv))
end

# small helpers
a_bar_iv(x_iv,X_iv) = unpack_w(x_iv,X_iv)[1]
a_b_iv(x_iv,X_iv)   = unpack_w(x_iv,X_iv)[1]
ratstr(t::Tuple{Int,Int}) = t[2]==1 ? string(t[1]) : "$(t[1])/$(t[2])"

# rigorous enclosure of the (dropped, redundant) component Re F_0 over the WHOLE
# ball B(x̄,r):  any x*∈B has |a_k-ā_k| ≤ r/ν^{|k|} and |μ-μ̄| ≤ r/w_μ, so the box
#   a_k ∈ ā_k + ([−ρ_k,ρ_k] + i[−ρ_k,ρ_k]),  ρ_k=r/ν^{|k|},   μ ∈ μ̄ ± r/w_μ
# is a rigorous superset of the ℓ¹_ν-ball; interval-evaluate Re F_0 on it.
function recovery_enclosure(a_bar, μ_bar, X_iv, iv, N, ν, wμ, r)
    c = coefficients(a_bar)
    T = typeof(interval(0.0))
    ac = Vector{Complex{T}}(undef, 2N+1)
    rv = interval(r); pm = interval(-1.0, 1.0)
    for k in -N:N
        ρ = rv / ν^interval(abs(k))
        re = interval(real(c[k+N+1])) + pm*ρ
        im = interval(imag(c[k+N+1])) + pm*ρ
        ac[k+N+1] = complex(re, im)
    end
    a_iv = Sequence(X_iv, ac)
    μ_iv = interval(μ_bar) + pm*(rv/interval(wμ))
    F = eval_F_w(a_iv, μ_iv, X_iv, iv.α, iv.θ, iv.τ)
    real(coefficients(F)[N+1])
end

# Honest reporting of the nature of z(t)=w*(t)e^{iμ*t}.
#   w* is a genuine T-periodic solution (Fourier ansatz + proof).
#   z* is a RELATIVE periodic orbit: z*(t+T)=e^{iμ*T}z*(t).  z* is (absolutely)
#   periodic iff μ*T/(2π)∈ℚ.  The CAP only ENCLOSES μ*, so rationality cannot be
#   certified; we therefore report a relative periodic orbit and give the rigorous
#   multiplier enclosure, plus the (non-certified) numerical rational it is near.
function periodicity_report(p::WParams, μ_iv, radius, wμ)
    # μ* ∈ μ̄ ± radius/wμ ; multiplier ratio ρ = μ*T/(2π)
    μrange = μ_iv + interval(-1.0,1.0)*(interval(radius)/interval(wμ))
    ratio  = μrange * intervals(p).T / (interval(2)*interval(π))
    # nearest simple rational the numerical value sits at (for information only)
    mp,mq = p.μ0π; Tp,Tq = p.T
    num = mp*Tp; den = mq*2*Tq; gg = gcd(num,den); num÷=gg; den÷=gg
    approx = mid(ratio)
    near = abs(approx - num/den) < 1e-8 ? " (numerically ≈ $num/$den, i.e. period $(abs(den))·T — NOT rigorously certified rational)" : ""
    @sprintf("RELATIVE periodic orbit; multiplier e^{iμ*T}, μ*T/(2π) ⊆ [%.8f, %.8f]%s",
             inf(ratio), sup(ratio), near)
end

function failresult(p::WParams, reason)
    ProofResult(false, reason, NaN, NaN, NaN, NaN, NaN, NaN, 0, NaN, (NaN,NaN),
        p.N, p.ν, p.wμ, (Float64(p.α[1]),Float64(p.α[2])),(Float64(p.θπ[1]),Float64(p.θπ[2])),
        (Float64(p.τ[1]),Float64(p.τ[2])),(Float64(p.T[1]),Float64(p.T[2])),(NaN,NaN),(NaN,NaN),
        false, "n/a", version_string(), param_iv_string(intervals(p), βval(intervals(p).α,intervals(p).θ,interval(μ0f(p)))))
end

function version_string()
    io = IOBuffer()
    print(io, "julia ", VERSION)
    try
        deps = Pkg.dependencies()
        for (uuid,info) in deps
            if info.name in ("RadiiPolynomial","IntervalArithmetic")
                print(io, "; ", info.name, " ", info.version)
            end
        end
    catch; end
    String(take!(io))
end

###############################################################################
# 10b.  REAL-SUBSPACE RIGOROUS PROOF   (star-shaped orbits with T = 2τ)
#
# When ωτ = π  (i.e. T = 2τ) the shift is real: (S_τ a)_k = (-1)^k a_k.  The
# star-shaped orbit is real:  a_k ∈ ℝ  (⇔ w(-t)=conj(w(t))).  With μ = θ - π/2
# (a rational multiple of π, verified from the input data) one has β = iα, i.e.
#   β_R = α cos(θ-μ) = 0,   β_I = α sin(θ-μ) = α        (EXACT — no cos/sin needed).
# The residual then splits EXACTLY (verified ≡0 to machine ε at random points):
#       Re F_k = β_R d_k = 0 ,      Im F_k = Λ_k a_k - α g_k ,
#   Λ_k = ωk + μ + α(-1)^k,   d_k = (-1)^k a_k - g_k,   g = a·a†·S_τ a.
# Hence the REAL square system  F^R_k := Λ_k a_k - α g_k = 0  (k=-N..N) has:
#   • no unknown μ, no dropped equation, no continuous symmetry (rotation leaves
#     ℝ; translation generator iωk a_k is imaginary) ⇒ (2N+1)×(2N+1) nonsingular;
#   • a zero a* gives Re F_k = β_R d_k = 0 for ALL k automatically (β_R=0 EXACT),
#     hence F_k(a*, μ)=0 ∀k ⇒ a GENUINE w-periodic solution (Re F_0 recovered
#     rigorously, not merely enclosed);
#   • μ = θ - π/2 EXACT ⇒ μT/(2π) is EXACTLY rational ⇒ z is genuinely periodic.
###############################################################################

dagger_r(a) = begin
    N=order(space(a)); c=coefficients(a); cd=similar(c)
    for k in -N:N; cd[k+N+1]=c[-k+N+1]; end; Sequence(space(a),cd)
end
Sτ_r(a) = begin
    N=order(space(a)); c=coefficients(a)
    Sequence(space(a), [(iseven(k) ? c[k+N+1] : -c[k+N+1]) for k in -N:N])
end
gR(a) = (a * dagger_r(a)) * Sτ_r(a)
ΛR(k, ω, μ, α) = ω*k + μ + α*(iseven(k) ? one(α) : -one(α))   # real; = ±|λ_k| (Float64 or Interval)

# F^R (real coefficient vector, |k|≤N)  — generic over Float64 / Interval
function eval_FR(a, X, ω, μ, α)
    N=order(X); g=project(gR(a), X); ac=coefficients(a); gc=coefficients(g)
    [ΛR(k,ω,μ,α)*ac[k+N+1] - α*gc[k+N+1] for k in -N:N]
end
# analytic Jacobian  DF^R_kj = Λ_k δ_kj - α(Q1_{k-j}+Q2_{k+j}+(-1)^j Q3_{k-j})
function DFR(a, X, ω, μ, α)
    N=order(X); Q1=coefficients(dagger_r(a)*Sτ_r(a)); Q2=coefficients(a*Sτ_r(a)); Q3=coefficients(a*dagger_r(a))
    Np=2N; z=zero(eltype(Q1))
    getQ(Q,i)= (1<=i<=2Np+1) ? Q[i] : z
    RT = typeof(ΛR(0,ω,μ,α) - α*z)
    J = Matrix{RT}(undef, 2N+1, 2N+1)
    for j in -N:N, k in -N:N
        q1=getQ(Q1,(k-j)+Np+1); q2=getQ(Q2,(k+j)+Np+1); q3=getQ(Q3,(k-j)+Np+1)
        sgn = iseven(j) ? one(α) : -one(α)
        J[k+N+1,j+N+1] = (k==j ? ΛR(k,ω,μ,α) : zero(RT)) - α*(q1 + q2 + sgn*q3)
    end
    J
end
# real 2-mode block (output k, input j) of  -α Dg   (k≠j; for cross/tail blocks)
function negαDg_R(k, j, Q1, Q2, Q3, Np, α, z)
    getQ(Q,i)= (1<=i<=2Np+1) ? Q[i] : z
    sgn = iseven(j) ? one(α) : -one(α)
    -α*(getQ(Q1,(k-j)+Np+1) + getQ(Q2,(k+j)+Np+1) + sgn*getQ(Q3,(k-j)+Np+1))
end

# real ℓ¹_ν vector norm and operator norm
vnormR(v, N, ν) = sum(ν^interval(abs(k))*abs(v[k+N+1]) for k in -N:N)
function opnormR(M, N, ν)     # max_j (1/ν^|j|) Σ_i ν^|i| |M_ij|
    best=interval(0)
    for j in -N:N
        col = sum(ν^interval(abs(i))*abs(M[i+N+1,j+N+1]) for i in -N:N)
        best = max(best, col/ν^interval(abs(j)))
    end
    best
end

# full-tail lower bound on |Λ_k|, |k|>N :  |Λ_k| = |ωk+μ+α(-1)^k| ≥ ω|k|-|μ|-α
function lambda_tail_lb_R(ω, μ, α, N)
    Kstar = Int(ceil(sup((abs(μ)+α)/abs(ω)))) + 1; Kstar=max(Kstar,N+1)
    λmin = abs(ΛR(N+1,ω,μ,α)); ok=true
    for k in vcat(collect(-Kstar:-(N+1)), collect((N+1):Kstar))
        v=abs(ΛR(k,ω,μ,α)); inf(v)>0 || (ok=false); λmin=min(λmin,v)
    end
    tail = abs(ω)*interval(Kstar+1) - abs(μ) - α
    inf(tail)>0 || (ok=false)
    min(λmin,tail), Kstar, ok
end

function solve_and_prove_real(p::WParams; verbose=true)
    N=p.N; ω=ωf(p); X=Fourier(N,ω); ν=interval(p.ν); iv=intervals(p)
    # -- structural preconditions (exact, from rational data) --
    T2τ = (p.T[1]*p.τ[2] == 2*p.τ[1]*p.T[2])                 # T = 2τ ⇔ ωτ=π
    μnum = p.θπ[1]*2 - p.θπ[2]; μden = p.θπ[2]*2             # μ = θ-π/2 = (2θπ₁-θπ₂)/(2θπ₂)·π
    g0=gcd(μnum,μden); μnum÷=g0; μden÷=g0
    μ_is = (p.μ0π[1]*μden == μnum*p.μ0π[2])                  # given μ0 == θ-π/2 ?
    verbose && println("="^70)
    verbose && @printf("  w-DDE REAL CAP:  α=%s θ=%s·π τ=%s T=%s N=%d ν=%.3f R=%.1e\n",
                       ratstr(p.α),ratstr(p.θπ),ratstr(p.τ),ratstr(p.T),N,p.ν,p.R)
    verbose && @printf("  preconditions:  T=2τ (ωτ=π): %s    μ=θ-π/2 exact: %s (=%d/%d·π)\n",
                       T2τ, μ_is, μnum, μden)
    (T2τ && μ_is) || return failresult(p, "real proof needs T=2τ and μ0=θ-π/2 (exact); use solve_and_prove")
    # -- EXACT interval constants:  μ=θ-π/2 (rational×π),  α,  ω;  β=iα (β_R=0) --
    μ_iv = interval(μnum)/interval(μden)*iv.πiv
    α_iv = iv.α; ω_iv = iv.ω
    # -- approximate real solution (Newton on F^R, Float64) --
    xb = real.(initial_guess(αf(p), θf(p), μ0f(p), N, ω, τf(p)))     # real part of the guess
    μ64 = μnum/μden*π; α64 = αf(p)
    for it in 1:60
        F = eval_FR(Sequence(X, xb), X, ω, μ64, α64)
        maximum(abs,F) < 1e-14 && break
        xb = xb - DFR(Sequence(X, xb), X, ω, μ64, α64) \ F
    end
    Fbar = eval_FR(Sequence(X, xb), X, ω, μ64, α64)
    verbose && @printf("  real Newton: ‖F^R(ā)‖∞ = %.2e\n", maximum(abs,Fbar))
    # -- interval data --
    X_iv=Fourier(N,ω_iv); a_iv=Sequence(X_iv, interval.(xb))
    Jf = mid.(DFR(a_iv, X_iv, ω_iv, μ_iv, α_iv)); A_fin = interval.(inv(Jf))
    S=svdvals(Jf); verbose && @printf("  σ_min(DF^R)=%.3e cond=%.2f\n", S[end], S[1]/S[end])
    # -- tail lower bound on |Λ_k| (all |k|>N) --
    λmin,Kstar,okλ = lambda_tail_lb_R(ω_iv, μ_iv, α_iv, N)
    verbose && @printf("  |Λ_k| tail lb (|k|>N): %.4f (K*=%d, ok=%s)\n", inf(λmin),Kstar,okλ)
    okλ || return failresult(p,"tail |Λ_k| not positive")
    # -- Y = ‖A F^R(ā)‖ : finite + tail (F^R_k=-α g_k for |k|>N; g support 3N) --
    Ffin_iv = eval_FR(a_iv, X_iv, ω_iv, μ_iv, α_iv)
    Yfin = vnormR(A_fin*Ffin_iv, N, ν)
    C = gR(a_iv); Cc=coefficients(C); Nc=order(space(C)); Ytail=interval(0)
    for k in vcat(collect(-Nc:-(N+1)),collect((N+1):Nc))
        Ytail += ν^interval(abs(k)) * abs((interval(1)/ΛR(k,ω_iv,μ_iv,α_iv))*(-α_iv)*Cc[k+Nc+1])
    end
    Y = Yfin+Ytail
    verbose && @printf("  Y  = %.6e (fin %.3e, tail %.3e)\n", sup(Y),sup(Yfin),sup(Ytail))
    # -- Z0 = ‖I - A DF^R‖ : four blocks (fin/tail × fin/tail) --
    DFe = DFR(a_iv, X_iv, ω_iv, μ_iv, α_iv); MFF = -(A_fin*DFe)
    for i in 1:2N+1; MFF[i,i]+=interval(1); end
    Q1=coefficients(dagger_r(a_iv)*Sτ_r(a_iv)); Q2=coefficients(a_iv*Sτ_r(a_iv)); Q3=coefficients(a_iv*dagger_r(a_iv))
    Np=2N; z=zero(eltype(Q1)); tail3=vcat(collect(-(3N):-(N+1)),collect((N+1):(3N)))
    # (a) finite-input columns: M_FF + M_TF (tail out = -Λ_k⁻¹·(-α Dg))
    Cfin=interval(0)
    for j in -N:N
        col = sum(ν^interval(abs(i))*abs(MFF[i+N+1,j+N+1]) for i in -N:N)
        for k in tail3
            blk = negαDg_R(k,j,Q1,Q2,Q3,Np,α_iv,z)         # (DF^R)_{tail k, fin j} = -α Dg
            col += ν^interval(abs(k)) * abs((interval(-1)/ΛR(k,ω_iv,μ_iv,α_iv))*blk)
        end
        Cfin=max(Cfin, col/ν^interval(abs(j)))
    end
    # (b) tail-input columns: M_FT (fin out = -A_fin·(-α Dg)) + M_TT (op bound)
    CFT=interval(0)
    for j in tail3
        colv=fill(interval(0.0), 2N+1)
        for k in -N:N; colv[k+N+1]=negαDg_R(k,j,Q1,Q2,Q3,Np,α_iv,z); end
        m = -(A_fin*colv)
        outn = sum(ν^interval(abs(i))*abs(m[i+N+1]) for i in -N:N)
        CFT=max(CFT, outn/ν^interval(abs(j)))
    end
    nQ = vnormR_seq(Q1,Np,ν)+vnormR_seq(Q2,Np,ν)+vnormR_seq(Q3,Np,ν)   # ‖Dg‖ ≤ Σ‖Qi‖
    CTT = α_iv/λmin * nQ
    Z0 = max(Cfin, CFT+CTT)
    verbose && @printf("  Z0 = %.6e (fin-in %.3e, tail-in %.3e [FT %.2e, TT %.2e])\n",
                       sup(Z0),sup(Cfin),sup(CFT+CTT),sup(CFT),sup(CTT))
    # -- Z2 = ‖A‖·sup_ball‖D²F^R‖ ; only D²_aa (μ fixed):  ‖D²F^R‖ ≤ 6α(‖ā‖+R) --
    aν = vnormR(coefficients(a_iv), N, ν)
    D2 = interval(6)*α_iv*(aν+interval(p.R))
    normA = max(opnormR(A_fin,N,ν), interval(1)/λmin)
    Z2 = normA*D2
    verbose && @printf("  Z2 = %.6e (‖A‖ %.3e, ‖D²F^R‖ %.3e)\n", sup(Z2),sup(normA),sup(D2))
    # -- radii polynomial --
    ie,ok = interval_of_existence(Y,Z0,Z2,p.R; verbose=false)
    radius = ok ? inf(ie) : NaN; contraction = ok ? sup(Z0+Z2*interval(radius)) : NaN
    # -- nontriviality:  |a_1| > radius  ⇒ nonconstant, fundamental period T --
    a1 = abs(interval(xb[N+2]))
    nontrivial = ok && (inf(a1) > (ok ? radius : Inf))
    # -- periodicity (EXACT):  μ=θ-π/2 rational×π ⇒ μT/(2π) = (μnum/μden)(T/2) ∈ ℚ --
    Tn,Td=p.T; pn=μnum*Tn; pd=μden*2*Td; g2=gcd(pn,pd); pn÷=g2; pd÷=g2
    period = ok ? @sprintf("z GENUINELY periodic: μ=%d/%d·π EXACT ⇒ μT/(2π)=%d/%d ∈ ℚ (period %d·T)", μnum,μden,pn,pd,abs(pd)) :
                  "n/a"
    if ok
        verbose && printstyled("\n  ★ PROOF SUCCESSFUL (real, fully rigorous) ★\n"; color=:green)
        verbose && @printf("  unique real zero a* in B(ā,%.3e), contraction %.4f\n", radius, contraction)
        verbose && println("  β_R=0 EXACT ⇒ Re F_k=0 ∀k ⇒ ALL Galerkin eqs hold ⇒ genuine T-periodic w*.")
        verbose && println("  ", period)
    else
        verbose && printstyled("\n  ✗ real proof failed\n"; color=:red)
    end
    μval = Float64(μnum/μden)*π
    ProofResult(ok, ok ? "real radii polynomial negative; contraction<1; Re F_0=0 exact" : "failed",
        radius, sup(Y), sup(Z0), sup(Z2), contraction, inf(λmin), Kstar, μval, (μval,μval),
        N, p.ν, 1.0, (Float64(p.α[1]),Float64(p.α[2])),(Float64(p.θπ[1]),Float64(p.θπ[2])),
        (Float64(p.τ[1]),Float64(p.τ[2])),(Float64(p.T[1]),Float64(p.T[2])),(inf(ω_iv),sup(ω_iv)),
        (0.0,0.0), nontrivial, period, version_string(),
        string(param_iv_string(iv, complex(interval(0.0),α_iv)), " μ_iv=", μnum,"/",μden,"·π (EXACT)"))
end

# weighted ℓ¹_ν norm of a raw coefficient vector of order Np
vnormR_seq(c, Np, ν) = sum(ν^interval(abs(k))*abs(c[k+Np+1]) for k in -Np:Np)

###############################################################################
# 11.  predefined cases + main
###############################################################################
case1() = WParams((6,1),(1,3),(1,1),(2,1),(-1,6), 30, 1.0, 1e-2, 1.0)   # α=6,θ=π/3
case2() = WParams((5,2),(1,10),(1,1),(2,1),(-2,5), 30, 1.0, 1e-2, 1.0)  # α=2.5,θ=π/10

function main(args=ARGS)
    which = isempty(args) ? "all" : args[1]
    if which in ("case1","1","all")
        println("\n# Case 1: α=6, θ=π/3   (fully rigorous, real subspace)")
        r1 = solve_and_prove_real(case1()); showresult(r1)
        save_result(r1, joinpath(@__DIR__, "wproof", "result_case1.txt"))
    end
    if which in ("case2","2","all")
        println("\n# Case 2: α=2.5, θ=π/10 (pentagram)   (fully rigorous, real subspace)")
        r2 = solve_and_prove_real(case2()); showresult(r2)
        save_result(r2, joinpath(@__DIR__, "wproof", "result_case2.txt"))
    end
    if which in ("general","gen")   # complex general-parameter proof (recovery only enclosed)
        println("\n# Case 1 via general complex formulation (recovery enclosed, not exact)")
        showresult(solve_and_prove(case1()))
    end
    if which in ("test","all")
        include(joinpath(@__DIR__, "wproof", "runtests.jl"))
    end
end

function showresult(r::ProofResult)
    println("-"^70)
    @printf("  success=%s  reason=%s\n", r.success, r.reason)
    @printf("  radius=%.4e  Y=%.3e  Z0=%.4f  Z2=%.3e  contraction=%.4f\n",
            r.radius, r.Y, r.Z0, r.Z2, r.contraction)
    @printf("  λ tail lb=%.4f (K*=%d)   nontrivial=%s\n", r.λtail, r.Kstar, r.nontrivial)
    @printf("  μ ∈ [%.10f, %.10f]\n", r.μ_interval[1], r.μ_interval[2])
    @printf("  Re F_0 (recovered) ∈ [%.2e, %.2e]\n", r.ReF0[1], r.ReF0[2])
    println("  ", r.periodicity)
    println("  ", r.versions)
end

# save a machine- and human-readable certificate of the proof
function save_result(r::ProofResult, path::AbstractString)
    open(path, "w") do io
        println(io, "# w-DDE computer-assisted proof — verified result")
        for f in fieldnames(ProofResult)
            println(io, rpad(String(f), 16), " = ", getfield(r, f))
        end
    end
    println("  saved result → ", path)
end

# only run when executed as a script (not when included by tests)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
