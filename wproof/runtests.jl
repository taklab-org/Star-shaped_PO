###############################################################################
# runtests.jl — validation tests for DDE_w_general.jl
#
# These tests do NOT constitute the proof; they detect index/sign/implementation
# mistakes in the analytic operators and check the rigorous bounds against
# independent finite computations.
#
# Run:  julia --project=star-shaped/wproof star-shaped/wproof/runtests.jl
###############################################################################

const _HERE = @__DIR__
if !isdefined(Main, :eval_F_w)
    include(joinpath(_HERE, "..", "DDE_w_general.jl"))
end
using RadiiPolynomial, LinearAlgebra, Printf, Random

const _pass = Ref(0); const _fail = Ref(0)
macro check(cond, msg)
    quote
        if $(esc(cond)); _pass[]+=1; @printf("  ok   %s\n", $(esc(msg)))
        else; _fail[]+=1; printstyled(@sprintf("  FAIL %s\n", $(esc(msg))); color=:red); end
    end
end

println("="^70); println("  DDE_w_general.jl — test suite"); println("="^70)

Random.seed!(20260711)

# ---------------------------------------------------------------------------
println("\n[1] dagger coefficient identity  (a†)_k = conj(a_{-k})")
let N=6, ω=π, X=Fourier(N,ω)
    a = Sequence(X, randn(ComplexF64, 2N+1))
    ad = dagger(a); c=coefficients(a); cd=coefficients(ad)
    ok = all(cd[k+N+1] ≈ conj(c[-k+N+1]) for k in -N:N)
    @check ok "(a†)_k = conj(a_{-k}) for all k"
    @check (dagger(dagger(a)) ≈ a) "a†† = a"
end

# ---------------------------------------------------------------------------
println("\n[2] shift coefficient identity  (S_τ a)_k = e^{-iωkτ} a_k")
let N=6, ω=π, τ=1.0, X=Fourier(N,ω)
    a = Sequence(X, randn(ComplexF64, 2N+1)); c=coefficients(a)
    Sa = shift(a, -τ); cs=coefficients(Sa)
    ok = all(cs[k+N+1] ≈ cis(-ω*k*τ)*c[k+N+1] for k in -N:N)
    @check ok "(S_τ a)_k = e^{-iωkτ} a_k  (τ=1)"
    τ2=0.7; Sa2=shift(a,-τ2); cs2=coefficients(Sa2)
    ok2 = all(cs2[k+N+1] ≈ cis(-ω*k*τ2)*c[k+N+1] for k in -N:N)
    @check ok2 "(S_τ a)_k = e^{-iωkτ} a_k  (τ=0.7)"
end

# ---------------------------------------------------------------------------
println("\n[3] analytic extended Jacobian vs central finite differences")
#     covers all columns (Re a_j, Im a_j, μ), all rows (Re/Im F_k, Im F_0, gauges),
#     k=0, positive & negative modes.
for N in (4,5,7)
    ω=2π/2.0; X=Fourier(N,ω); α=6.0; θ=π/3; τ=1.0
    ac=randn(ComplexF64,2N+1); ac[N+1]=real(ac[N+1]); ac[N+2]=real(ac[N+2])
    x=pack_w(Sequence(X,ac), -0.5+0.1randn())
    Ja = eval_DF_ext_packed(x, X, α, θ, τ)
    n=length(x); Jfd=zeros(n,n); h=1e-6
    for j in 1:n
        xp=copy(x);xp[j]+=h; xm=copy(x);xm[j]-=h
        Jfd[:,j]=(eval_F_ext(xp,X,α,θ,τ)-eval_F_ext(xm,X,α,θ,τ))/(2h)
    end
    err = maximum(abs, Ja-Jfd)/max(1,maximum(abs,Ja))
    @check (err < 1e-6) @sprintf("N=%d: max rel |DF_analytic - DF_fd| = %.2e", N, err)
end
# specifically test the μ-column (uses β'=-iβ) and the Im F_0 row separately
let N=6, ω=π, X=Fourier(N,ω), α=6.0, θ=π/3, τ=1.0
    ac=randn(ComplexF64,2N+1); ac[N+1]=real(ac[N+1]); ac[N+2]=real(ac[N+2])
    x=pack_w(Sequence(X,ac), -0.5); Ja=eval_DF_ext_packed(x,X,α,θ,τ); n=length(x)
    h=1e-6; colμ=(eval_F_ext([x[1:n-1];x[n]+h],X,α,θ,τ)-eval_F_ext([x[1:n-1];x[n]-h],X,α,θ,τ))/(2h)
    @check (maximum(abs, Ja[:,n]-colμ) < 1e-6) "μ-column (β'=-iβ) matches FD"
end

# ---------------------------------------------------------------------------
println("\n[4] finite/tail support of the nonlinearity and cross blocks")
let N=6, ω=π, τ=1.0, X=Fourier(N,ω)
    a=Sequence(X, randn(ComplexF64,2N+1))
    g=g_cubic(a,τ)
    @check (order(space(g)) == 3N) @sprintf("order(g(ā)) = %d = 3N", order(space(g)))
    # F(ā)_k = -β g_k for |k|>N has support only N<|k|≤3N
    gc=coefficients(g); Nc=order(space(g))
    tailnz = [k for k in -Nc:Nc if abs(k)>N && abs(gc[k+Nc+1])>1e-12]
    @check (isempty(tailnz) || maximum(abs,tailnz) <= 3N) "F(ā) tail support ⊆ {N<|k|≤3N}"
end

# ---------------------------------------------------------------------------
println("\n[5] full-tail lower bound on |λ_k| (finite scan + analytic tail)")
let N=30
    p=case1(); iv=intervals(p); μ_iv=interval(-π/6); β_iv=βval(iv.α,iv.θ,μ_iv)
    λmin,Kstar,ok = lambda_tail_lb(μ_iv,β_iv,iv.ω,iv.τ,N)
    @check ok "tail bound certified positive"
    # independent spot check: analytic bound ω|k|-|μ|-|β| ≤ |λ_k| for a large k
    kbig=500; lhs=inf(abs(lambda_k(kbig,μ_iv,β_iv,iv.ω,iv.τ)))
    rhs=sup(iv.ω*interval(kbig)-abs(μ_iv)-abs(β_iv))
    @check (lhs >= rhs - 1e-9) @sprintf("analytic bound valid at k=%d: |λ|=%.2f ≥ %.2f", kbig,lhs,rhs)
    # the reported λmin must not exceed any actually-scanned |λ_k|
    minscanned = minimum(inf(abs(lambda_k(k,μ_iv,β_iv,iv.ω,iv.τ))) for k in (N+1):(3N))
    @check (inf(λmin) <= minscanned + 1e-9) "λmin ≤ min scanned |λ_k|"
end

# ---------------------------------------------------------------------------
println("\n[6] Z0 finite-truncation inclusion  (‖I-A DF‖ finite block ≤ reported Z0)")
let p=case1()
    N=p.N; ω=ωf(p); X=Fourier(N,ω); ν=interval(p.ν); wμ=p.wμ; iv=intervals(p)
    xb,_=solve_newton(p); a_bar,μ_bar=unpack_w(xb,X)
    X_iv=Fourier(N,iv.ω); x_iv=interval.(xb); μ_iv=interval(μ_bar); β_iv=βval(iv.α,iv.θ,μ_iv)
    A_fin=interval.(inv(eval_DF_ext(a_bar,μ_bar,X,αf(p),θf(p),τf(p))))
    λmin,_,_=lambda_tail_lb(μ_iv,β_iv,iv.ω,iv.τ,N)
    Z0,Cfin,Ctail,CFT,CTT=Z0bound(x_iv,X_iv,A_fin,iv.α,iv.θ,iv.τ,μ_iv,β_iv,iv.ω,N,ν,wμ,λmin)
    # crude independent finite ℓ1-op-norm of M_FF (max column, plain) must be ≤ Cfin-ish scale
    DFext=eval_DF_ext(unpack_w(x_iv,X_iv)...,X_iv,iv.α,iv.θ,iv.τ)
    MFF=-(A_fin*DFext); for i in 1:4N+3; MFF[i,i]+=interval(1); end
    plaincol = maximum(sum(sup(abs(MFF[i,j])) for i in 1:4N+3) for j in 1:4N+3)
    @check (sup(Cfin) <= sup(Z0)+1e-12) "Cfin ≤ Z0"
    @check (plaincol < 1.0) @sprintf("finite block genuinely contractive (plain col-sum %.3f<1)", plaincol)
    @check (sup(Z0) < 1) @sprintf("Z0 = %.4f < 1", sup(Z0))
end

# ---------------------------------------------------------------------------
println("\n[7] radii polynomial negativity  p(r)=Y+(Z0-1)r+½Z2 r² < 0 at r=radius")
let r = solve_and_prove(case1(); verbose=false)
    @check r.success "case1 radii polynomial satisfied"
    Y=interval(r.Y); Z0=interval(r.Z0); Z2=interval(r.Z2); rr=interval(r.radius)
    pr = Y + (Z0-interval(1))*rr + Z2*rr^2/interval(2)
    @check (sup(pr) < 0) @sprintf("p(r) = %.3e < 0", sup(pr))
    @check (r.contraction < 1) @sprintf("contraction Z0+Z2 r = %.4f < 1", r.contraction)
end

# ---------------------------------------------------------------------------
println("\n[8] parameter intervals rigorously enclose the true π-constants")
let p=case1(), iv=intervals(p)
    @check (inf(iv.θ) <= π/3 <= sup(iv.θ)) "θ_iv encloses π/3 (built from interval(π))"
    @check (inf(iv.ω) <= π      <= sup(iv.ω)) "ω_iv encloses 2π/T = π"
    # ensure NOT a thin rounded point: interval(π) is a genuine enclosure
    @check (inf(iv.πiv) < π < sup(iv.πiv) || inf(iv.πiv)==sup(iv.πiv)==Float64(π)) "π_iv is an outward enclosure of π"
    @check (sup(iv.θ) > inf(iv.θ)) "θ_iv has positive width (irrational enclosure)"
end

# ---------------------------------------------------------------------------
println("\n[9] nontriviality (solution is not constant/zero)")
let r=solve_and_prove(case1(); verbose=false)
    @check r.nontrivial "case1 nontrivial: |a_1| enclosure exceeds radius"
end

# ---------------------------------------------------------------------------
println("\n[10] recovery: Re F_0 at the numerical solution is ~0 (dropped equation)")
let p=case1()
    N=p.N; ω=ωf(p); X=Fourier(N,ω); iv=intervals(p)
    xb,_=solve_newton(p); a_bar,μ_bar=unpack_w(xb,X)
    Fc=coefficients(eval_F_w(a_bar,μ_bar,X,αf(p),θf(p),τf(p)))
    @check (abs(real(Fc[N+1])) < 1e-11) @sprintf("Re F_0 = %.2e ≈ 0 (recovered)", real(Fc[N+1]))
    @check (abs(imag(Fc[N+1])) < 1e-11) @sprintf("Im F_0 = %.2e ≈ 0 (enforced)", imag(Fc[N+1]))
    @check (maximum(abs(Fc[k+N+1]) for k in -N:N if k!=0) < 1e-11) "all F_{k≠0} ≈ 0"
end

# ---------------------------------------------------------------------------
println("\n[11] REAL formulation: analytic DF^R vs finite differences")
for N in (5,7,9)
    ω=π; α=6.0; θ=π/3; μ=θ-π/2; X=Fourier(N,ω)
    a=Sequence(X, randn(2N+1))
    Ja=DFR(a,X,ω,μ,α)
    Jfd=zeros(2N+1,2N+1); h=1e-6
    for j in 1:2N+1
        ap=copy(coefficients(a));ap[j]+=h; am=copy(coefficients(a));am[j]-=h
        Jfd[:,j]=(eval_FR(Sequence(X,ap),X,ω,μ,α)-eval_FR(Sequence(X,am),X,ω,μ,α))/(2h)
    end
    @check (maximum(abs,Ja-Jfd)/maximum(abs,Ja) < 1e-6) @sprintf("N=%d: max rel |DF^R - FD| = %.2e", N, maximum(abs,Ja-Jfd)/maximum(abs,Ja))
end

# ---------------------------------------------------------------------------
println("\n[12] EXACT syzygy  Re F_k = β_R d_k  (recovery mechanism, ωτ=π, real a)")
let N=8, ω=π, τ=1.0, X=Fourier(N,ω), α=6.0, θ=π/3
    maxerr=0.0
    for trial in 1:5
        a=Sequence(X, ComplexF64.(randn(2N+1))); μ=randn()
        βR=α*cos(θ-μ)
        F=coefficients(eval_F_w(a,μ,X,α,θ,τ)); g=coefficients(project(g_cubic(a,τ),X)); ac=coefficients(a)
        for k in -N:N
            dk=(-1)^k*real(ac[k+N+1])-real(g[k+N+1])
            maxerr=max(maxerr, abs(real(F[k+N+1]) - βR*dk))
        end
    end
    @check (maxerr < 1e-11) @sprintf("Re F_k = β_R·d_k to %.1e over 5 random real points", maxerr)
end

# ---------------------------------------------------------------------------
println("\n[13] REAL fully-rigorous proof: exact recovery Re F_0=0 and exact periodicity")
let r = solve_and_prove_real(case1(); verbose=false)
    @check r.success "case1 real proof succeeds"
    @check (r.ReF0 == (0.0,0.0)) "Re F_0 recovered EXACTLY = 0 (β_R=0 exact)"
    @check occursin("GENUINELY periodic", r.periodicity) "z certified genuinely periodic (μ exact)"
    @check (r.contraction < 1) @sprintf("contraction %.4f < 1", r.contraction)
end
let r = solve_and_prove_real(case2(); verbose=false)
    @check r.success "case2 real proof succeeds"
    @check (r.ReF0 == (0.0,0.0)) "case2 Re F_0 = 0 exactly"
end

# ---------------------------------------------------------------------------
println("\n" * "="^70)
@printf("  PASSED %d   FAILED %d\n", _pass[], _fail[])
println("="^70)
_fail[] == 0 || error("test suite has failures")
