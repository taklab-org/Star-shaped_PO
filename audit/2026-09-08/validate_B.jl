using SHA
const srcpath = normpath(joinpath(@__DIR__, "..", "..", "DDE_w_general.jl"))
@assert bytes2hex(sha256(read(srcpath))) == "00fa3d252c4e58300a1aab7c3bb141edba0f072bede389b494575055ca4fd1ba"
include(srcpath)
# Audit-only explicit exact integer conversion; no change to endpoints.
ΛR(k, ω::RadiiPolynomial.IntervalArithmetic.Interval, μ, α) = ω*interval(k)+μ+α*(iseven(k) ? one(α) : -one(α))
# The extra return exposes already-computed state; original routine is also run.
src=read(srcpath,String)
body="function solve_and_prove_real_audit(" * split(split(src,"function solve_and_prove_real(")[2],"# weighted ℓ¹_ν norm of a raw coefficient vector")[1]
body=replace(body,"    ProofResult(ok, ok ?" => "    result = ProofResult(ok, ok ?")
lastend=findlast("\nend",body)
body=body[1:first(lastend)-1] * "\n    return (;result, xb, Y, Z0, Z2, aν, a_iv, normA, λmin, ie, Cfin, CFT, CTT, MFF)\nend\n"
include_string(Main,body,"DDE_w_general_audit_state")
const outdir=joinpath(@__DIR__, "certificates")
mkpath(outdir)
println("SOURCE_SHA256=",bytes2hex(sha256(src)))
println("VERSIONS=",version_string())
alphas=Dict(3=>(6,1),4=>(4,1),5=>(4,1),6=>(13,5),7=>(5,2),10=>(5,2),35=>(5,2))
tables=Dict(3=>(46//10^15,237//1000,210//1,61//10^15),4=>(31//10^15,158//1000,110//1,38//10^15),5=>(31//10^15,179//1000,70//1,39//10^15),6=>(22//10^15,77//1000,86//1,25//10^15),7=>(22//10^15,75//1000,64//1,25//10^15),10=>(17//10^15,82//1000,39//1,20//10^15),35=>(15//10^15,85//1000,25//1,17//10^15))
for n in (3,4,5,6,7,10,35)
    mun=2-n; mud=2*n; gg=gcd(mun,mud)
    p=WParams(alphas[n],(1,n),(1,1),(2,1),(mun÷gg,mud÷gg),30,1.0,1e-2,1.0)
    official=solve_and_prove_real(p;verbose=false)
    state=solve_and_prove_real_audit(p;verbose=false)
    @assert all(getfield(official,f)==getfield(state.result,f) for f in fieldnames(ProofResult))
    @assert official.success
    save_result(official,joinpath(outdir,"case_n$(n).txt"))
    open(joinpath(outdir,"coefficients_n$(n).txt"),"w") do io
        for (j,x) in enumerate(state.xb)
            println(io,j-31," ",repr(x)," ",bitstring(x))
        end
    end
    y,z,h,r=map(x -> interval(Float64,x), tables[n])
    tabp=y+(z-interval(1))*r+h*r^2/interval(2)
    tabq=z+h*r
    dominates=inf(y)>=sup(state.Y) && inf(z)>=sup(state.Z0) && inf(h)>=sup(state.Z2)
    a0=abs(interval(state.xb[31])); a1=abs(interval(state.xb[32]))
    off=sum(abs(interval(state.xb[k+31])) for k in -30:30 if k!=0)
    margin=a0-off-r
    println("CASE n=",n," official_radius=",repr(official.radius)," Y=",repr(sup(state.Y))," Z0=",repr(sup(state.Z0))," Z1=",repr(sup(state.Z2)))
    coarse=max(opnormR(state.MFF,p.N,interval(p.ν)),state.CFT)+state.CTT
    println("COARSE n=",n," sup=",repr(sup(coarse))," code_sup=",repr(sup(state.Z0))," table_ok=",sup(coarse)<=inf(z)," guaranteed=",RadiiPolynomial.IntervalArithmetic.isguaranteed(state.Y),",",RadiiPolynomial.IntervalArithmetic.isguaranteed(state.Z0),",",RadiiPolynomial.IntervalArithmetic.isguaranteed(state.Z2))
    println("TABLE n=",n," dominates=",dominates," p_sup=",repr(sup(tabp))," q_sup=",repr(sup(tabq))," radius=",tables[n][4])
    println("COEFF n=",n," a0=",a0," a1=",a1," offsum=",off," domination_margin=",margin)
    paired=sum(abs(interval(state.xb[k+31])+interval(state.xb[-k+31])) for k in 1:30)
    realmargin=interval(state.xb[31])-paired-r
    println("REALPART n=",n," lower=",repr(inf(realmargin))," paired_sum_sup=",repr(sup(paired))," a0_lower=",repr(inf(a0))," a1_lower=",repr(inf(a1)))
    @assert inf(realmargin)>0.1 && inf(a0)>0.56 && inf(a1)>0.12
    @assert dominates && sup(tabp)<0 && sup(tabq)<1
    @assert all(RadiiPolynomial.IntervalArithmetic.isguaranteed, (state.Y,state.Z0,state.Z2,state.ie,tabp,tabq,realmargin))
    println("GUARANTEE_ASSERTIONS n=",n," Y,Z0,Z1,existence_interval,table_p,table_q,realpart all=true")
end
