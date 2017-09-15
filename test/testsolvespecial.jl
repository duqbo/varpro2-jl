
include("../src/levmarq.jl")

m = 100
n = 10

R = randn(m,n) + im*0.0;
triu!(R)
D = diagm(randn(n)) + im*0.0;

A = vcat(R,D)
B = copy(A)

rhs = randn(m+n)+im*0.0

xA = zeros(rhs); xB = zeros(rhs)

copy!(xA,rhs)
levmarq_solve_special!(A,xA)

xB = B\rhs

println(norm(xA[1:n]-xB))
