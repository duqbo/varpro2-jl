
include("../src/levmarq.jl")

# model

function fun1!(f,x,xpts)
    a = x[1]; b = x[2]
    copy!(f,a*cos(b*xpts) + b*sin(a*xpts))
    return
end

function jac1!(jacmat,x,xpts)
    a = x[1]; b = x[2]
    jacmat[:,1] = cos(b*xpts) + b*xpts.*cos(a*xpts)
    jacmat[:,2] = -a*xpts.*sin(b*xpts) + sin(a*xpts)
    return
end

# set parameters and data

a = 5.0+im*0.0; b = 6.0+im*0.0; x = [a; b]
npts = 30
sigma = 1.0e-14
delta = 1.0e-1
xpts = linspace(0,2*pi,npts)
yclean = zeros(Complex{Float64},xpts)

fun1!(yclean,x,xpts)
y = yclean + sigma*randn(size(yclean))

# set up solver and call

x_init = x + delta*randn(size(x))
fun! = (f,x) -> fun1!(f,x,xpts)
jac! = (jacmat,x) -> jac1!(jacmat,x,xpts)

y_init = zeros(y)
fun!(y_init,x_init)

println("starting err : ",vecnorm(y_init-y)/vecnorm(y))

opts = LevMarqOpts(rel_tol=1.0e-16)
rslt = levmarq(x_init,y,fun!,jac!,opts)

xfound = rslt.minimizer

println("minimizer found: ", xfound)

yfound = zeros(y)
fun!(yfound,xfound)
jactemp = zeros(Complex{Float64},length(y),2)
jac!(jactemp,xfound)

println("err (params) : ",vecnorm(x-xfound)/vecnorm(x))
println("err (fit)    : ",vecnorm(yfound-y)/vecnorm(y))
