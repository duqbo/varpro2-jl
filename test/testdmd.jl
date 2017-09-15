
include("../src/dmd_varpro2.jl")
include("../src/varpro2.jl")
include("../src/levmarq.jl")

## generate synthetic data

# set up modes in space

x0 = 0;
x1 = 1;
nx = 1000;

# space

xspace = linspace(x0,x1,nx);

# modes

f1 = sin(xspace);
f2 = cos(xspace);
f3 = tanh(xspace);

# set up time dynamics

t0 = 0;
t1 = 1;
nt = 2000;

ts = linspace(t0,t1,nt);

# eigenvalues

e1 = 1;
e2 = -2;
e3 = im;

evals = [e1;e2;e3];

# create clean dynamics

xclean = exp(e1*ts)*f1.' + exp(e2*ts)*f2.' + exp(e3*ts)*f3.';

# add noise (just a little ... this problem is 
# actually pretty challenging)

sigma = 1e-12;
delta = 0.01
xdata = xclean + sigma*(randn(size(xclean))+im*randn(size(xclean)));

## compute modes in various ways

# target rank

r = 3;

# varpro2 names for these params

m = nt;
n = r;
is = nx;
ia = r;
lmax = 1;

pars = VarPro2Params1(xdata,r,ia,lmax)

e_init = evals+delta*(randn(size(evals))+im*randn(size(evals)))
phi! = (phimat,alpha) -> phidmd!(phimat,alpha,ts)
dphi! = (dphislab,islab,alpha,i) -> dphidmd_slab!(dphislab,islab,alpha,i,ts)

optsLM = LevMarqOpts(rel_tol=0.0,eps_stall=1.0e-12)
rslt = varpro2(pars,e_init,phi!,dphi!,opts = optsLM, iffulljac=true)

e = rslt.alpha

println("e true = ",evals)
println("e init = ",e_init)
println("e comp = ",e)
