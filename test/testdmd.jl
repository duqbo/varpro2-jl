
include("../src/dmd_varpro2.jl")
include("../src/varpro2.jl")
include("../src/levmarq.jl")

srand(15)


## generate synthetic data

# set up modes in space

x0 = 0;
x1 = 1;
nx = 100;

# space

xspace = linspace(x0,x1,nx);

# modes

f1 = sin(xspace);
f2 = cos(xspace);
f3 = tanh(xspace);

# set up time dynamics

t0 = 0;
t1 = 1;
nt = 200;

ts = linspace(t0,t1,nt);

# eigenvalues

e1 = 3*im;
e2 = -2*im;
e3 = im;

evals = [e1;e2;e3];

# create clean dynamics

xclean = exp(e1*ts)*f1.' + exp(e2*ts)*f2.' + exp(e3*ts)*f3.';

# add noise (just a little ... this problem is 
# actually pretty challenging)

sigma = 0.0;
delta = 0.05
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
rslt = varpro2(pars,e_init,phi!,dphi!,opts = optsLM)

e = rslt.alpha

println("err in recovered eigenvalues ",vecnorm(e-evals)/vecnorm(evals))
