# # 2D Brusselator
#
# 
#
# This example solves the 2D Brusselator.
# 
# ```math
# \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + E - (B+1)u + v u^2, \\
# \frac{\partial v}{\partial t} = \frac{\partial^2 v}{\partial x^2} + B u - v u^2.
# ```
#

module Brusselator

using Printf,
      Random,
      Reexport
      
@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!

export
  set_uv!,
  updatevars!,
  get_righthandside


# ## Coding up the equations
# ### A demonstration of FourierFlows.jl framework
#
# What follows is a step-by-step tutorial showing how you can create your own
# solver for an equation of your liking.

# The basic building blocks for a `FourierFlows.Problem()` are:
# - `Grid` struct containining the physical and wavenumber grid for the problem,
# - `Params` struct containining all the parameters of the problem,
# - `Vars` struct containining arrays with the variables used in the problem,
# - `Equation` struct containining the coefficients of the linear operator $\mathcal{L}$ and the function that computes the nonlinear terms, usually named `calcN!()`.
# 
# The `Grid` structure is provided by FourierFlows.jl. One simply has to call one of
# the `OneDGrid()`,  `TwoDGrid()`, or `ThreeDGrid()` grid constructors, depending
# on the dimensionality of the problem. All other structs mentioned above are problem-specific
# and need to be constructed for every set of equations we want to solve.

# First lets construct the `Params` struct that contains all parameters of the problem 

struct Params{T} <: AbstractParams
   B :: T         # B parameter
   D :: T         # D parameter
   E :: T         # E parameter
end

# Now the `Vars` struct that contains all variables used in this problem. For this
# problem `Vars` includes the represenations of the flow fields in physical space 
# `u` and `v` and their Fourier transforms `uh` and `vh`.

struct Vars{Aphys, Atrans} <: AbstractVars
     u :: Aphys
     v :: Aphys
   vu² :: Aphys
    uh :: Atrans
    vh :: Atrans
  vu²h :: Atrans
end

# A constructor populates empty arrays based on the dimension of the `grid`
# and then creates `Vars` struct.
"""
    Vars!(dev, grid)
Constructs Vars based on the dimensions of arrays of the `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T grid.nx u v vu²
  @devzeros Dev Complex{T} grid.nkr uh vh vu²h
  return Vars(u, v, vu², uh, vh, vu²h)
end

# In Fourier space, the 1D linear shallow water dynamics are:
#
# ```math
# \frac{\partial \hat{u}}{\partial t} = \underbrace{- D |\boldsymbol{k}|^2 -(B+1) }_{\mathcal{L}_u} \hat{u} + \underbrace{ E + \widehat{v u^2} }_{\mathcal{N}_u} \; , \\
# \frac{\partial \hat{v}}{\partial t} = \underbrace{- D |\boldsymbol{k}|^2 }_{\mathcal{L}_v} \hat{v} } + \underbrace{ B \hat{u} - \widehat{v u^2} }_{\mathcal{N}_v} \; . \\
# ```
#
# With these in mind, we construct function `calcN!` that computes the nonlinear terms.
#
"""
    calcN!(N, sol, t, clock, vars, params, grid)
The function that computes the nonlinear terms for our problem.
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  @. vars.uh = sol[:, 1]
  @. vars.vh = sol[:, 2]
  
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)
  
  @. vars.vu² = vars.v * vars.u^2
  
  mul!(vars.vu²h, grid.rfftplan, vars.vu²)
  
  @views @. N[:, 1] = + vars.vu²h - (params.B + 1) * sol[:, 1]    # + v*u² - (B+1) u
  N[1, 1] += params.E * grid.nx # note that fft(constant) = constant * nx
  
  @views @. N[:, 2] = - vars.vu²h + params.B * sol[:, 1]          # - v*u² + B*u
    
  return nothing
end
 
# Next we construct the `Equation` struct:

"""
    Equation!(prob)
Construct the equation: the linear part and the nonlinear part, which is 
computed by `caclN!` function.
"""
function Equation(dev, params, grid::AbstractGrid)
  T = eltype(grid)
  L = zeros(dev, T, (grid.nkr, 2))
  
  diffusion = @. - grid.kr^2
  
  @. L[:, 1] = params.D * diffusion  # D*∂²u/∂x² 
  @. L[:, 2] = diffusion             #   ∂²v/∂x²
  
  return FourierFlows.Equation(L, calcN!, grid)
end

# We now have all necessary building blocks to construct a `FourierFlows.Problem`. 
# It would be useful, however, to define some more "helper functions". For example,
# a function that updates all variables given the solution `sol` which comprises $\hat{u}$
# and $\hat{v}$:

"""
    updatevars!(prob)
Update the variables in `prob.vars` using the solution in `prob.sol`.
"""
function updatevars!(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol
  
  @. vars.uh = sol[:, 1]
  @. vars.vh = sol[:, 2]
  
  ldiv!(vars.u, grid.rfftplan, deepcopy(sol[:, 1])) # use deepcopy() because irfft destroys its input
  ldiv!(vars.v, grid.rfftplan, deepcopy(sol[:, 2])) # use deepcopy() because irfft destroys its input
  return nothing
end

# Another useful function is one that prescribes an initial condition to the state variable `sol`.

"""
    set_uv!(prob, u0, v0)
Sets the state variable `prob.sol` as the Fourier transforms of `u0` and `v0`
and update all variables in `prob.vars`.
"""
function set_uv!(prob, u0, v0)
  vars, grid, sol = prob.vars, prob.grid, prob.sol
    
  @. vars.u = u0
  @. vars.v = v0
  
  mul!(vars.uh, grid.rfftplan, vars.u)
  mul!(vars.vh, grid.rfftplan, vars.v)

  @. sol[:, 1] = vars.uh
  @. sol[:, 2] = vars.vh
    
  updatevars!(prob)
  return nothing
end

"""
    get_righthandside!(prob)
Returns the right-hand-side of the equation. In this case, the function returns
a 2-column array with the time-tendencies for uh and vh.
"""
function get_righthandside(prob)
  
  vars, params, grid, sol, clock = prob.vars, prob.params, prob.grid, prob.sol, prob.clock
  
  L = prob.eqn.L
  
  N = zeros(eltype(sol), size(sol))
  
  prob.eqn.calcN!(N, sol, clock.t, clock, vars, params, grid)
    
  return @. L*sol + N
end

end # module
