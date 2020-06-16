using Brusselator 


# ## Let's prescibe parameter values and solve the PDE
#
# We are now ready to write up a program that sets up parameter values, constructs 
# the problem `prob`, # time steps the solutions `prob.sol` and plots it.

# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide

# ## Numerical parameters and time-stepping parameters

     nx = 256            # grid resolution
stepper = "ETDRK4"      # timestepper
     dt = 0.005           # timestep
 nsteps = 5000           # total number of time-steps
nothing # hide


# ## Physical parameters

E = 1.4

L = 137.37     # Domain length
ε = 0.1
μ = 25
ρ = 0.178

D_c = ((sqrt(1 + E^2) - 1) / E)^2
B_H = (1 + E * sqrt(D_c))^2 

B = B_H + ε^2 * μ
D = D_c + ε^2 * ρ
   
kc = sqrt(E/sqrt(D))

nothing # hide


# ## Construct the `struct`s and you are ready to go!
# Create a `grid` and also `params`, `vars`, and the `equation` structs. Then 
# give them all as input to the `FourierFlows.Problem()` constructor to get a
# problem struct, `prob`, that contains all of the above.

grid = OneDGrid(dev, nx, L)
params = Params(B, D, E)
vars = Vars(dev, grid)
equation = Equation(dev, params, grid)

prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
nothing #hide

get_u(prob) = irfft(prob.sol[:, 1], prob.grid.nx)
get_v(prob) = irfft(prob.sol[:, 2], prob.grid.nx)

u = Diagnostic(get_u, prob; nsteps=nsteps)
v = Diagnostic(get_v, prob; nsteps=nsteps)
diags = [u]
# ## Setting initial conditions

# For initial condition we take the fluid at rest ($u=v=0$). The free surface elevation
# is perturbed from its rest position ($\eta=0$); the disturbance we impose a Gaussian 
# bump with half-width greater than the deformation radius and on top of that we 
# superimpose some random noise with scales smaller than the deformation radius. 
# We mask the small-scale perturbations so that it only applies in the central part 
# of the domain by applying
#
# The system develops geostrophically-balanced jets around the Gaussian bump, 
# while the smaller-scale noise propagates away as inertia-gravity waves. 

# First let's construct the Gaussian bump.
# Next the noisy perturbation. The `mask` is simply a product of $\tanh$ functions.

l = 28;
θ = @. (grid.x > -l/2) & (grid.x < l/2)


au, av = -(E^2 + kc^2) / B, -1
cu, cv = -E * (E + im) / B, 1

u0 = @.  E  + ε * real( au * exp(im * kc * grid.x) * θ + cu * (1-θ) )
v0 = @. B/E + ε * real( av * exp(im * kc * grid.x) * θ + cv * (1-θ) )

# plot_u0 = plot(grid.x, u0,
#                color = :black,
#               legend = :false,
#            linewidth = [3 2],
#                alpha = 0.7,
#                xlims = (-L/2, L/2),
#                # ylims = (-0.3, 0.3),
#               xlabel = "x",
#               ylabel = "u(x, 0)")
# 
# plot_v0 = plot(grid.x, v0,
#                color = :black,
#               legend = :false,
#            linewidth = [3 2],
#                alpha = 0.7,
#                xlims = (-L/2, L/2),
#                # ylims = (-0.3, 0.3),
#               xlabel = "x",
#               ylabel = "v(x, 0)")
# 
# 
# title = plot(title = "initial conditions", grid = false, showaxis = false, bottom_margin = -20Plots.px)
# 
# plot(title, plot_u0, plot_v0,
#            layout = @layout([A{0.01h}; [B; C]]),
#              size = (600, 400))

# Sum the Gaussian bump and the noise and then call `set_uvη!()` to set the initial condition to the problem `prob`.


# ## Visualizing the simulation

# We define a function that plots the surface elevation $\eta$ and the 
# depth-integrated velocities $u$ and $v$. 

function plot_output(prob)
  plot_u = plot(grid.x, vars.u,
                 color = :blue,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-L/2, L/2),
                 # ylims = (-0.3, 0.3),
                xlabel = "x",
                ylabel = "u")

  plot_v = plot(grid.x, vars.v,
                 color = :red,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-L/2, L/2),
                 # ylims = (-0.3, 0.3),
                xlabel = "x",
                ylabel = "v")

  
  title = plot(title = "Brusselator", grid = false, showaxis = false, bottom_margin = -20Plots.px)
  
  return plot(title, plot_u, plot_v, 
           layout = @layout([A{0.01h}; [B; C]]),
             size = (600, 400))
end
nothing # hide



set_uv!(prob, u0, v0)
plot_output(prob)


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time. We update variables by calling 
# `updatevars!()` and we also update the plot. We enclose the `for` loop in 
# an `@animate` macro to produce an animation of the solution.
 
# 
stepforward!(prob, diags, nsteps)
U = zeros(grid.nx, nsteps+1)
[ U[:, j] = u.data[j] for j=1:nsteps+1 ]

heatmap(grid.x, u.t, U', c = :grayC, clims=(1, 3))

# p = plot_output(prob)
# 
# anim = @animate for j=0:nsteps
#   updatevars!(prob)
# 
#   p[2][1][:y] = vars.u    # updates the plot for u
#   p[2][:title] = "t = "*@sprintf("%.1f", prob.clock.t) # updates time in the title
#   p[3][1][:y] = vars.v    # updates the plot for v
# 
#   stepforward!(prob)
# end
# 
# mp4(anim, "brusselator.mp4", fps=18)
