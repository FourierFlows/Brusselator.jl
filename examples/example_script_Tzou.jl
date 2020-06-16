using Brusselator 

using Plots, Printf

using FFTW: irfft

# ## Let's prescibe parameter values and solve the PDE
#
# We are now ready to write up a program that sets up parameter values, constructs 
# the problem `prob`, # time steps the solutions `prob.sol` and plots it.

# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide

# ## Numerical parameters and time-stepping parameters

     nx = 512            # grid resolution
stepper = "RK4"          # timestepper
     dt = 0.01           # timestep
 nsteps = 2500           # total number of time-steps
 nsubs  = 25             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

E = 1.4

L = 137.37     # Domain length

# Parameters used by Tzou et al. (2013)
ε = 0.1
μ = 25
ρ = 0.178

D_c = ((sqrt(1 + E^2) - 1) / E)^2
B_H = (1 + E * sqrt(D_c))^2 

B = B_H + ε^2 * μ
D = D_c + ε^2 * ρ
   
kc = sqrt(E / sqrt(D))

nothing # hide


# ## Construct the `struct`s and you are ready to go!
# Create a `grid` and also `params`, `vars`, and the `equation` structs. Then 
# give them all as input to the `FourierFlows.Problem()` constructor to get a
# problem struct, `prob`, that contains all of the above.

grid = OneDGrid(dev, nx, L)
params = Brusselator.Params(B, D, E)
vars = Brusselator.Vars(dev, grid)
equation = Brusselator.Equation(dev, params, grid)

prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
nothing #hide

get_u(prob) = irfft(prob.sol[:, 1], prob.grid.nx)
get_v(prob) = irfft(prob.sol[:, 2], prob.grid.nx)

u_solution = Diagnostic(get_u, prob; nsteps=nsteps+1)
diags = [u_solution]

# ## Visualizing the simulation

# We define a function that plots $u$ and $v$. 

function plot_output(prob)
  plot_u = plot(grid.x, vars.u,
                 color = :blue,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-L/2, L/2),
                 ylims = (E-1.5, E+1.5),
                xlabel = "x",
                ylabel = "u")

  plot_v = plot(grid.x, vars.v,
                 color = :red,
                legend = false,
             linewidth = 2,
                 alpha = 0.7,
                 xlims = (-L/2, L/2),
                 ylims = (B/E-1.5, B/E+1.5),
                xlabel = "x",
                ylabel = "v")
  
  title = plot(title = "Brusselator", grid = false, showaxis = false, bottom_margin = -20Plots.px)
  
  return plot(title, plot_u, plot_v, 
           layout = @layout([A{0.01h}; [B; C]]),
             size = (600, 400),
              dpi = 150)
end
nothing # hide

# ## Setting initial conditions

# We reproduce here (to our best knowledge possible) the initial condition used
# by Tzou et al. (2013).

l = 28;
θ = @. (grid.x > -l/2) & (grid.x < l/2)

au, av = -(E^2 + kc^2) / B, -1
cu, cv = -E * (E + im) / B, 1

u0 = @.  E  + ε * real( au * exp(im * kc * grid.x) * θ + cu * (1 - θ) )
v0 = @. B/E + ε * real( av * exp(im * kc * grid.x) * θ + cv * (1 - θ) )

set_uv!(prob, u0, v0)

# Let's plot the initial condition.

plot_output(prob)


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time. We update variables by calling 
# `updatevars!()` and we also update the plot. We enclose the `for` loop in 
# an `@animate` macro to produce an animation of the solution.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nsubs)
  
  log = @sprintf("step: %04d, t: %.1f, walltime: %.2f min",
  prob.clock.step, prob.clock.t, (time()-startwalltime)/60)

  if j%(500/nsubs)==0; println(log) end
  
  updatevars!(prob)

  p[2][1][:y] = vars.u    # updates the plot for u
  p[2][:title] = "t = "*@sprintf("%.1f", prob.clock.t) # updates time in the title
  p[3][1][:y] = vars.v    # updates the plot for v

  stepforward!(prob, diags, nsubs)
end

mp4(anim, "brusselator.mp4", fps=18)

# Now let's plot a space-time diagram of the $u$ solution.

t = u_solution.t[1:nsteps+1]
U_xt = zeros(grid.nx, nsteps+1)
[ U_xt[:, j] = u_solution.data[j] for j=1:nsteps+1 ]

heatmap(grid.x, t, U_xt',
             c = :grayC,
         clims = (1, 3))
