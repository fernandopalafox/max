"""
Differentiable MPC with Implicit Differentiation (General Horizon T)

This file implements:
  1) A unicycle dynamics model with parametric dynamics (theta2)
  2) A horizon-T MPC problem with states eliminated via forward rollout
  3) Solution of the reduced problem using L-BFGS
  4) Implicit differentiation of the optimal control sequence u*
     with respect to cost and dynamics parameters using the
     Implicit Function Theorem (IFT)

Mathematical setup:
-------------------
Dynamics:
    x_{t+1} = f_{theta2}(x_t, u_t)

Reduced MPC problem:
    min_{u_0,...,u_{T-1}} J(u; theta)
    where states are defined implicitly by rollout

Optimality condition:
    g(u, theta) = ∇_u J(u, theta) = 0
    d^2 J / d^2 u * du / dtheta + d^2 J / du dtheta = 0
    du / dtheta = - (d^2 J / d^2 u)^{-1} (d^2 J / du dtheta)

IFT sensitivity:
    du*/dtheta = - (∂g/∂u)^{-1} (∂g/∂theta)

This implementation is suitable for:
  - differentiable MPC
  - bilevel optimization
  - learning cost / dynamics parameters
  - verification against finite differences
"""

import torch
torch.set_default_dtype(torch.float64)

# ============================================================
# Dynamics model
# ============================================================

def unicycle_step(x, u, dt, theta2):
    """
    One-step discrete-time unicycle dynamics.

    State:
        x = [px, py, alpha, v]
            px, py : position
            alpha  : heading angle
            v      : forward velocity

    Control:
        u = [w, a]
            w : angular velocity
            a : acceleration

    Parameter:
        theta2 : dynamics parameter scaling acceleration

    Dynamics:
        px_{t+1}     = px_t + dt * v_t * cos(alpha_t)
        py_{t+1}     = py_t + dt * v_t * sin(alpha_t)
        alpha_{t+1}  = alpha_t + dt * w_t
        v_{t+1}      = v_t + dt * a_t * theta2
    """
    px, py, alpha, v = x
    w, a = u

    px1 = px + dt * v * torch.cos(alpha)
    py1 = py + dt * v * torch.sin(alpha)
    alpha1 = alpha + dt * w
    v1 = v + dt * a * theta2

    return torch.stack([px1, py1, alpha1, v1])


# ============================================================
# Forward rollout (state elimination)
# ============================================================

def rollout_T(x0, u_seq, dt, theta2):
    """
    Forward rollout of dynamics over horizon T.

    Inputs:
        x0     : initial state, shape (n_x,)
        u_seq  : control sequence, shape (T, n_u)
        dt     : time step
        theta2 : dynamics parameter

    Returns:
        xs : state trajectory, shape (T+1, n_x)
             xs[0] = x0
             xs[t+1] = f(xs[t], u_seq[t])

    Purpose:
        Eliminates state variables so the MPC problem
        can be solved over controls only.
    """
    T = u_seq.shape[0]
    xs = [x0]
    x = x0
    for t in range(T):
        x = unicycle_step(x, u_seq[t], dt, theta2)
        xs.append(x)
    return torch.stack(xs, dim=0)


# ============================================================
# Cost function (reduced objective)
# ============================================================

def cost_T(x0, u_seq, dt, theta1, theta2, p_target,
           u_weight=1.0, terminal_only=False):
    """
    Reduced horizon-T cost J(u; theta).

    Parameters:
        theta1 : tracking weight (cost parameter)
        theta2 : dynamics parameter (passed through rollout)

    Cost:
        terminal_only = True:
            J = theta1 * ||p_T - p_target||^2
                + u_weight * sum_t ||u_t||^2

        terminal_only = False:
            J = theta1 * sum_{t=1}^T ||p_t - p_target||^2
                + u_weight * sum_t ||u_t||^2
    """
    xs = rollout_T(x0, u_seq, dt, theta2)
    ps = xs[:, :2]  # positions [px, py]

    if terminal_only:
        track = theta1 * torch.sum((ps[-1] - p_target) ** 2)
    else:
        track = theta1 * torch.sum((ps[1:] - p_target) ** 2)
    # theta 1, pursuit cost; theta 2, 
    reg = 0.1 * torch.sum(u_seq[:,0]**2) + 1.0 * torch.sum(u_seq[:,1]**2) #u_weight * torch.sum(u_seq ** 2)
    return track + reg + 5.0*torch.sum((xs[-1, 2] - 3.0)**2)


# ============================================================
# Solve reduced MPC problem
# ============================================================

def solve_u_T(x0, dt, theta1, theta2, p_target, T,
              u_init=None, iters=200, u_weight=1.0, terminal_only=True):
    """
    Solve a horizon-T optimal control problem with states eliminated
    via forward rollout.

    Problem solved (reduced form):
        min_{u_0,...,u_{T-1}}  J(u; theta)
    where
        J(u; theta) = cost_T(x0, u, dt, theta1, theta2, p_target)
    and the state trajectory is implicitly defined by
        x_{t+1} = f_{theta2}(x_t, u_t),   x_0 given.

    This function returns a locally optimal control sequence u*
    obtained by unconstrained optimization using L-BFGS.
    """

    # Number of control inputs per time step (unicycle: [w, a])
    n_u = 2

    # Initialize optimization variable:
    # u_flat ∈ R^{T * n_u} is the flattened control sequence
    if u_init is None:
        # Default initialization: zero controls
        u_flat = torch.zeros(T * n_u, requires_grad=True)
    else:
        # Warm-start from a previous solution (important for FD checks)
        u_flat = u_init.clone().detach().requires_grad_(True)

    # L-BFGS optimizer for smooth unconstrained problems
    opt = torch.optim.LBFGS(
        [u_flat],
        lr=1.0,
        max_iter=iters,
        line_search_fn="strong_wolfe"
    )

    def closure():
        """
        L-BFGS closure:
        - reshapes the flattened control vector into (T, n_u)
        - evaluates the reduced objective J(u; theta)
        - computes ∇_u J via automatic differentiation
        """
        opt.zero_grad(set_to_none=True)

        # Reshape to control sequence u = (u_0, ..., u_{T-1})
        u_seq = u_flat.view(T, n_u)

        # Evaluate reduced cost (states computed internally by rollout)
        J = cost_T(
            x0, u_seq, dt, theta1, theta2, p_target,
            u_weight=u_weight,
            terminal_only=terminal_only
        )

        # Compute gradient ∇_u J(u; theta)
        J.backward()
        return J

    # Run L-BFGS optimization until convergence or iteration limit
    opt.step(closure)

    # Extract optimal solution and objective value
    with torch.no_grad():
        u_star = u_flat.view(T, n_u).clone()
        J_star = cost_T(
            x0, u_star, dt, theta1, theta2, p_target,
            u_weight=u_weight,
            terminal_only=terminal_only
        ).item()

    return u_star, J_star


# ============================================================
# Implicit differentiation (IFT)
# ============================================================

def implicit_du_dtheta_T(x0, dt, theta, p_target, u_star,
                         u_weight=1.0, terminal_only=True):
    """
    Compute sensitivity du*/dtheta using the Implicit Function Theorem.

    Stationarity condition:
        g(u, theta) = ∇_u J(u, theta) = 0

    IFT:
        du*/dtheta = - (∂g/∂u)^{-1} (∂g/∂theta)

    Inputs:
        theta   : tensor [theta1, theta2]
        u_star  : optimal control sequence, shape (T, n_u)

    Returns:
        du_dtheta : shape (T*n_u, 2)
    """
    theta = theta.clone().detach().requires_grad_(True)
    u_flat = u_star.reshape(-1).clone().detach().requires_grad_(True)

    T, n_u = u_star.shape

    def J_of(u_flat_vec, th_vec):
        theta1, theta2 = th_vec[0], th_vec[1]
        u_seq = u_flat_vec.view(T, n_u)
        return cost_T(
            x0, u_seq, dt, theta1, theta2, p_target,
            u_weight=u_weight,
            terminal_only=terminal_only
        )

    def g_of(u_flat_vec, th_vec):
        J = J_of(u_flat_vec, th_vec)
        g = torch.autograd.grad(J, u_flat_vec, create_graph=True)[0]
        return g

    # Hessian: ∂²J / ∂u²
    H = torch.autograd.functional.jacobian(
        lambda uu: g_of(uu, theta),
        u_flat
    )

    # Mixed derivative: ∂²J / ∂u∂theta
    Gtheta = torch.autograd.functional.jacobian(
        lambda th: g_of(u_flat, th),
        theta
    )

    # Solve linear system from IFT
    du_dtheta = torch.linalg.solve(H, -Gtheta)
    return du_dtheta.detach()


# 


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    dt = 0.2
    x0 = torch.tensor([0.0, 0.0, 0.0, 1.0])   # [px, py, alpha, v]
    p_target = torch.tensor([2.0, 0.5])

    theta1 = torch.tensor(5.0)   # tracking weight
    theta2 = torch.tensor(1.2)   # dynamics parameter

    T = 4

    # Solve MPC
    u_star, J_star = solve_u_T(
        x0, dt, theta1, theta2, p_target, T,
        iters=200, u_weight=1.0, terminal_only=True
    )

    # Implicit sensitivity
    du_dtheta = implicit_du_dtheta_T(
        x0, dt, torch.tensor([theta1, theta2]),
        p_target, u_star,
        u_weight=1.0, terminal_only=True
    )

    print("Horizon T =", T)
    print("u* shape:", u_star.shape)
    print("J* =", J_star)
    print("du*/dtheta shape:", du_dtheta.shape)
    print("First control sensitivity (u0):\n", du_dtheta[:2, :])