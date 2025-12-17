
import torch

def rk4_step(f, x, u, dt):
    k1=f(x,u); k2=f(x+0.5*dt*k1,u); k3=f(x+0.5*dt*k2,u); k4=f(x+dt*k3,u)
    return x+(dt/6)*(k1+2*k2+2*k3+k4)

def pde_loss(W, policy, system, xs):
    Wx, g = W.forward_with_grad(xs)
    fx = system.f_torch(xs, policy(xs))
    return ((g*fx).sum(dim=1) + (1-Wx)).pow(2).mean()
