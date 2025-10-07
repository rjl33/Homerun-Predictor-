import torch

def load_flat_params(model, flat_params):
   idx = 0
   for param in model.parameters():
      numel = param.numel()
      param.data.copy_(flat_params[idx:idx + numel].view_as(param))
      idx += numel

def get_flat_params(model):
   return torch.cat([p.view(-1) for p in model.parameters()])



def hvp(model, loss, v):
   grad = torch.autograd.grad(loss, model.parameters(), create_graph=True) #compute gradient of the loss wrt to each parameter
   flat_grad = torch.cat([g.reshape(-1) for g in grad]) #flattens gradient tensors into 1 long vector
   dot = torch.dot(flat_grad, v) #computes the scalar value gradf*v (dot product of gradient and vector v)
   hv = torch.autograd.grad(dot, model.parameters(), retain_graph=True)  #Computes Hv where H is the hessian
   return torch.cat([h.reshape(-1) for h in hv])



def comp_tau(z ,d ,delta):
   #set up for qaudratic formula
   a = torch.dot(d, d)
   b = 2 * torch.dot(z, d)
   c = torch.dot(z, z) - delta**2

   discriminant = b**2 - 4 * a * c
   if discriminant < 0:
      raise RuntimeError("Discriminant is negative: no solution to Tau")
   
   sqrt_dsc = torch.sqrt(discriminant)
   #calc both possible values of tau
   tau1 = (-b - sqrt_dsc) / (2 * a)
   tau2 = (-b + sqrt_dsc) / (2 * a)
   
   tau_candidates = [tau for tau in (tau1, tau2) if tau > 0]
   if not tau_candidates:
    raise RuntimeError("No Positive solution for tau")
   
   tau = min(tau_candidates)
   return tau

def compute_grad(loss, params):
   grad = torch.autograd.grad(loss, params, retain_graph=True)
   return torch.cat([g.reshape(-1) for g in grad])

def newton_tr_cg(grad, hv_func, delta, tol = 1e-6, max_iter=100):
   r = grad.clone()
   d = -r 
   i = 0
   z = torch.zeros_like(grad) #initiailize p to be the all zeros but the corrext shape and size
   p = z.clone()
   if torch.norm(r) < tol:
      return p
   
   for i in range(max_iter):
      hv = hv_func(d)
      djbkdj = torch.dot(d, hv)
      if djbkdj <= 0:
         tau = comp_tau(z, d, delta)
         p = z + tau * d
         return p
      alpha_j = torch.dot(r, r) / djbkdj
      z_new = z + alpha_j * d
      if torch.norm(z_new) >= delta:
         tau = comp_tau(z, d, delta)
         p = z + tau * d
         return p
      r_new = r + alpha_j * hv
      if torch.norm(r_new) < tol:
         return z_new
      beta = torch.dot(r_new, r_new) / torch.dot(r, r)
      d_new = -r_new + beta * d
      d = d_new
      r = r_new
      z = z_new


def tr_sizing(model, loss_fn, X, y, delta_init=1.0, delta_max=10.0, eta = 0.25, max_iter=100):
   delta = delta_init
   params = list(model.parameters())
   losses = []


   for k in range (max_iter):

      #Compute loss using cureent model weights
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      losses.append(loss.item())

      loss = loss_fn(y_pred, y)
      grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
      grad = torch.cat([g.reshape(-1) for g in grad])

      def hv_func(v):
         return hvp(model, loss, v) #pass x as list
      
      #calculate pk
      p = newton_tr_cg(grad, hv_func, delta)

      #calculate rho
      # Save original model parameters
      original_params = get_flat_params(model).detach()

       # Apply the proposed trust region step
      new_params = (get_flat_params(model) + p).detach()
      load_flat_params(model, new_params)

# Compute new loss after the step
      y_pred_new = model(X)
      new_loss = loss_fn(y_pred_new, y)

# Compute actual reduction
      actual_red = loss.item() - new_loss.item()

# Restore original parameters (optional but safe)
      load_flat_params(model, original_params)

      pred_red = -(grad @ p + 0.5 * (p @ hv_func(p))).item()
      rho  =actual_red / pred_red

      #Trust region actual update

      if rho < 0.25:
         delta = delta * 0.25
      elif rho > 0.75 and torch.norm(p) == delta:
         delta = min(delta_max, 2 * delta)
      
      #Accept or reject step

      if rho > eta:
         params = (get_flat_params(model) +p).detach().requires_grad_(True)

      load_flat_params(model, (params + p))
         
      print(f"Iter {k}, Loss = {loss.item():.6f}, ||p|| = {torch.norm(p):.4f}, delta = {delta:.4f}, rho = {rho:.4f}")
      print(f"Grad norm: {torch.norm(grad):.4f}")
   
   return params, losses




    

        









