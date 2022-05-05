"""Pytorch multilayer QG model, Louis Thiry, 2022.
Followed Q-GCM user guide, Hogg et al (2014), http://q-gcm.org/downloads/q-gcm-v1.5.0.pdf.
   - State variables are pressure p and the potential vorticity q.
  - Rectangular domain, mixed slip boundary condition for velocity.
  - Advection of q is carried with the conservative Arakawa-Lamb scheme.
  - (Bi-)Laplacian diffusion discretized with centered finite difference.
  - Idealized double-gyre wind forcing
  - Time integration with Heun.
  - Tested on Intel CPU and NVIDIA GPU with pytorch 1.10 and CUDA 11.2.
"""
import numpy as np
import torch
import torch.nn.functional as F


## functions to solve elliptic equation with homogeneous boundary conditions
def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs):
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(1,nx-1, **arr_kwargs),
                          torch.arange(1,ny-1, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/(nx-1)*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/(ny-1)*y) - 1)/dy**2


def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)


def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
       using float32 discrete sine transform."""
    # return dstI2D((dstI2D(f.type(torch.float32)) / operator_dst).type(torch.float64))
    return dstI2D(dstI2D(f.type(torch.float32)) / operator_dst).type(torch.float64)


## discrete spatial differential operators
def jacobi_h(f, g):
    """Arakawa discretisation of Jacobian J(f,g).
       Scalar fields f and g must have the same dimension.
       Grid is regular and dx = dy."""
    dx_f = f[...,2:,:] - f[...,:-2,:]
    dx_g = g[...,2:,:] - g[...,:-2,:]
    dy_f = f[...,2:] - f[...,:-2]
    dy_g = g[...,2:] - g[...,:-2]
    return (
            (   dx_f[...,1:-1] * dy_g[...,1:-1,:] - dx_g[...,1:-1] * dy_f[...,1:-1,:]  ) +
            (   (f[...,2:,1:-1] * dy_g[...,2:,:] - f[...,:-2,1:-1] * dy_g[...,:-2,:]) -
                (f[...,1:-1,2:]  * dx_g[...,2:] - f[...,1:-1,:-2] * dx_g[...,:-2])     ) +
            (   (g[...,1:-1,2:] * dx_f[...,2:] - g[...,1:-1,:-2] * dx_f[...,:-2]) -
                (g[...,2:,1:-1] * dy_f[...,2:,:] - g[...,:-2,1:-1] * dy_f[...,:-2,:])  )
           ) / 12.


def laplacian_h_boundaries(f, fc):
    return fc*(torch.cat([f[...,1,1:-1],f[...,-2,1:-1], f[...,1], f[...,-2]], dim=-1) -
               torch.cat([f[...,0,1:-1],f[...,-1,1:-1], f[...,0], f[...,-1]], dim=-1))


def laplacian_h_nobc(f):
    return (f[...,2:,1:-1] + f[...,:-2,1:-1] + f[...,1:-1,2:] + f[...,1:-1,:-2]
            - 4*f[...,1:-1,1:-1])


def matmul(M, f):
    return (M @ f.reshape(f.shape[:-2] + (-1,))).reshape(f.shape)


def laplacian_h(f, fc):
    delta_f = torch.zeros_like(f)
    delta_f[...,1:-1,1:-1] = laplacian_h_nobc(f)
    delta_f_bound = laplacian_h_boundaries(f, fc)
    nx, ny = f.shape[-2:]
    delta_f[...,0,1:-1] = delta_f_bound[...,:ny-2]
    delta_f[...,-1,1:-1] = delta_f_bound[...,ny-2:2*ny-4]
    delta_f[...,0] = delta_f_bound[...,2*ny-4:nx+2*ny-4]
    delta_f[...,-1] = delta_f_bound[...,nx+2*ny-4:2*nx+2*ny-4]
    return delta_f


def grad_perp(f):
    """Orthogonal gradient computed ...,on staggered grid."""
    return f[...,:-1] - f[...,1:], f[...,1:,:] - f[...,:-1,:]


def curl_wind(tau, dx, dy):
    tau_x = 0.5 * (tau[:-1,:,0] + tau[1:,:,0])
    tau_y = 0.5 * (tau[:,:-1,1] + tau[:,1:,1])
    curl_stagg = (tau_y[1:] - tau_y[:-1]) / dx - (tau_x[:,1:] - tau_x[:,:-1]) / dy
    return  0.25*(curl_stagg[:-1,:-1] + curl_stagg[:-1,1:] + curl_stagg[1:,:-1] + curl_stagg[1:,1:])



class QGM:
    """Implementation of multilayer quasi-geostrophic model
    in variables pressure p and potential vorticity q.
    """

    def __init__(self, param):
        self.nx = param['nx']
        self.Lx = param['Lx']
        self.ny = param['ny']
        self.Ly = param['Ly']
        self.nl = param['nl']
        self.heights = param['heights']
        self.reduced_gravities = param['reduced_gravities']
        self.f0 = param['f0']
        self.a_2 = param['a_2']
        self.a_4 = param['a_4']
        self.beta = param['beta']
        self.delta_ek = param['delta_ek']
        self.dt = param['dt']
        self.bcco = param['bcco']
        self.n_ens = param['n_ens']
        self.zfbc = self.bcco / (1. + 0.5*self.bcco)
        self.device = param['device']
        if param['p_prime']:
            self.p_prime = torch.from_numpy(np.load(param['p_prime'])).type(torch.float64).to(self.device)
            if self.n_ens > 0:
                self.p_prime.unsqueeze_(0)
        else:
            self.p_prime = None
        self.arr_kwargs = {'dtype':torch.float64, 'device':self.device}

        # grid
        self.x, self.y = torch.meshgrid(torch.linspace(0, self.Lx, self.nx, **self.arr_kwargs),
                                        torch.linspace(0, self.Ly, self.ny, **self.arr_kwargs),
                                        indexing='ij')
        self.y0 = 0.5 * self.Ly
        self.dx = self.Lx / (self.nx-1)
        self.dy = self.Ly / (self.ny-1)

        assert self.dx == self.dy, f'dx {self.dx} != dy {self.dy}, must be equal'
        self.diff_coef = self.a_2 / self.f0**2 / self.dx**4
        self.hyperdiff_coef = (self.a_4 / self.f0**2) / self.dx**6
        self.jac_coef =  1. / (self.f0 * self.dx * self.dy)
        self.bottom_friction_coef = self.delta_ek / (2*np.abs(self.f0)*self.dx**2*(-self.heights[-1]))

        tau = torch.zeros((self.nx, self.ny, 2), **self.arr_kwargs)
        tau[:,:,0] = - param['tau0'] * torch.cos(2*torch.pi*(torch.arange(self.ny, **self.arr_kwargs)+0.5)/self.ny).reshape((1, self.ny))
        self.wind_forcing = (curl_wind(tau, self.dx, self.dy) / (self.f0 * self.heights[0])).unsqueeze(0)
        if self.n_ens > 0:
            self.wind_forcing.unsqueeze_(0)

        # init matrices
        self.compute_A_matrix()
        self.compute_layer_to_mode_matrices()
        self.compute_helmoltz_matrices()
        self.compute_alpha_matrix()
        self.helmoltz_dst = self.helmoltz_dst.type(torch.float32)

        # precomputations
        self.beta_y_y0_over_f0 = (self.beta / self.f0) * (self.y - self.y0)

        # initialize pressure p and potential voritcity q
        self.p_shape = (self.nl, self.nx, self.ny) if self.n_ens == 0 else (self.n_ens, self.nl, self.nx, self.ny)
        self.p_shape_flat = self.p_shape[:-2] + (self.nx*self.ny,)
        self.p = torch.zeros(self.p_shape, **self.arr_kwargs)
        self.p_modes = torch.zeros_like(self.p)
        self.compute_q_over_f0_from_p()

        # precompile torch functions
        self.zfbc = torch.tensor(self.zfbc, **self.arr_kwargs) # convert to Tensor for tracing
        self.grad_perp = torch.jit.trace(grad_perp, (self.p,))
        self.inverse_elliptic_dst = torch.jit.trace(inverse_elliptic_dst, (self.q_over_f0[...,1:-1,1:-1], self.helmoltz_dst))
        self.jacobi_h = torch.jit.trace(jacobi_h, (self.q_over_f0, self.p))
        self.laplacian_h = torch.jit.trace(laplacian_h, (self.p, self.zfbc))
        self.laplacian_h_boundaries = torch.jit.trace(laplacian_h_boundaries, (self.p, self.zfbc))
        self.laplacian_h_nobc = torch.jit.trace(laplacian_h_nobc, (self.p,))
        self.matmul = torch.jit.trace(matmul, (self.Cl2m, self.q_over_f0, ))


    def compute_A_matrix(self):
        A = torch.zeros((self.nl,self.nl), **self.arr_kwargs)
        A[0,0] = 1./(self.heights[0]*self.reduced_gravities[0])
        A[0,1] = -1./(self.heights[0]*self.reduced_gravities[0])
        for i in range(1, self.nl-1):
            A[i,i-1] = -1./(self.heights[i]*self.reduced_gravities[i-1])
            A[i,i] = 1./self.heights[i]*(1/self.reduced_gravities[i] + 1/self.reduced_gravities[i-1])
            A[i,i+1] = -1./(self.heights[i]*self.reduced_gravities[i])
        A[-1,-1] = 1./(self.heights[self.nl-1]*self.reduced_gravities[self.nl-2])
        A[-1,-2] = -1./(self.heights[self.nl-1]*self.reduced_gravities[self.nl-2])
        self.A = A.unsqueeze(0) if self.n_ens > 0 else A


    def compute_layer_to_mode_matrices(self):
        """Matrices to change from layers to modes."""
        A = self.A[0] if self.n_ens > 0 else self.A
        lambd_r, R = torch.linalg.eig(A)
        lambd_l, L = torch.linalg.eig(A.T)
        self.lambd = lambd_r.real
        R, L = R.real, L.real
        self.Cl2m = torch.diag(1./torch.diag(L.T @ R)) @ L.T
        self.Cm2l = R
        if self.n_ens > 0:
            self.Cl2m.unsqueeze_(0), self.Cm2l.unsqueeze_(0)


    def compute_helmoltz_matrices(self):
        self.helmoltz_dst = compute_laplace_dst(self.nx, self.ny, self.dx, self.dy, self.arr_kwargs).reshape((1, self.nx-2, self.ny-2)) / self.f0**2 - self.lambd.reshape((self.nl , 1, 1))
        constant_field = torch.ones((self.nl, self.nx, self.ny), **self.arr_kwargs) / (self.nx * self.ny)
        s_solutions = torch.zeros_like(constant_field)
        s_solutions[:,1:-1,1:-1] = inverse_elliptic_dst(constant_field[:,1:-1,1:-1], self.helmoltz_dst)
        self.homogeneous_sol = (constant_field +  s_solutions*self.lambd.reshape((self.nl, 1, 1)))[:-1] # ignore last solution correponding to lambd = 0, i.e. Laplace equation
        if self.n_ens > 0:
            self.helmoltz_dst.unsqueeze_(0), self.homogeneous_sol.unsqueeze_(0)


    def compute_alpha_matrix(self):
        (Cm2l, Cl2m, hom_sol) = (self.Cm2l[0], self.Cl2m[0], self.homogeneous_sol[0]) if self.n_ens > 0 else (self.Cm2l, self.Cl2m, self.homogeneous_sol)
        M = (Cm2l[1:] - Cm2l[:-1])[:self.nl-1,:self.nl-1] * hom_sol.mean((1,2)).reshape((1, self.nl-1))
        M_inv = torch.linalg.inv(M)
        alpha_matrix = -M_inv @ (Cm2l[1:,:-1] - Cm2l[:-1,:-1])
        self.alpha_matrix = alpha_matrix.unsqueeze(0) if self.n_ens > 0 else alpha_matrix


    def compute_q_over_f0_from_p(self):
        Ap = (self.A @ self.p.reshape(self.p.shape[:len(self.p.shape)-2]+(-1,))).reshape(self.p.shape)
        self.q_over_f0 = laplacian_h(self.p, self.zfbc) / (self.f0*self.dx)**2 - Ap + (self.beta / self.f0) * (self.y - self.y0)

    def compute_u(self):
        """Compute velocity on staggered grid."""
        return self.grad_perp(self.p/(self.f0*self.dx))


    def advection_rhs(self):
        """Advection diffusion RHS for vorticity, only inside domain"""
        rhs = self.jac_coef * self.jacobi_h(self.q_over_f0, self.p)

        p_diff = self.p if self.p_prime is None else self.p - self.p_prime
        delta2_p = self.laplacian_h(p_diff, self.zfbc)
        if self.a_2 != 0.:
            rhs += self.diff_coef * self.laplacian_h_nobc(delta2_p)
        if self.a_4 != 0.:
            rhs -= self.hyperdiff_coef * self.laplacian_h_nobc(self.laplacian_h(delta2_p, self.zfbc))

        rhs[...,0:1,:,:] += self.wind_forcing
        rhs[...,-1:,:,:] += self.bottom_friction_coef * self.laplacian_h_nobc(self.p[...,-1:,:,:])
        return rhs


    def compute_time_derivatives(self):
        # advect vorticity inside of the domain
        self.dq_over_f0 = F.pad(self.advection_rhs(), (1,1,1,1))

        # Solve helmoltz eq for pressure
        rhs_helmoltz = self.matmul(self.Cl2m, self.dq_over_f0)
        dp_modes = F.pad(self.inverse_elliptic_dst(rhs_helmoltz[...,1:-1,1:-1], self.helmoltz_dst), (1,1,1,1))

        # Ensure mass conservation
        dalpha =  (self.alpha_matrix @ dp_modes[...,:-1,:,:].mean((-2,-1)).unsqueeze(-1)).unsqueeze(-1)
        dp_modes[...,:-1,:,:] += dalpha * self.homogeneous_sol
        self.dp = self.matmul(self.Cm2l, dp_modes)

        # update voriticity on the boundaries
        dp_bound = torch.cat([self.dp[...,0,1:-1], self.dp[...,-1,1:-1], self.dp[...,:,0], self.dp[...,:,-1]], dim=-1)
        delta_p_bound = self.laplacian_h_boundaries(self.dp/(self.f0*self.dx)**2, self.zfbc)
        dq_over_f0_bound = delta_p_bound - self.A @ dp_bound
        self.dq_over_f0[...,0,1:-1] = dq_over_f0_bound[...,:self.ny-2]
        self.dq_over_f0[...,-1,1:-1] = dq_over_f0_bound[...,self.ny-2:2*self.ny-4]
        self.dq_over_f0[...,0] = dq_over_f0_bound[...,2*self.ny-4:self.nx+2*self.ny-4]
        self.dq_over_f0[...,-1] = dq_over_f0_bound[...,self.nx+2*self.ny-4:2*self.nx+2*self.ny-4]


    def step(self):
        """ Time itegration with Heun (RK2) scheme."""
        self.compute_time_derivatives()
        dq_over_f0_0, dp_0 = torch.clone(self.dq_over_f0), self.dp
        self.q_over_f0 += self.dt * dq_over_f0_0
        self.p += self.dt * dp_0

        self.compute_time_derivatives()
        self.q_over_f0 += self.dt * 0.5 * (self.dq_over_f0 - dq_over_f0_0)
        self.p += self.dt * 0.5 * (self.dp - dp_0)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    param = {
        # 'nx': 769, # HR
        # 'ny': 961, # HR
        'nx': 97, # LR
        'ny': 121, # LR
        'Lx': 3840.0e3, # Length in the x direction (m)
        'Ly': 4800.0e3, # Length in the y direction (m)
        'nl': 3, # number of layers
        'heights': [350., 750., 2900.], # heights between layers (m)
        'reduced_gravities': [0.025, 0.0125], # reduced gravity numbers (m/s^2)
        'f0': 9.375e-5, # coriolis (s^-1)
        'a_2': 0., # laplacian diffusion coef (m^2/s)
        # 'a_4': 2.0e9, # HR
        'a_4': 5.0e11, # LR
        'beta': 1.754e-11, # coriolis gradient (m^-1 s^-1)
        'delta_ek': 2.0, # eckman height (m)
        # 'dt': 600., # HR
        'dt': 1200., # LR
        'bcco': 0.2, # boundary condition coef. (non-dim.)
        'tau0': 2.0e-5, # wind stress magnitude m/s^2
        'n_ens': 0, # 0 for no ensemble,
        'device': 'cpu', # torch only, 'cuda' or 'cpu'
        'p_prime': ''
    }

    import time
    qg_multilayer = QGM(param)

    if param['nx'] == 97:
        qg_multilayer.p = torch.from_numpy(np.load('./p_380yrs_HRDS.npy')).to(param['device'])
    else:
        qg_multilayer.p = torch.from_numpy(np.load('./p_506yrs_HR.npy')).to(param['device'])
    qg_multilayer.compute_q_over_f0_from_p()


    # time params
    dt = param['dt']
    t = 0

    freq_plot = 1000 # LR
    # freq_plot = 50 # HR
    freq_checknan = 10000
    freq_log = 1000
    n_years = 2
    n_steps = int(n_years*365*24*3600 / dt)

    if freq_plot > 0:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure()
        f,a = plt.subplots(1,2)
        u = (qg_multilayer.compute_u()[0]).cpu().numpy()
        um, uM = -1.1*np.abs(u).max(), 1.1*np.abs(u).max()
        im = a[0].imshow(u[0].T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True)
        a[0].set_title('zonal velocity')
        f.colorbar(im, ax=a[0])
        q = (qg_multilayer.q_over_f0*qg_multilayer.f0).cpu().numpy()
        qm, qM = -1.1*np.abs(q).max(), 1.1*np.abs(q).max()
        im = a[1].imshow(q[0].T, cmap='bwr', origin='lower', vmin=qm, vmax=qM, animated=True)
        a[1].set_title('potential vorticity')
        f.colorbar(im, ax=a[1])
        plt.pause(5)

    times, outputs = [], []

    t0 = time.time()
    for n in range(1, n_steps+1):
        qg_multilayer.step()
        t += dt

        if n % freq_checknan == 0 and torch.isnan(qg_multilayer.p).any():
            raise ValueError('Stopping, NAN number in p at iteration {n}.')

        if freq_plot > 0 and n % freq_plot == 0:
            u = (qg_multilayer.compute_u()[0]).cpu().numpy()
            a[0].imshow(u[0].T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True)
            q = (qg_multilayer.q_over_f0*qg_multilayer.f0).cpu().numpy()
            a[1].imshow(q[0].T, cmap='bwr', origin='lower', vmin=qm, vmax=qM, animated=True)
            plt.suptitle(f't={t/(365*24*3600):.2f} years.')
            plt.pause(0.1)

        if freq_log > 0 and n % freq_log == 0:
            q, p = (qg_multilayer.f0 * qg_multilayer.q_over_f0).cpu().numpy(), qg_multilayer.p.cpu().numpy()
            print(f'{n=:06d}, t={t/(365*24*60**2):.2f} yr, ' \
                  f'p: ({p.mean():+.1E}, {np.abs(p).mean():.6E}), ' \
                  f'q: ({q.mean():+.1E}, {np.abs(q).mean():.6E}).')
    print(100*(time.time()-t0)/(60*60))
