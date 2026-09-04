"""
Finite strain J2 plasticity with isotropic hardening using JAX-FEM
Return mapping algorithm with Newton iteration for consistency condition
"""

import jax
import jax.numpy as np
from jax_fem_checkpoint.problem_new import Problem
import jax.flatten_util
from jax import config, jit
import logging
import jax.lax.linalg as lax_linalg
from jax import custom_jvp
from functools import partial
from jax import lax
from jax.numpy.linalg import solve


class Plasticity(Problem):
    def __init__(self, mesh, ele_type, vec, dim, dirichlet_bc_info, p_bounds, E, sig0, Q, b):
        super().__init__(mesh=mesh, 
                         ele_type=ele_type, 
                         vec=vec, dim=dim, 
                         dirichlet_bc_info=dirichlet_bc_info, 
                         p_bounds=p_bounds)
        self.E = E
        self.sig0 = sig0
        self.Q = Q
        self.b = b

    def custom_init(self):
        """Initialize internal variables and material parameters"""
        self.fe = self.fes[0]
        
        # Deformation gradient (identity initially)
        self.F_old = np.repeat(np.repeat(np.eye(self.dim)[None, None, :, :], len(self.fe.cells), axis=0),
                               self.fe.num_quads, axis=1)
        
        # Elastic left Cauchy-Green tensor (identity initially)
        self.Be_old = np.array(self.F_old)
        
        # Accumulated plastic strain (scalar, zero initially)
        self.alpha_old = np.zeros((len(self.fe.cells), self.fe.num_quads))
        
        # Shape function gradients at element center for F-bar method
        # (num_cells, num_quads, num_nodes, dim)
        self.shape_grads_center = self.fe.shape_grads_center
        self.ugrad_center = np.zeros_like(self.F_old)
        
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        
        # Material parameters per cell (11 parameters, currently using only 4)
        full_params = np.ones((self.fe.num_cells, 11))
        self.thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        
        # Pack internal variables
        self.internal_vars = [self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center, 
                             self.ugrad_center, self.thetas]
        
        self.nodes_to_compare = None
        self.target_displacement = None

    # def get_surface_maps(self):
        # """Define pressure boundary condition"""
        # def surface_map(u, point, scale, norm_vec):
        #     curr_pressure = scale * self.pressure_mag
        #     return np.array(curr_pressure * norm_vec)
        # return [surface_map]
    
    # def get_surface_kernel(self, surface_map):
        """Compute surface traction contribution (Total Lagrangian formulation)"""
        # def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale,
        #                   *cell_internal_vars_surface):
        #     # Unflatten solution
        #     cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
        #     cell_sol = cell_sol_list[0]
        #     face_shape_vals = face_shape_vals[:, :self.fes[0].num_nodes]
        #     face_shape_grads = face_shape_grads[:, :self.fes[0].num_nodes, :]
        #     face_nanson_scale = face_nanson_scale[0]

        #     # Interpolate displacement and gradient at face quadrature points
        #     u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
        #     u_grad = jax.vmap(lambda dNdx: dNdx.T @ cell_sol)(face_shape_grads)
        #     u_grad = np.transpose(u_grad, axes=(0, 2, 1))

        #     # Current configuration kinematics
        #     I = np.eye(self.dim)
        #     F = u_grad + I
        #     J = np.linalg.det(F)
        #     FinvT = jax.vmap(lambda A: np.linalg.inv(A).T)(F)

        #     # Get pressure vector from boundary condition
        #     pressure_vec = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)

        #     # Convert to reference traction: t0 = -J * F^{-T} @ pressure_vec
        #     t0 = -J[:, None] * jax.vmap(np.matmul)(FinvT, pressure_vec)

        #     # Weak form: integrate over reference surface
        #     val = np.sum(face_shape_vals[:, :, None] * t0[:, None, :] * face_nanson_scale[:, None, None], axis=0)

        #     return jax.flatten_util.ravel_pytree(val)[0]
        # return surface_kernel

    def get_laplace_kernel(self, tensor_map):
        """Compute residual (weak form of momentum balance)"""
        def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
            # Unflatten solution and extract shape functions
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, :self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]   # (num_nodes, vec)
            cell_v_grads_JxW = cell_v_grads_JxW[:, :self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            # Compute displacement gradient at quadrature points
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)

            ### Hu: TODO: Chapter 8 Element Technology
            # Compute displacement gradient at element center (F-bar method)
            shape_grads_center = cell_internal_vars[3]
            ## Hu: (None, num_nodes, vec, None) * (num_quads, num_nodes, None, dim)
            u_grads_center = cell_sol[None, :, :, None] * shape_grads_center[:, :, None, :]
            u_grads_center = np.sum(u_grads_center, axis=1)  ## Hu: (num_quads, 1, vec, dim)
            u_grads_center_reshape = u_grads_center.reshape(-1, vec, self.dim)  # Hu: (num_quads, vec, dim)

            # Update internal variables with center gradient
            # [self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center, self.ugrad_center, self.thetas]
            a1, a2, a3, a4, a5, a6 = cell_internal_vars

            print("a1: {0}, a2: {1}, a3:{2}, a4:{3}, a5:{4}, a6:{5}".format(a1.shape, a2.shape, a3.shape, a4.shape, a5.shape, a6.shape))

            cell_internal_vars_updated = (a1, a2, a3, a4, u_grads_center_reshape, a6)

            # Compute first Piola-Kirchhoff stress
            u_grads_reshape = u_grads.reshape(-1, vec, self.dim)
            ## Hu: first PK stress
            stress = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars_updated).reshape(u_grads.shape)
            
            # Internal virtual work: ∫ P : ∇δu dV
            val = np.sum(stress[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0]
            return val
        return laplace_kernel

    def get_tensor_map(self):
        """Get first Piola-Kirchhoff stress map"""
        tensor_map, _, _, _, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        """Define constitutive model and post-processing functions"""
        
        # [self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center, self.ugrad_center, self.thetas]
        def get_partial_tensor_map(F_old, be_old, alpha_old, shape_grads_center, ugrad0, theta):
            """
            J2 plasticity with nonlinear isotropic hardening
            F-bar method for volumetric locking prevention
            """
            
            # Material parameters
            E = self.E        # Young's modulus
            sig0 = self.sig0  # Initial yield stress
            Q = self.Q        # Hardening saturation
            b = self.b        # Hardening rate
            
            nu = 0.272
            K = E / (3. * (1. - 2. * nu))  # Bulk modulus
            G = E / (2. * (1. + nu))        # Shear modulus

            def first_PK_stress(u_grad):
                """Compute first Piola-Kirchhoff stress P = τ F^{-T}"""
                F, _, _, tau, _ = return_map(u_grad)
                ### Hu: Eq. (2)
                ### P -- first PK stress; tau -- Kirchhoff stress
                P = tau @ np.linalg.inv(F).T
                return P

            def update_int_vars(u_grad):
                """Update internal variables after converged increment"""
                F, be_bar, alpha, _, ugrad0_updated = return_map(u_grad)
                return F, be_bar, alpha, shape_grads_center, ugrad0_updated, theta

            def compute_cauchy_stress(u_grad):
                """Compute Cauchy stress σ = (1/J) τ"""
                F, _, _, tau, _ = return_map(u_grad)
                J = np.linalg.det(F)
                sigma = (1. / J) * tau
                return sigma

            def compute_lagrangian_strain(u_grad):
                """Compute Green-Lagrange strain E = 0.5(F^T F - I)"""
                F = u_grad + np.eye(self.dim)
                strain = 0.5 * (F @ F.T - np.eye(self.dim))
                return strain

            def compute_logarithmic_strain(u_grad):
                """Compute logarithmic (Hencky) strain ε = 0.5 ln(F F^T)"""
                F, be_updated, _, _, _ = return_map(u_grad)
                e_val, e_vec = np.linalg.eigh(F @ F.T)  # Spectral decomposition
                log_strain = 0.5 * (e_vec @ np.diag(np.log(e_val)) @ e_vec.T)
                
                # Separate diagonal and off-diagonal components
                ldg = np.eye(self.dim) * log_strain
                loff = 2. * (log_strain - ldg)
                log_st = ldg + loff
                return log_st

            def get_tau(F, be_bar):
                """Compute Kirchhoff stress from elastic left Cauchy-Green"""
                J = np.linalg.det(F)
                ## Hu: Eq. (70) & Eq. (71)
                tau = 0.5 * K * (J**2 - 1.) * np.eye(self.dim) + G * deviatoric(be_bar)
                return tau

            def deviatoric(A):
                """Deviatoric part of tensor A"""
                return A - 1. / self.dim * np.trace(A) * np.eye(self.dim)

            def K_fun(a):
                """Nonlinear isotropic hardening: K(α) = Q(1 - exp(-bα))"""
                nonlinear = Q * (1. - np.exp(-b * a))
                return nonlinear

            def return_map(u_grad):
                """
                Radial return mapping algorithm for J2 plasticity
                F-bar method for near-incompressibility
                """
                # Modified deformation gradient (F-bar method)
                ### Hu: Eq. (9) -- be: elastic part of left Cauchy-Green tensors
                be_bar_old = (np.linalg.det(F_old)**(-2. / 3.)) * be_old


                Fact = u_grad + np.eye(self.dim)
                F0 = ugrad0 + np.eye(self.dim)

                
                F = ((np.linalg.det(F0) / np.linalg.det(Fact))**(1. / 3.)) * Fact


                ### Hu: Eq. (9.3.5) -- f: relative deformation gradient
                ### Hu: Eq. (9.3.5) -- f_bar: volume-preserving deformation gradient
                F_old_inv = np.linalg.inv(F_old)
                f = F @ F_old_inv   # Incremental deformation gradient
                
                
                # Elastic predictor (9.3.14)
                f_bar = (np.linalg.det(f)**(-1. / 3.)) * f    # Eq. (68)
                be_bar_trial = f_bar @ be_bar_old @ f_bar.T   # Eq. (68)
                s_trial = G * deviatoric(be_bar_trial)        # Eq. (70)/(24)
                # Coefficient for Return-mapping algorithm
                Ie_bar = (1. / 3.) * np.trace(be_bar_trial)
                G_bar = Ie_bar * G
                

                # Check yield condition
                # Eq. (73)
                ############## Why K_fun ????????????
                yield_f_trial = np.linalg.norm(s_trial) - np.sqrt(2. / 3.) * (sig0 + K_fun(alpha_old))
                
                # Plastic corrector (if yielding)
                Delta_gamma = np.where(yield_f_trial > 0., newton_conv_mod(s_trial, be_bar_trial), 0.)
                direction = np.where(Delta_gamma > 0., s_trial / np.linalg.norm(s_trial), 0.)
                
                # Updated stress and internal variables
                s = s_trial - 2. * G_bar * Delta_gamma * direction
                alpha = alpha_old + np.sqrt(2. / 3.) * Delta_gamma
                be_bar = (s / G) + Ie_bar * np.eye(self.dim)   # Eq. (9.3.33)
                tau = get_tau(F, be_bar)
                be_updated = be_bar * (np.linalg.det(F)**(2. / 3.))
                
                return F, be_updated, alpha, tau, ugrad0

            def newton_conv_mod(s_trial, be_bar_trial):
                """
                Newton-Raphson iteration for plastic multiplier Δγ
                Solves: ||s_trial|| - √(2/3)[σ_0 + K(α)] - 2G̅Δγ = 0
                """
                Ie_bar = (1. / 3.) * np.trace(be_bar_trial)
                G_bar = Ie_bar * G

                def implicit_residual(d_gamma):
                    """Consistency condition residual"""
                    alpha_eval = alpha_old + np.sqrt(2. / 3.) * d_gamma
                    stress_dgamma = np.where(alpha_eval > 0., K_fun(alpha_eval), 0.)
                    res = (np.linalg.norm(s_trial) - np.sqrt(2. / 3.) * sig0
                           - (np.sqrt(2. / 3.) * stress_dgamma + 2. * G_bar * d_gamma))
                    return res

                def body_fun(carry, _):
                    """Newton iteration body"""
                    d_gamma, converged = carry
                    res = implicit_residual(d_gamma)
                    res_grad = jax.grad(implicit_residual)(d_gamma)
                    
                    # Update only if not converged
                    d_gamma_u = jax.lax.cond(converged, lambda d: d, lambda d: d - (res / res_grad), d_gamma)

                    # Check convergence
                    res = implicit_residual(d_gamma_u)
                    converged_updated = np.linalg.norm(res) < tol
                    return (d_gamma_u, converged_updated), None

                tol = 1.e-6
                max_iters = 50
                init_d_gamma = 0.0
                carry_init = (init_d_gamma, False)

                ## Hu: lax.scan(f, init, xs, length=None)
                ## Hu: f(carry, x) → (carry_new, y)
                # Fixed-point iteration using scan
                (d_gamma_final, _), _ = jax.lax.scan(body_fun, carry_init, None, length=max_iters)

                return d_gamma_final

            return (first_PK_stress, update_int_vars, compute_cauchy_stress, compute_lagrangian_strain,
                    compute_logarithmic_strain)

        # Wrapper functions for vmapping
        # [self.F_old, self.Be_old, self.alpha_old, self.shape_grads_center, self.ugrad_center, self.thetas]
        def tensor_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas):
            first_PK_stress, _, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                 ugrad0, thetas)
            return first_PK_stress(u_grad)

        def update_int_vars_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas):
            _, update_int_vars, _, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                 ugrad0, thetas)
            return update_int_vars(u_grad)

        def compute_cauchy_stress_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas):
            _, _, compute_cauchy_stress, _, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                       ugrad0, thetas)
            return compute_cauchy_stress(u_grad)

        def compute_lagrangian_strain_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas):
            _, _, _, compute_lagrangian_strain, _ = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                           ugrad0, thetas)
            return compute_lagrangian_strain(u_grad)

        def compute_logarithmic_strain_map(u_grad, F_old, Be_old, alpha_old, shape_grads_center, ugrad0, thetas):
            _, _, _, _, compute_logarithmic_strain = get_partial_tensor_map(F_old, Be_old, alpha_old, shape_grads_center,
                                                                            ugrad0, thetas)
            return compute_logarithmic_strain(u_grad)
            
        return (tensor_map, update_int_vars_map, compute_cauchy_stress_map, compute_lagrangian_strain_map,
                compute_logarithmic_strain_map)

    def update_int_vars_gp(self, sol, int_vars):
        """Update internal variables at all Gauss points"""
        _, update_int_vars_map, _, _, _ = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        
        # Compute displacement gradient at quadrature points
        u_grads1 = (np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] *
                   self.fe.shape_grads[:, :, :, None, :])
        u_grads = np.sum(u_grads1, axis=2)
        
        # Compute displacement gradient at element centers
        shape_grads_center = self.fe.shape_grads_center
        u_grads1c = (np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] *
                    shape_grads_center[:, :, :, None, :])
        u_gradsc = np.sum(u_grads1c, axis=2)
        
        a1, a2, a3, a4, a5, a6 = int_vars
        int_vars_updated = (a1, a2, a3, a4, u_gradsc, a6)
        
        updated_int_vars = vmap_update_int_vars_map(u_grads, *int_vars_updated)
        return updated_int_vars

    def update_shape_grads(self, sol):
        """Update shape function gradients (for updated Lagrangian)"""
        old_shape_grads = self.fe.shape_grads
        self.fe.shape_grads, self.fe.JxW = self.fe.get_shape_grads(sol)

    def compute_stress(self, sol, int_vars):
        """Compute Cauchy stress at all Gauss points"""
        _, _, compute_cauchy_stress, _, _ = self.get_maps()
        vmap_compute_cauchy_stress = jax.jit(jax.vmap(jax.vmap(compute_cauchy_stress)))
        
        u_grads = (np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] *
                   self.fe.shape_grads[:, :, :, None, :])
        u_grads = np.sum(u_grads, axis=2)
        
        sigma = vmap_compute_cauchy_stress(u_grads, *int_vars)
        return sigma

    def compute_von_mises(self, s):
        """Compute von Mises equivalent stress"""
        def von_mises(sigma):
            return np.sqrt(0.5 * ((sigma[0][0] - sigma[1][1])**2 + (sigma[1][1] - sigma[2][2])**2 + 
                                  (sigma[2][2] - sigma[0][0])**2) + 
                          3 * ((sigma[0][1])**2 + (sigma[1][2])**2 + (sigma[2][0])**2))

        von_mises_fn = jax.vmap(von_mises)
        return von_mises_fn(s)

    def compute_mag_logarithmic_strain(self, e):
        """Compute magnitude of maximum principal logarithmic strain"""
        def logarithmic_strain(epsilon):
            w, v = np.linalg.eigh(2. * epsilon + np.eye(3))
            return np.abs(np.log(np.sqrt(np.abs(w[2]))))

        log_strain_fn = jax.vmap(logarithmic_strain)
        return log_strain_fn(e)

    def compute_strain(self, sol, int_vars):
        """Compute Green-Lagrange strain at all Gauss points"""
        _, _, _, compute_lagrangian_strain, _ = self.get_maps()
        vmap_compute_lagrangian_strain = jax.jit(jax.vmap(jax.vmap(compute_lagrangian_strain)))
        
        u_grads = (np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] *
                   self.fe.shape_grads[:, :, :, None, :])
        u_grads = np.sum(u_grads, axis=2)
        
        strain = vmap_compute_lagrangian_strain(u_grads, *int_vars)
        return strain

    def compute_log_strain(self, sol, int_vars):
        """Compute logarithmic strain at all Gauss points"""
        _, _, _, _, compute_logarithmic_strain = self.get_maps()
        vmap_compute_logarithmic_strain = jax.jit(jax.vmap(jax.vmap(compute_logarithmic_strain)))
        
        u_grads = (np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] *
                   self.fe.shape_grads[:, :, :, None, :])
        u_grads = np.sum(u_grads, axis=2)
        
        strain = vmap_compute_logarithmic_strain(u_grads, *int_vars)
        return strain
