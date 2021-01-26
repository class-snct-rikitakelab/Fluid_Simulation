from pysph.sph.equation import Group, Equation

from math import sqrt
import numpy as np
import sys
import time
import os

from pysph.tools.geometry import get_2d_block, remove_overlap_particles
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

dim = 2
Lx = 1.0
Ly = 1.0
drop_radious=0.2

rho0d = 1.0
rho0l = 0.001
nud = 0.05
nul = 0.0005
sigma = 1.0

r0 = 0.05
v0 = 1.0

c0 = 30.0

nx = 60
dx = Lx / nx
hdx = 1.5
h0 = hdx * dx

dt_cfl = 0.25*h0/(c0+v0)
dt_visc = 0.125*rho0l*h0*h0/nul
dt_surf = 0.25*np.sqrt(rho0l*h0*h0*h0/(2.0*np.pi*sigma))
dt = 0.9*min(dt_cfl, dt_visc, dt_surf)

print("CFL条件："+ str(dt_cfl))
print("粘性の条件："+str(dt_visc))
print("表面張力の条件："+ str(dt_surf))
print("カーネル半径："+ str(h0))

#アプリケーションの設定
tf = 0.5
pfreq =  tf/dt//200
if pfreq == 0:
    pfreq = 1

print("終了時刻："+ str(tf))
print("時間刻み："+ str(dt))
print("出力周期："+ str(pfreq))


class SummationDensity(Equation):
    def initialize(self, d_idx, d_V, d_rho):
        d_V[d_idx] = 0.0
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ):
        d_V[d_idx] += WIJ
        d_rho[d_idx] += d_m[d_idx]*WIJ

class StateEquation(Equation):
    def __init__(self, dest, sources, p0, b=1.0):
        self.b = b
        self.p0 = p0
        super(StateEquation, self).__init__(dest, sources)

    def loop(self, d_idx, d_p, d_rho,d_rho0):
        d_p[d_idx] = self.p0 * ((d_rho[d_idx]/d_rho0[d_idx])**7 - self.b)
        
class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, pb, gx=0., gy=0., gz=0.,
                 tdamp=0.0):

        self.pb = pb
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw, d_auhat, d_avhat, d_awhat):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho,
             d_au, d_av, d_aw, d_p, s_p,
             d_auhat, d_avhat, d_awhat, d_V, s_V, DWIJ):

        # averaged pressure Eq. (7)
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        p_i = d_p[d_idx]
        p_j = s_p[s_idx]

        pij = rhoj * p_i + rhoi * p_j
        pij /= (rhoj + rhoi)

        # particle volumes; d_V is inverse volume
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # inverse mass of destination particle
        mi1 = 1.0/d_m[d_idx]

        # accelerations 1st term in Eq. (8)
        tmp = -pij * mi1 * (Vi2 + Vj2)

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]

        # contribution due to the background pressure Eq. (13)
        tmp = -self.pb * mi1 * (Vi2 + Vj2)

        d_auhat[d_idx] += tmp * DWIJ[0]
        d_avhat[d_idx] += tmp * DWIJ[1]
        d_awhat[d_idx] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, t):
        # damped accelerations due to body or external force
        damping_factor = 1.0
        if t < self.tdamp:
            damping_factor = 0.5 * (sin((-0.5 + t/self.tdamp)*M_PI) + 1.0)

        d_au[d_idx] += self.gx * damping_factor
        d_av[d_idx] += self.gy * damping_factor
        d_aw[d_idx] += self.gz * damping_factor

class SolidWallPressureBCnoDensity(Equation):
    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        d_p[d_idx] += s_p[s_idx]*WIJ

        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        if d_wij[d_idx] > 1e-14:
#        if d_wij[d_idx] > 0.0:
            d_p[d_idx] /= d_wij[d_idx]


class SolidWallNoSlipBC(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(SolidWallNoSlipBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho, d_V, s_V,
             d_u, d_v, d_w,
             d_au, d_av, d_aw,
             s_ug, s_vg, s_wg,
             DWIJ, R2IJ, EPS, XIJ):

        # averaged shear viscosity Eq. (6).
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # particle volumes; d_V inverse volume.
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # scalar part of the kernel gradient
        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # viscous contribution (third term) from Eq. (8), with VIJ
        # defined appropriately using the ghost values
        tmp = 1./d_m[d_idx] * (Vi2 + Vj2) * (etaij * Fij/(R2IJ + EPS))

        d_au[d_idx] += tmp * (d_u[d_idx] - s_ug[s_idx])
        d_av[d_idx] += tmp * (d_v[d_idx] - s_vg[s_idx])
        d_aw[d_idx] += tmp * (d_w[d_idx] - s_wg[s_idx])  

class CSFSurfaceTensionForceAdami(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_kappa, d_cx, d_cy, d_cz,
                  d_m, d_alpha, d_rho):
        d_au[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cx[d_idx]/d_rho[d_idx]
        d_av[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cy[d_idx]/d_rho[d_idx]
        d_aw[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cz[d_idx]/d_rho[d_idx]

class AdamiColorGradient(Equation):
    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz, d_ddelta,
                  d_N):
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

        # reliability indicator for normals
        d_N[d_idx] = 0.0

        # Discretized dirac-delta
        d_ddelta[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_V, s_V, d_rho, s_rho,
             d_cx, d_cy, d_cz, d_color, s_color, DWIJ):

        # particle volumes
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # averaged particle color
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        rhoij1 = 1./(rhoi + rhoj)

        # Eq. (15) in [A10]
        cij = rhoj*rhoij1*d_color[d_idx] + rhoi*rhoij1*s_color[s_idx]

        # comute the gradient
        tmp = cij * (Vi2 + Vj2)/Vi

        d_cx[d_idx] += tmp * DWIJ[0]
        d_cy[d_idx] += tmp * DWIJ[1]
        d_cz[d_idx] += tmp * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz, d_h,
                  d_nx, d_ny, d_nz, d_ddelta, d_N):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        h2 = d_h[d_idx]*d_h[d_idx]
        #if mod_gradc2 > 1e-4/h2:
        if mod_gradc2 > 0.0:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            one_mod_gradc = 1./sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * one_mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * one_mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * one_mod_gradc

            # discretized dirac delta
            d_ddelta[d_idx] = 1./one_mod_gradc

class AdamiReproducingDivergence(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super(AdamiReproducingDivergence, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa, d_wij_sum):
        d_kappa[d_idx] = 0.0
        d_wij_sum[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_wij_sum,
             d_nx, d_ny, d_nz, s_nx, s_ny, s_nz, d_V, s_V,
             DWIJ, XIJ, RIJ, EPS):
        # particle volumes
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]

        # dot product in the numerator of Eq. (20)
        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]

        # dot product in the denominator of Eq. (20)
        xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # accumulate the contributions
        d_kappa[d_idx] += nijdotdwij * Vj
        d_wij_sum[d_idx] += xijdotdwij * Vj

    def post_loop(self, d_idx, d_kappa, d_wij_sum):
        # normalize the curvature estimate
        #if d_wij_sum[d_idx] > 1e-12:        
        if d_wij_sum[d_idx] > 0.0:
            d_kappa[d_idx] /= d_wij_sum[d_idx]
        d_kappa[d_idx] *= -self.dim


class MomentumEquationViscosityAdami(Equation):

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx,
             d_m, R2IJ, XIJ, EPS, VIJ, d_nu, s_nu):
        factor = 2.0*d_nu[d_idx]*s_nu[s_idx]/(d_nu[d_idx] + s_nu[s_idx])
        V_i = 1/(d_V[d_idx]*d_V[d_idx])
        V_j = 1/(s_V[s_idx]*s_V[s_idx])
        dwijdotrij = (DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2])
        dwijdotrij /= (R2IJ + EPS)
        factor = factor*(V_i + V_j)*dwijdotrij/d_m[d_idx]
        d_au[d_idx] += factor*VIJ[0]
        d_av[d_idx] += factor*VIJ[1]
        d_aw[d_idx] += factor*VIJ[2]

#squared radious 半径二乗
def r(x, y):
    return x*x + y*y


class MultiPhase(Application):
    def create_particles(self):
        h0 = hdx * dx

        additional_props = ['V', 
                'scolor', 'cx', 'cy', 'cz',
                'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                'kappa', 'arho', 'wg', 'ug', 'vg',
                'pi00', 'pi01', 'pi02', 'pi10', 'pi11', 'pi12',
                'pi20', 'pi21', 'pi22'
        ]
        
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx, height=Ly, center=np.array([0., 0.]))
        
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        
        u=np.zeros((len(fluid_x)))
        v=np.zeros((len(fluid_x)))
        
        for i in range(len(fluid_x)):
            R = sqrt(r(fluid_x[i], fluid_y[i]) + 0.0001*h_fluid[i]*h_fluid[i])
            f = np.exp(-R/r0)/r0
            u[i] = v0*fluid_x[i]*(1.0-(fluid_y[i]*fluid_y[i])/(r0*R))*f
            v[i] = -v0*fluid_y[i]*(1.0-(fluid_x[i]*fluid_x[i])/(r0*R))*f
            
        color=np.zeros((len(fluid_x)))
        rho_fluid = np.zeros((len(fluid_x)))       
        nu_fluid = np.zeros((len(fluid_x)))
        rho0_fluid = np.zeros((len(fluid_x)))
        for i in range(len(fluid_x)):
            if r(fluid_x[i],fluid_y[i]) < drop_radious**2 :
                color[i] = 1.0
                rho_fluid[i]=rho0d
                rho0_fluid[i]=rho0d
                nu_fluid[i]=nud
            else:
                color[i] = 0.0
                rho_fluid[i]=rho0l
                rho0_fluid[i]=rho0l
                nu_fluid[i]=nul
                
        m_fluid = rho_fluid * dx * dx
        consts = {'max_ddelta': np.zeros(1, dtype=float)}
        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props,
            constants=consts,u=u,v=v)
        
        fluid.add_property("color",data=color)
        fluid.add_property("rho0",data=rho0_fluid)
        fluid.add_property("nu",data=nu_fluid)
        
        fluid.add_property("alpha",default=sigma)
        
        wall_x, wall_y = get_2d_block(dx=dx, length=Lx+6*dx, height=Ly+6*dx,
                                      center=np.array([0., 0.]))
        rho_wall = np.ones_like(wall_x) * rho0l
        m_wall = rho_wall * dx * dx
        h_wall = np.ones_like(wall_x) * h0
        cs_wall = np.ones_like(wall_x) * c0

        wall = get_particle_array(
            name='wall', x=wall_x, y=wall_y, h=h_wall, m=m_wall,
            rho=rho_wall, cs=cs_wall, additional_props=additional_props)
        wall.add_property("color",default=0)
        
        remove_overlap_particles(wall, fluid, dx_solid=dx, dim=2)
        
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                 'ddelta', 'kappa', 'N', 'scolor', 'p','rho0','nu','wij_sum'])
        wall.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                'kappa', 'N', 'scolor', 'p','u','v'])

        return [fluid, wall]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False,
            pfreq = pfreq)
        return solver

    def create_equations(self):
        result = []

        equations = []
        equations.append(SummationDensity(dest='fluid', sources=['fluid']+['wall']))
        equations.append(SummationDensity(dest='wall', sources=['fluid']+['wall']))
        result.append(Group(equations, real=True))

        equations = []
#         equations.append(StateEquation(dest='fluid', sources=None, rho0=rho0,p0=c0**2 * rho0, b=0))        
        equations.append(StateEquation(dest='fluid', sources=None,p0=c0**2 * rho0d/7))
        print(c0**2 * rho0d / 7)

        equations.append(SolidWallPressureBCnoDensity(dest='wall',sources=['fluid']))
        result.append(Group(equations, real=True))

        equations = []
        equations.append(AdamiColorGradient(dest='fluid', sources=['fluid']+['wall']))
        result.append(Group(equations, real=True))

        equations = []
        equations.append(AdamiReproducingDivergence(dest='fluid',
                        sources=['fluid']+['wall'],dim=2))
        result.append(Group(equations, real=True))
        equations = []
        equations.append(MomentumEquationPressureGradient(
              dest='fluid', sources=['fluid']+['wall'], pb=0.0))
        equations.append(MomentumEquationViscosityAdami(dest='fluid',sources=['fluid']))
        equations.append(CSFSurfaceTensionForceAdami(dest='fluid', sources=None))
        equations.append(SolidWallNoSlipBC(dest='fluid', sources=['wall'],nu=nul))
        result.append(Group(equations))
        return result

    def post_process(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Post processing requires Matplotlib")
            return
        from pysph.solver.utils import load
        files = self.output_files
        amat = []
        t = []
        centerx = []
        centery = []
        velx = []
        vely = []

        for f in files:
            data = load(f)
            pa = data['arrays']['fluid']
            t.append(data['solver_data']['t'])
            x = pa.x
            y = pa.y
            u = pa.u
            v = pa.v
            color = pa.color
            length = len(color)
            min_x = 0.0
            max_x = 0.0
            cx = 0
            cy = 0
            vx = 0
            vy = 0
            count = 0
            for i in range(length):
                if color[i] == 1:
                    if x[i] < min_x:
                        min_x = x[i]
                    if x[i] > max_x:
                        max_x = x[i]
                    if x[i] > 0 and y[i] > 0:
                        cx += x[i]
                        cy += y[i]
                        vx += u[i]
                        vy += v[i]
                        count += 1
            amat.append(0.5*(max_x - min_x))
            centerx.append(cx/count)
            centery.append(cy/count)
            velx.append(vx/count)
            vely.append(vy/count)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, semimajor=amat, centerx=centerx, centery=centery,
                 velx=velx, vely=vely)
        plt.plot(t, amat)
        fig = os.path.join(self.output_dir, 'semimajorvst.png')
        plt.savefig(fig)
        plt.close()
        plt.plot(t, centerx, label='x position')
        plt.plot(t, centery, label='y position')
        plt.legend()
        fig1 = os.path.join(self.output_dir, 'centerofmassposvst')
        plt.savefig(fig1)
        plt.close()
        plt.plot(t, velx, label='x velocity')
        plt.plot(t, vely, label='y velocity')
        plt.legend()
        fig2 = os.path.join(self.output_dir, 'centerofmassvelvst')
        plt.savefig(fig2)
        plt.close()
        
if __name__ == '__main__':
    app = MultiPhase(output_dir="/opt/ml/model/experiment")
    app.run(["--opencl"])
    app.post_process()

