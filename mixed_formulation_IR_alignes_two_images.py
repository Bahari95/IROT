"""
mixed_formulation_IR.py

# Alignes two given images by computing the optimal mapping assiciated to the Monge-Ampere equation
# using the Picard BFO algorithm

@author : M. BAHARI
"""

from   simplines                    import compile_kernel
from   simplines                    import SplineSpace
from   simplines                    import TensorSpace
from   simplines                    import StencilMatrix
from   simplines                    import StencilVector
from   simplines                    import pyccel_sol_field_2d
from   simplines                    import quadratures_in_admesh
#.. Prologation by knots insertion matrix
from   simplines                    import prolongation_matrix
# ... Using Kronecker algebra accelerated with Pyccel
from   simplines                    import Poisson
#--- Image registration
from   interpolimage import         image_in_quadraturpoints
from   interpolimage import         image_in_uniformpoints

from gallery_section_21 import assemble_stiffnessmatrix1D
from gallery_section_21 import assemble_massmatrix1D
from gallery_section_21 import assemble_matrix_ex11
from gallery_section_21 import assemble_matrix_ex12
from gallery_section_21 import assemble_Quality_ex01

assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex11, arity=1)
assemble_matrix_ex02 = compile_kernel(assemble_matrix_ex12, arity=1)
assemble_Quality     = compile_kernel(assemble_Quality_ex01, arity=1)

#---mae : In Monge-Ampere equation
from gallery_section_21 import assemble_vector_ex0mae
assemble_rhsmae = compile_kernel(assemble_vector_ex0mae, arity=1)

#..using density based on image pixels
from gallery_section_21 import assemble_vector_rhsmae
assemble_monmae       = compile_kernel(assemble_vector_rhsmae, arity=1)


#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   tabulate                     import tabulate
import numpy                        as     np
import time

#==============================================================================
#.......Picard BFO ALGORITHM

#==============================================================================
class Picard(object):
    def __init__(self, V1, V2, V3, V4, V00, V11, V01, V10):
       #___
       I1         = np.eye(V3.nbasis)
       I2         = np.eye(V4.nbasis)

       # .. computes basis and sopans in adapted quadrature
       V          = TensorSpace(V1, V2, V3, V4)
       self.Quad  = quadratures_in_admesh(V)
       #... We delete the first and the last spline function
       #.. as a technic for applying Neumann boundary condition
       #.in a mixed formulation

       #..Stiffness and Mass matrix in 1D in the first deriction
       D1         = assemble_mass1D(V3)
       D1         = D1.tosparse()
       D1         = D1.toarray()
       D1         = csr_matrix(D1)
       #___
       M1         = assemble_mass1D(V1)
       M1         = M1.tosparse()
       m1         = M1
       M1         = M1.toarray()[1:-1,1:-1]
       M1         = csc_matrix(M1)
       m1         = csr_matrix(m1)

       #..Stiffness and Mass matrix in 1D in the second deriction
       D2         = assemble_mass1D(V4)
       D2         = D2.tosparse()
       D2         = D2.toarray()
       D2         = csr_matrix(D2)
       #___
       M2         = assemble_mass1D(V2)
       M2         = M2.tosparse()
       m2         = M2
       M2         = M2.toarray()[1:-1,1:-1]
       M2         = csc_matrix(M2)
       m2         = csr_matrix(m2)

       #...
       R1         = assemble_matrix_ex01(V01)
       R1         = R1.toarray()
       R1         = R1.reshape(V01.nbasis)
       r1         = R1.T
       R1         = R1[1:-1,:].T
       R1         = csr_matrix(R1)
       r1         = csr_matrix(r1)
       #___
       R2         = assemble_matrix_ex02(V10)
       R2         = R2.toarray()
       R2         = R2.reshape(V10.nbasis)
       r2         = R2
       R2         = R2[:,1:-1]
       R2         = csr_matrix(R2)
       r2         = csr_matrix(r2)

       #...step 0.1
       mats_1     = [M1, M1]
       mats_2     = [D2, D2]

       # ...Fast Solver
       poisson_c1 = Poisson(mats_1, mats_2)
              
       #...step 0.2
       mats_1     = [D1, D1]
       mats_2     = [M2, M2]

       # ...Fast Solver
       poisson_c2 = Poisson(mats_1, mats_2)
       
       #...step 1
       M1         = sla.inv(M1)
       A1         = M1.dot(R1.T)
       K1         = R1.dot( A1)
       K1         = csr_matrix(K1)
       #___
       M2         = sla.inv(M2)
       A2         = M2.dot( R2.T)
       K2         = R2.dot( A2)
       K2         = csr_matrix(K2)

       #...step 2
       mats_1     = [D1, K1]
       mats_2     = [D2, K2]

       # ...Fast Solver
       poisson    = Poisson(mats_1, mats_2)

       #  ... Strong form of Neumann boundary condition which is Dirichlet because of Mixed formulation
       u_01       = StencilVector(V01.vector_space)
       u_10       = StencilVector(V10.vector_space)
       #..
       x_D        = np.zeros(V01.nbasis)
       y_D        = np.zeros(V10.nbasis)

       x_D[-1, :] = 1. 
       y_D[:, -1] = 1.
       #..
       #..
       u_01.from_array(V01, x_D)
       u_10.from_array(V10, y_D)
       #...non homogenoeus Neumann boundary 
       b01        = -kron(r1, D2).dot(u_01.toarray())
       #__
       b10        = -kron(D1, r2).dot( u_10.toarray())
       self.b_0   = b01 + b10
       #...
       b11        = -kron(m1[1:-1,:], D2).dot(u_01.toarray())
       #___
       b12        = -kron(D1, m2[1:-1,:]).dot(u_10.toarray())
       
       #___Solve first system
       self.r_0   =  kron(A1.T, I2).dot(b11) + kron(I1, A2.T).dot(b12)

       #___
       self.x11_1 = kron(A1, I2)
       self.x12_1 = kron(I1, A2)
       #___
       self.C1    = poisson_c1.solve(2.*b11)
       self.C2    = poisson_c2.solve(2.*b12)
        
       self.spaces = [V1, V2, V3, V4, V11, V01, V10]
       self.poisson= poisson
       self.D1 = D1
       self.D2 = D2
    def solve(self, V= None, x_01 = None, x_10 = None, u_H = None, u2_H = None, niter = None):

        V1, V2, V3, V4, V11, V01, V10    = self.spaces[:]
        poisson       = self.poisson

        from numpy import sqrt
        #----------------------------------------------------------------------------------------------
        tol    = 1.e-7
        if niter is None:
             niter  = 50
        
        if x_01 is None :
            # ... for Two or Multi grids
            u11     = StencilVector(V01.vector_space)
            u12     = StencilVector(V10.vector_space)
            x11     = np.zeros(V01.nbasis) # dx/ appr.solution
            x12     = np.zeros(V10.nbasis) # dy/ appr.solution
            # ...
            u11.from_array(V01, x11)
            u12.from_array(V10, x12)
            # ...Assembles Neumann boundary conditions
            x11[-1,:]  = 1.
            x12[:,-1]  = 1.
            # .../
            x_2     = zeros(V3.nbasis*V4.nbasis)
        else:
            u11          = StencilVector(V01.vector_space)
            u12          = StencilVector(V10.vector_space)
            x11          = np.zeros(V01.nbasis) # dx/ appr.solution
            x12          = np.zeros(V10.nbasis) # dy/ appr.solution
            # ...Assembles Neumann (Dirichlet) boundary conditions
            x11[-1,:]    = 1.
            x12[:,-1]    = 1.
            # ...
            x11[1:-1,:]  =  x_01[1:-1,:] #(self.C1 - self.x11_1.dot(x_2)).reshape([V1.nbasis-2,V3.nbasis])
            u11.from_array(V01, x11)
            #___
            x12[:,1:-1]  =  x_10[:,1:-1] #(self.C2 - self.x12_1.dot(x_2)).reshape([V4.nbasis,V2.nbasis-2])
            u12.from_array(V10, x12)     
            i = 0
            l2_residual = 0.
            x_2     = zeros(V3.nbasis*V4.nbasis)
        # ... for assembling residual
        M_res      = kron(self.D1, self.D2)
        for i in range(niter) :          
           
                   # ... computes spans and basis in adapted quadrature 
                   spans_ad1, spans_ad2, basis_ad1, basis_ad2 = self.Quad.ad_quadratures(u11, u12)
                   # --- Assembles a right hand side of Poisson equation
                   rhs          = StencilVector(V11.vector_space)
                   rhs          = assemble_rhsmae(V, fields = [u11, u12, u_H, u2_H], value = [spans_ad1, spans_ad2, basis_ad1, basis_ad2], out= rhs)
                   b            = rhs.toarray()
                   b            = self.b_0 + b.reshape(V4.nbasis*V3.nbasis)
                   #___
                   r            = self.r_0 - b
           
                   # ... Solve first system
                   x2           = poisson.solve(r)
                   x2           = x2 -sum(x2)/len(x2)
                   #___
                   x11[1:-1,:]  =  (self.C1 - self.x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])      
                   u11.from_array(V01, x11)
                   #___
                   x12[:,1:-1]  =  (self.C2 - self.x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
                   u12.from_array(V10, x12)
  
                   #..Residual   
                   dx           = x2[:]-x_2[:]
                   x_2[:]       = x2[:]
           
                   #... Compute residual for L2
                   l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
            
                   if l2_residual < tol:
                       break
        #print('-----> N-iter in Picard ={} -----> l2_residual= {}'.format(i, l2_residual))
        return u11, u12, x11, x12, x_2, l2_residual, i
       
#==============================================================================       
def   Pr_h_solve(V1, V2, V3, V4, u): 
       # Stiffness and Mass matrix in 1D in the first deriction
       M1      = assemble_mass1D(V3)
       M1      = M1.tosparse()

       # Stiffness and Mass matrix in 1D in the second deriction
       M2      = assemble_mass1D(V4)
       M2      = M2.tosparse()

       mats_1  = [M1, M1]
       mats_2  = [M2, M2]
       # ...
       poisson = Poisson(mats_1, mats_2)      
       # ..
       V_T     = TensorSpace(V1, V2, V3, V4)
       V       = TensorSpace(V3, V4)
       # ...
       u_L2    = StencilVector(V.vector_space)
       #...
       rhs     = StencilVector(V.vector_space)
       rhs     = assemble_monmae(V_T, value = [u], out = rhs)
       b       = 2.*rhs.toarray()
       
       #---Solve a linear system
       x       = poisson.solve(b)
       x       = x.reshape(V.nbasis)

       #...
       u_L2.from_array(V, x)
       return u_L2, x

degree      = 3
quad_degree = degree+2
nb_ne       = 8  # >= 5
nelements   = 16 #2**nb_ne

Geometry    = False
#----------------------
#..... Initialisation and computing optimal mapping for 16*16
#----------------------
# create the spline space for each direction --- MAE equation
Vm1  = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2, quad_degree = quad_degree)
Vm2  = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V1   = SplineSpace(degree=degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V2   = SplineSpace(degree=degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)

VH1  = SplineSpace(degree=degree-1, nelements= 2**nb_ne, nderiv = 2, quad_degree = quad_degree)
VH2  = SplineSpace(degree=degree-1, nelements= 2**nb_ne, nderiv = 2, quad_degree = quad_degree)
Vhf  = TensorSpace(VH1, VH2)

# create the tensor space
Vh   = TensorSpace(V1, V2)
if Geometry is True :
	xmp            = np.loadtxt('Piecex_2_16.txt')
	ymp            = np.loadtxt('Piecey_2_16.txt')

# create the tensor space
V00 = TensorSpace(Vm1, Vm2)
V11 = TensorSpace(V1, V2)
V01 = TensorSpace(Vm1, V1)
V10 = TensorSpace(V2, Vm2)
Vmh = TensorSpace(Vm1, Vm2, V1, V2, VH1, VH2)


#... Computes the tools for Mixed V-F of MAE
Pi = Picard(Vm1, Vm2, V1, V2, V00, V11, V01, V10)

#--------------§§§!!!!!! replace the image's name by the new one
#...  test 1
#image_name = "figures/brain_mri_transversal_t2_003.jpg"
#...  test 2 Flower
#image_name = "figures/flower.jpeg"
#...  test 3
#image_name = "figures/grayscal.jpg"
#image_name = "figures/print.jpg"
#image_name = "figures/1000_F.jpg"
#image_name = "figures/mandelboring.png"
#image_name = "figures/optical_illusion.jpeg"
#...  test 4
#image_name = "figures/cat.jpg"
#...  test 5
image_name  = "figures/r11egist.png"
image_name2 = "figures/r12egist.png"
#...  test 6
#image_name = "figures/hand.jpeg"
#...  test 7
#image_name = "figures/Beautiful.png"
# ...

# ...
ne1, k1          = VH1.points.shape
image_array      = zeros((ne1*k1, ne1*k1)) 
image_array[:,:] = image_in_quadraturpoints(VH1.points, VH2.points,image_name).T[:,:]
# ..
image_array2      = zeros((ne1*k1, ne1*k1)) 
image_array2[:,:] = image_in_quadraturpoints(VH1.points, VH2.points,image_name2).T[:,:]

#... computation of the optimal mapping using last solution 
print('#---IN-Adapted--MESH')
u_rho_H, x_rho    = Pr_h_solve(VH1, VH2, VH1, VH2, image_array)
u2_rho_H, x2_rho    = Pr_h_solve(VH1, VH2, VH1, VH2, image_array2)
#np.savetxt('saitama'+str(VH1.nelements)+'.txt', x_rho, fmt='%.2e')
niter   = 30
# ... new discretization for plot
nbpts   = 500
print("	\subcaption{Degree $p =",degree,"$}")
print("	\\begin{tabular}{r c c c c c}")
print("		\hline")
print("		$\#$cells & CPU-time (s) & N-iter  & Qual &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
print("		\hline")
# ... For multigrid method
#++
start0 = time.time()
u11H_mae, u12H_mae, x11_mae, x12_mae, x2H, l2_residual, n_ite = Pi.solve(V=Vmh, u_H = u_rho_H, u2_H = u2_rho_H)
start0 = time.time() - start0

# ...
Vmh              = TensorSpace(Vm1, Vm2, V1, V2)
Quality          = StencilVector(V11.vector_space)
Quality          = assemble_Quality(Vmh, fields=[u11H_mae, u12H_mae], value = [0.], out = Quality)
norm             = Quality.toarray()
l2_displacement  = norm[0]
#---Compute a solution
Xmae,uxx,uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11_mae , V01.knots, V01.degree)
Ymae,uyx,uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12_mae , V10.knots, V10.degree)[0:3]

# ... Jacobian function of Optimal mapping
det = uxx*uyy-uxy*uyx
# ...
det_min          = np.min( det[1:-1,1:-1])
det_max          = np.max( det[1:-1,1:-1])
# ... scientific format
l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
MG_time          = round(start0, 3)
det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
print("		",nelements, "&", MG_time, "&", n_ite+1, "&", l2_displacement, "&", det_min, "&", det_max,"\\\\")
for n in range(5,nb_ne+1):
	nelements   = 2**n
	# create the spline space for each direction --- MAE equation
	Vm1 = SplineSpace(degree=degree,   nelements= nelements,   nderiv = 2, quad_degree = quad_degree)
	Vm2 = SplineSpace(degree=degree,   nelements= nelements,   nderiv = 2, quad_degree = quad_degree)
	V1  = SplineSpace(degree=degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
	V2  = SplineSpace(degree=degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)

	# create the tensor space
	Vhp = TensorSpace(V1, V2)

	# create the tensor space
	V00 = TensorSpace(Vm1, Vm2)
	V11= TensorSpace(V1, V2)
	Vh01 = TensorSpace(Vm1, V1)
	Vh10 = TensorSpace(V2, Vm2)

	#... Computes the tools for Mixed V-F of MAE
	Pi = Picard(Vm1, Vm2, V1, V2, V00, V11, Vh01, Vh10)
	#.. Prologation by knots insertion matrix
	M           = prolongation_matrix(V01, Vh01)
	x11_mae     = (M.dot(u11H_mae.toarray())).reshape(Vh01.nbasis)

	M           = prolongation_matrix(V10, Vh10)
	x12_mae     = (M.dot(u12H_mae.toarray())).reshape(Vh10.nbasis)

	# ...
	#ne1, k1          = V1.points.shape
	#image_array      = zeros((ne1*k1, ne1*k1)) 
	#image_array[:,:] = image_in_quadraturpoints(V1.points, V2.points,image_name).T[:,:]

	#... computation of the optimal mapping using last solution 
	#print('#---IN-Adapted--MESH')
	#u_rho_H, x_rho    = Pr_h_solve(V1, V2, V1, V2, image_array)
	'''
	M           = prolongation_matrix(Vhp, Vhf)
	x_rho1       = ((M.T).dot(u_rho_H.toarray())).reshape(Vhp.nbasis)
	u_rho_H1     = StencilVector(Vhp.vector_space)
	u_rho_H1.from_array(Vhp, x_rho1)
	'''
	#++
	Vmh = TensorSpace(Vm1, Vm2, V1, V2, VH1, VH2)
	#..
	start1 = time.time()
	u11H_mae, u12H_mae, x11_mae, x12_mae, x2H, l2_residual, n_ite = Pi.solve(V=Vmh, u_H = u_rho_H, u2_H = u2_rho_H, x_01 = x11_mae, x_10 = x12_mae, niter = niter)
	start0 += time.time() - start1
	#...
	V01  = TensorSpace(Vm1, V1)
	V10  = TensorSpace(V2, Vm2)

	# ...
	Vmh              = TensorSpace(Vm1, Vm2, V1, V2)
	Quality          = StencilVector(V11.vector_space)
	Quality          = assemble_Quality(Vmh, fields=[u11H_mae, u12H_mae], value = [0.], out = Quality)
	norm             = Quality.toarray()
	l2_displacement  = norm[0]
	#---Compute a solution
	Xmae,uxx,uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11_mae , V01.knots, V01.degree)
	Ymae,uyx,uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12_mae , V10.knots, V10.degree)[0:3]

	# ... Jacobian function of Optimal mapping
	det = uxx*uyy-uxy*uyx
	# ...
	det_min          = np.min( det[1:-1,1:-1])
	det_max          = np.max( det[1:-1,1:-1])
	# ... scientific format
	l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
	MG_time          = round(start0, 3)
	det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
	det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
	print("		",nelements, "&", MG_time, "&", n_ite+1, "&", l2_displacement, "&", det_min, "&", det_max,"\\\\")
print("		\hline")
print("	\end{tabular}")
print('\n')
if False :
	V1mpH          = SplineSpace(degree=2,   nelements= 16)
	V2mpH          = SplineSpace(degree=2,   nelements= 16)	
	VHmp           = TensorSpace(V1mpH, V2mpH)
	xm             = np.loadtxt('Squarx_'+str(2)+'_16.txt')
	ym             = np.loadtxt('Squary_'+str(2)+'_16.txt')
	nxm  = pyccel_sol_field_2d( None,  x11_mae , Vh01.knots, Vh01.degree, meshes = (xm, ym))[0]
	nym  = pyccel_sol_field_2d( None,  x12_mae , Vh10.knots, Vh10.degree, meshes = (xm, ym))[0]	
	np.savetxt('Square1x'+str(degree)+'.txt', nxm, fmt='%.20e')
	np.savetxt('Square1y'+str(degree)+'.txt', nym, fmt='%.20e')
	

#---print errror results
#~~~~~~~~~~~~
if False :
	# ... new discretization for plot
	nbpts           = 400
	print("	\subcaption{Degree $p =",degree,"$}")
	print("	\\begin{tabular}{r c c c c c}")
	print("		\hline")
	print("		$\#$cells & CPU-time (s) & N-iter  & Qual &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
	print("		\hline")
	if True :
	   #---Compute a solution
	   Xmae,uxx,uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11_mae , V01.knots, V01.degree)
	   Ymae,uyx,uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12_mae , V10.knots, V10.degree)[0:3]

	   # ... Jacobian function of Optimal mapping
	   det = uxx*uyy-uxy*uyx
	   # ...
	   det_min          = np.min( det[1:-1,1:-1])
	   det_max          = np.max( det[1:-1,1:-1])
	   # ... scientific format
	   l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
	   MG_time          = round(start0, 3)
	   det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
	   det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
	   print("		",nelements, "&", MG_time, "&", n_ite+1, "&", l2_displacement, "&", det_min, "&", det_max,"\\\\")
	print("		\hline")
	print("	\end{tabular}")
	print('\n')



#--Solution of MAE equation (optimal mapping)
rho_Im              = pyccel_sol_field_2d((nbpts,nbpts),  x_rho, Vhp.knots, Vhp.degree)[0]
#rho_Im             = image_in_uniformpoints(X, Y, image_name).T
if Geometry is True :
	sX = pyccel_sol_field_2d( None, xmp , Vh.knots, Vh.degree, meshes = (Xmae, Ymae))[0]
	sY = pyccel_sol_field_2d( None, ymp , Vh.knots, Vh.degree, meshes = (Xmae, Ymae))[0]
	
for i in range(nbpts):
  for j in range(nbpts):
     if  ((det[i,j] < 0.) and (i * j * (i - nbpts +1) * (j - nbpts +1) != 0. )):
         print('Npoints =',nbpts,'min_Jac-F in the entire domain = ', det[i,j] ,'index =', i, j) 

#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
#-----adaptive mesh plot
#---------------------------------------------------------
fig =plt.figure() 
for i in range(nbpts):
   phidx = Xmae[:,i]
   phidy = Ymae[:,i]

   plt.plot(phidx, phidy, '-k', linewidth = 0.15)
for i in range(nbpts):
   phidx = Xmae[i,:]
   phidy = Ymae[i,:]

   plt.plot(phidx, phidy, '-k', linewidth = 0.15)
#plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
phidx = Xmae[:,0]
phidy = Ymae[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
# ...
phidx = Xmae[:,nbpts-1]
phidy = Ymae[:,nbpts-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
#''
phidx = Xmae[0,:]
phidy = Ymae[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
# ...
phidx = Xmae[nbpts-1,:]
phidy = Ymae[nbpts-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

#plt.xlim([-0.075,0.1])
#plt.ylim([-0.25,-0.1])
plt.axis('off')
plt.margins(0,0)
#fig.tight_layout()
plt.savefig('figs/'+str(image_name[8:11])+'_adaptive_meshes.png', bbox_inches='tight', pad_inches=0)	
plt.show(block=False)
plt.close()

levels = np.linspace(np.min(rho_Im),np.max(rho_Im), 100)
fig, axes =plt.subplots() 
im2 = plt.contourf(X, Y, rho_Im,levels, cmap= 'jet')
#divider = make_axes_locatable(axes) 
#cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
#plt.colorbar(im2, cax=cax) 
#fig.tight_layout()
axes.axis('off')
plt.savefig('figs/'+str(image_name[8:11])+'_density_function.png', bbox_inches='tight', pad_inches=0)
plt.show(block=False)
plt.close()

rho_Im              = pyccel_sol_field_2d((nbpts,nbpts),  x2_rho, Vhp.knots, Vhp.degree)[0]
levels = np.linspace(np.min(rho_Im),np.max(rho_Im), 100)
fig, axes =plt.subplots() 
im2 = plt.contourf(X, Y, rho_Im,levels, cmap= 'jet')
#divider = make_axes_locatable(axes) 
#cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
#plt.colorbar(im2, cax=cax) 
#fig.tight_layout()
axes.axis('off')
plt.savefig('figs/'+str(image_name2[8:11])+'_density_function.png', bbox_inches='tight', pad_inches=0)
plt.show(block=False)
plt.close()

levels = np.linspace(np.min(det),np.max(det), 100)
fig, axes =plt.subplots() 
im2 = plt.contourf(X, Y, det, levels, cmap= 'jet')
divider = make_axes_locatable(axes) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im2, cax=cax) 
fig.tight_layout()
plt.savefig('figs/'+str(image_name[8:11])+'_Jacobian_function.png')
plt.show(block=False)
plt.close()

X0, Y0   = np.meshgrid(np.linspace(0,1,nbpts), np.linspace(0,1,nbpts) )
#X0[:,:]  = Xmae[:,:]
#Y0[:,:]  = Ymae[:,:]
f        = zeros((nbpts, nbpts)) 
f[:,:]   = image_in_uniformpoints(X0, Y0, image_name2)
fig , axes=plt.subplots()
im2 = plt.contourf( Xmae, Ymae, f, np.linspace(np.min(f),np.max(f),100), cmap= 'gray')
#divider = make_axes_locatable(axes) 
#cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
#plt.colorbar(im2, cax=cax) 
#fig.tight_layout()
axes.axis('off')
plt.savefig('figs/'+str(image_name2[8:11])+'_deformed_image.png', bbox_inches='tight', pad_inches=0)
plt.show(block=False)
plt.close()

'''
from PIL import Image
an_image = Image.open(image_name)
fig, axes =plt.subplots() 
ima = plt.imshow(an_image)
plt.axis('off')
fig.tight_layout()
plt.savefig('Image_gen.png')
plt.show(block=False)
plt.close()
'''
if Geometry is True :
	fig =plt.figure() 
	for i in range(nbpts):
	   phidx = sX[:,i]
	   phidy = sY[:,i]

	   plt.plot(phidx, phidy, '-k', linewidth = 0.25)
	for i in range(nbpts):
	   phidx = sX[i,:]
	   phidy = sY[i,:]

	   plt.plot(phidx, phidy, '-k', linewidth = 0.25)
	#plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
	#~~~~~~~~~~~~~~~~~~~~
	#.. Plot the surface
	phidx = sX[:,0]
	phidy = sY[:,0]
	plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
	# ...
	phidx = sX[:,nbpts-1]
	phidy = sY[:,nbpts-1]
	plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
	#''
	phidx = sX[0,:]
	phidy = sY[0,:]
	plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
	# ...
	phidx = sX[nbpts-1,:]
	phidy = sY[nbpts-1,:]
	plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

	#plt.xlim([-0.075,0.1])
	#plt.ylim([-0.25,-0.1])
	plt.axis('off')
	plt.margins(0,0)
	fig.tight_layout()
	plt.savefig('figs/'+str(image_name[8:11])+'_Geom_adaptive_meshes.png')
	plt.show(block=False)
	plt.close()
	
	fig, axes =plt.subplots() 
	#levels = np.linspace(np.min(det),np.max(det), 150)
	levels = np.linspace(np.min(rho_Im),np.max(rho_Im), 100)
	plt.contourf( sX, sY, rho_Im, levels, cmap= 'gray')
	fig.tight_layout()
	plt.savefig('figs/r_refinement.png')
	plt.show(block=False)
	plt.close()
