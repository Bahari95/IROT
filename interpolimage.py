"""
interpolimage.py

# Compute the image in quadrature points

@author : M. BAHARI
"""
# needed imports
from PIL import Image
import numpy as np
from PIL import Image

# importing bsplines utilities
from simplines import find_span, point_on_bspline_surface, insert_knot_bspline_surface, insert_knot_nurbs_surface

from pyccel.decorators import types
from pyccel.epyccel import epyccel
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
@types('int', 'int', 'int', 'int', 'float[:,:]', 'float[:,:]', 'float[:,:]', 'float[:]', 'float[:]', 'int', 'int', 'double[:,:,:]')
def sol_field_2D_meshes(ne1, ne2, k1, k2, xs, ys, uh, Tu, Tv, pu, pv, Q):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = zeros((nu, nv))
    
    for i in range(nu):  
       for j in range(nv):
             P[i, j] = uh[i, j]    

    nders      = 0
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    
    #...
    for i_i in range(0, ne1):
      for j_i in range(0, ne2):
        for g1_i in range(k1):
           for g2_j in range(k2):
              i_x =  i_i + g1_i*ne1
              j_y =  j_i + g2_j*ne2
              x   = xs[i_i, g1_i] 
              y   = ys[j_i, g2_j] 
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              basis_y = dersv
              span_v  = span             
              #...
              bu      = basis_x[0,:]
              bv      = basis_y[0,:]
              c       = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                      c  += bu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv]
              #..
              Q[i_x, j_y, 0]   = c
              
f90_sol_field_2d_meshes = epyccel(sol_field_2D_meshes, language = 'c')

#... Interpol the quadraturpoints
def image_in_quadraturpoints(points_1, points_2,  image_name, knots = None, degree = None):    

    if degree is not None :
       pu, pv = degree
    else :
       pu = 3
       pv = 3

    if knots is not None :
       Tu, Tv = knots
    else :
       nbu           = 128
       Tu            = np.zeros(nbu+2*pu)
       Tu[pu:nbu+pu] = np.linspace(0,1,nbu)
       Tu[-pu:]      = 1.
      
       nbv           = 128
       Tv            = np.zeros(nbv+2*pv)
       Tv[pv:nbv+pv] = np.linspace(0,1,nbv)
       Tv[-pv:]      = 1.

    # ...
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = np.zeros((nu, nv,1))
    
    #---SOME OPERATIONS IN iMAGE REGUISTRATION --- ++++ +++
    # Image.open() can also open other image types
    img = Image.open( image_name)

    # WIDTH and HEIGHT are integers
    resized_img = img.resize((nu, nv))

    image_sequence = resized_img.getdata()

    image_array = np.array(image_sequence)
    if image_array.ndim != 1 :
       image_array = np.array(image_sequence)[:,0]

    K = np.zeros((nu, nv,1))
    K[:,:,0] = image_array.reshape((nu, nv))
    P = np.zeros((nu,nv))
    for i in range(nu):
       P[i,:] = K[nu-i-1,:,0]
    # ....
    ne1, k1 = points_1.shape
    ne2, k2 = points_2.shape

    nx = ne1*k1
    ny = ne2*k2
    
    Q = np.zeros((nx, ny, 1))
    f90_sol_field_2d_meshes(ne1, ne2, k1, k2, points_1, points_2, P, Tu, Tv, pu, pv, Q)
    '''
    for ie1 in range(0, ne1):
        for ie2 in range(0, ne2):
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                     x        =  points_1[ie1, g1]
                     y        =  points_2[ie2, g2]
                     i        =  ie1 + g1*ne1
                     j        =  ie2 + g2*ne2
                     Q[i,j,:] = point_on_bspline_surface(Tu, Tv, P, x, y)
    '''
    return Q[:,:,0]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
@types('int', 'int', 'float[:,:]', 'float[:]', 'float[:]', 'int', 'int', 'double[:,:,:]')
def sol_field_2D_un_meshes(nx, ny, uh, Tu, Tv, pu, pv, Q):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = zeros((nu, nv))
    
    for i in range(nu):  
       for j in range(nv):
             P[i, j] = uh[i, j]    

    nders      = 0
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    
    #...
    for i_x in range(0, nx):
         for j_y in range(0, ny):
              x   = Q[i_x, j_y, 1] 
              y   = Q[i_x, j_y, 2] 
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              basis_y = dersv
              span_v  = span             
              #...
              bu      = basis_x[0,:]
              bv      = basis_y[0,:]
              c       = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                      c  += bu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv]
              #..
              Q[i_x, j_y, 0]   = c
              
f90_sol_field_2d_uni_meshes = epyccel(sol_field_2D_un_meshes, language = 'c')

#... Interpol the image in uniform mesh
def image_in_uniformpoints(points_1, points_2, image_name, knots = None, degree = None):    

    if degree is not None :
       pu, pv = degree
    else :
       pu = 3
       pv = 3

    if knots is not None :
       Tu, Tv = knots
    else :
       nbu           = 128
       Tu            = np.zeros(nbu+2*pu)
       Tu[pu:nbu+pu] = np.linspace(0,1,nbu)
       Tu[-pu:]      = 1.
      
       nbv           = 128
       Tv            = np.zeros(nbv+2*pv)
       Tv[pv:nbv+pv] = np.linspace(0,1,nbv)
       Tv[-pv:]      = 1.

    # ...
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    #---SOME OPERATIONS IN iMAGE REGUISTRATION -- -++++ +++
    # Image.open() can also open other image types
    img = Image.open(image_name)
    # WIDTH and HEIGHT are integers
    resized_img = img.resize((nu, nv))

    image_sequence = resized_img.getdata()

    image_array = np.array(image_sequence)
    if image_array.ndim != 1 :
       image_array = np.array(image_sequence)[:,0]

    K = np.zeros((nu, nv,1))
    K[:,:,0] = image_array.reshape((nu, nv))
    P = np.zeros((nu,nv))
    for i in range(nu):
       P[i,:] = K[nu-i-1,:,0]
    # ....
    ne1, ne2 = points_1.shape

    Q = np.zeros((ne1, ne2, 3))
    Q[:,:,1] = points_1
    Q[:,:,2] = points_2
    f90_sol_field_2d_uni_meshes(ne1, ne2, P, Tu, Tv, pu, pv, Q)
    '''
    for ie1 in range(0, ne1):
        for ie2 in range(0, ne2):
                     x        =  points_1[ie1, ie2]
                     y        =  points_2[ie1, ie2]
                     Q[ie1, ie2,:] = point_on_bspline_surface(Tu, Tv, P, x, y)
    '''       
    return Q[:,:,0]

