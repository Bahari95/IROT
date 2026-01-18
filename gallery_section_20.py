__all__ = ['assemble_vector_ex0mae',
           'assemble_residual_ex0mae',
           ' assemble_residual_ex0grad'
]
#==============================================================================   
#---Assemble rhs of Mixed-BFO-Picard-Monge-Ampere equation
#==============================================================================
def assemble_vector_ex0mae(ne1:'int', ne2:'int', p1:'int', p2:'int', p3:'int', p4:'int', p5:'int', p6:'int', spans_1:'int[:]', spans_2:'int[:]',  spans_3:'int[:]', spans_4:'int[:]', spans_5:'int[:]', spans_6:'int[:]', basis_1:'float64[:,:,:,:]', basis_2:'float64[:,:,:,:]', basis_3:'float64[:,:,:,:]', basis_4:'float64[:,:,:,:]', basis_5:'float64[:,:,:,:]', basis_6:'float64[:,:,:,:]', weights_1:'float64[:,:]', weights_2:'float64[:,:]', points_1:'float64[:,:]', points_2:'float64[:,:]', vector_u:'float64[:,:]', vector_w:'float64[:,:]', vector_z:'float64[:,:]', spans_ad1:'int[:,:,:,:]', spans_ad2:'int[:,:,:,:]', basis_ad1:'double[:,:,:,:,:,:]', basis_ad2:'double[:,:,:,:,:,:]', rhs:'double[:,:]'):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    k5 = weights_1.shape[1]
    k6 = weights_2.shape[1]
    # ...
    lcoeffs_u  = zeros((p1+1,p3+1))
    lcoeffs_w  = zeros((p4+1,p2+1))
    lcoeffs_z  = zeros((p5+1,p6+1))
    
    lvalues_u  = zeros((k1, k2))

    #--Computes coefficient of ratio in MAE = int rho_1/int rho_0
    C_ratio = 0.0
    for ie1 in range(0, ne1):
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_6 = spans_6[ie2]
    
            lcoeffs_z[ : , : ]  =  vector_z[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            duh_k = 0.0 
            for g1 in range(0, k5):
                for g2 in range(0, k6):

                    wvol     = weights_1[ie1, g1]*weights_2[ie2, g2]
                    u_p        = 0.0
                    for il_1 in range(0, p5+1):
                          for il_2 in range(0, p6+1):
                          
                              coef_z = lcoeffs_z[il_1,il_2]
                              #...
                              bi_0   = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,0,g2]
                              #...
                              u_p   += coef_z*bi_0

                    C_ratio += wvol * abs(u_p)

    # ...
    int_rhsP    = 0.0
    # ...
    lvalues_u1x = zeros((k1, k2))
    lvalues_u1y = zeros((k1, k2))
    lvalues_u2x = zeros((k1, k2))
    lvalues_u2y = zeros((k1, k2))
    rho         = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            # ...
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol       = weights_1[ie1, g1]*weights_2[ie2, g2]

                    #... We compute firstly the span in new adapted points
                    span_5    = spans_ad1[ie1, ie2, g1, g2]
                    span_6    = spans_ad2[ie1, ie2, g1, g2]  

                    #------------------   
                    lcoeffs_z[ : , : ]  =  vector_z[span_5 : span_5+p5+1, span_6 : span_6+p6+1]
                    #------------------   
                    u_p        = 0.0
                    for il_1 in range(0, p5+1):
                          for il_2 in range(0, p6+1):

                              coef_z = lcoeffs_z[il_1,il_2]                                         
                              bi_0   = basis_ad1[ie1, ie2, il_1, 0, g1, g2]*basis_ad2[ie1, ie2, il_2, 0, g1, g2]
                              #...
                              u_p   += coef_z * bi_0
                              
                    rho[g1, g2] = (1+1.*C_ratio)/ (1+1.*abs(u_p) )

            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w[ : , : ]  = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            #...
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p3+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_3[ie2,il_2,0,g2]  #M^p2-1
                            db2  = basis_3[ie2,il_2,1,g2]  #M^p2-1

                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2

            lvalues_u2x[ : , : ] = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p2+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_4[ie1,il_1,0,g1] #M^p1-1
                        db1  = basis_4[ie1,il_1,1,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] 
                            db2  = basis_2[ie2,il_2,1,g2] 

                            lvalues_u2x[g1,g2] += coeff_w*db1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2
            lvalues_u[ : , : ] = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    #...
                    u1x              = lvalues_u1x[g1,g2]
                    u1y              = lvalues_u1y[g1,g2]
                    #___
                    u2x              = lvalues_u2x[g1,g2]
                    u2y              = lvalues_u2y[g1,g2]
                    # ...
                    lvalues_u[g1,g2] = sqrt(u1x**2 + u2y**2 + 2. * rho[g1, g2] + 2.*u1y**2)
                    # ...
                    wvol             = weights_1[ie1, g1]*weights_2[ie2, g2]
                    int_rhsP        += sqrt(u1x**2 + u2y**2 + 2. * rho[g1, g2] + 2.*u1y**2)*wvol 
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p3+1):
                    i1 = i_span_4 - p4 + il_1
                    i2 = i_span_3 - p3 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_4[ie1, il_1, 0, g1] * basis_3[ie2, il_2, 0, g2]
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            #...
                            u     = lvalues_u[g1,g2]
                            #...
                            v    += bi_0 * u * wvol

                    rhs[i1+p4,i2+p3] += v   
    # Integral in Neumann boundary
    int_N = 2.
    # Assuring Compatiblity condition
    coefs = int_N/int_rhsP  
    rhs   = rhs*coefs
    # ...
        
#==============================================================================   
def assemble_vector_rhsmae(ne1:'int', ne2:'int',  p1:'int', p2:'int', p3:'int', p4:'int', spans_1:'int[:]', spans_2:'int[:]', spans_3:'int[:]', spans_4:'int[:]', basis_1:'float[:,:,:,:]', basis_2:'float[:,:,:,:]', basis_3:'float[:,:,:,:]', basis_4:'float[:,:,:,:]', weights_1:'float[:,:]', weights_2:'float[:,:]', points_1:'float[:,:]', points_2:'float[:,:]', vector_w:'float[:,:]', rhs:'float[:,:]'):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1         = weights_1.shape[1]
    k2         = weights_2.shape[1]
    # ...
    lcoeffs_w  = zeros((p1+1,p2+1))
    lvalues_u  = zeros((k1, k2))
    # ...
    Sol_weith  = zeros((ne1, ne2, k1, k2))
    #-- Computes coefficient d'intensity
    uhk_min = 1e5
    uhk_max = 0.0
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
    
            rh_k               = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol       = weights_1[ie1, g1]*weights_2[ie2, g2]
                    x    =  points_1[ie1, g1]
                    y    =  points_2[ie2, g2]
                    ii     = ie1+ g1*ne1
                    jj     = ie2+ g2*ne2
                    #  ... arc-length
                    u_p    = vector_w[ii,jj]
                    rh_k   = abs(u_p) #*wvol
                    
                    #f abs(u_p) < 60. :
                    #    rh_k   = -1.
                    #if  abs(x-0.75) < 0.2 and abs(y-0.4) < 0.2 :
                    #     rh_k = 1.
                    #if sqrt((x-0.4)**2 + (y-0.4)**2 ) < 0.2 :
                    #   rh_k = 1.
                    Sol_weith[ie1,ie2, g1, g2] = rh_k
                    if uhk_max < rh_k :
                         uhk_max = rh_k 
                    if uhk_min > rh_k :
                         uhk_min = rh_k
                 
    # for ie1 in range(0, ne1):
    #    for ie2 in range(0, ne2):
    #        for g1 in range(0, k1):
    #            for g2 in range(0, k2):
    #                if Sol_weith[ie1,ie2, g1, g2] >= uhk_min+0.1*(uhk_min+uhk_max):
    #                     Sol_weith[ie1,ie2, g1, g2] = uhk_min+0.1*(uhk_min+uhk_max)
    # uhk_min = uhk_min+0.5*(uhk_min+uhk_max)
    if   (uhk_max-uhk_min) <= 1e-4  :
         int_uh_0 = 0.
         int_uh_1 = 1.
    else :
        int_uh_0 = (20.-0.5)/(uhk_max-uhk_min)
        int_uh_1  = (0.5*uhk_max-20.*uhk_min)/(uhk_max-uhk_min)
    # ... build rhs
    for ie3 in range(0, ne1):
        i_span_3 = spans_3[ie3]
        for ie4 in range(0, ne2):
            i_span_4 = spans_4[ie4]

            for il_1 in range(0, p3+1):
                for il_2 in range(0, p4+1):
                    i3 = i_span_3 - p3 + il_1
                    i4 = i_span_4 - p4 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            # ...
                            bi_0      = basis_3[ie3,il_1,0,g1]*basis_4[ie4,il_2,0,g2]
                            # ...
                            wvol      = weights_1[ie3, g1]*weights_2[ie4, g2] 

                            #...
                            v        += bi_0 *((int_uh_0 * Sol_weith[ie3,ie4, g1, g2] + int_uh_1) ) * wvol
                            #...
                    rhs[i3+p3,i4+p4] +=  v 
    # ... 

# Assembles Quality of mesh adaptation
#==============================================================================
def assemble_Quality_ex01(ne1:'int', ne2:'int',  p1:'int', p2:'int', p3:'int', p4:'int', spans_1:'int[:]', spans_2:'int[:]', spans_3:'int[:]', spans_4:'int[:]', basis_1:'float[:,:,:,:]', basis_2:'float[:,:,:,:]', basis_3:'float[:,:,:,:]', basis_4:'float[:,:,:,:]', weights_1:'float[:,:]', weights_2:'float[:,:]', points_1:'float[:,:]', points_2:'float[:,:]', vector_u:'float[:,:]', vector_w:'float[:,:]', times:'float', rhs:'float[:,:]'):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    
    # ... sizes
    k1           = weights_1.shape[1]
    k2           = weights_2.shape[1]
    # ...
    lcoeffs_u    = zeros((p1+1,p3+1))
    lcoeffs_w    = zeros((p4+1,p2+1))
    lvalues_u    = zeros((k1, k2))
    # ...
    lvalues_u1   = zeros((k1, k2))
    lvalues_u1x  = zeros((k1, k2))
    lvalues_u1y  = zeros((k1, k2))
    lvalues_u2   = zeros((k1, k2))
    #lvalues_u2x = zeros((k1, k2))
    lvalues_u2y  = zeros((k1, k2))

    # ..
    displacement = 0.
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            lvalues_u1[ : , : ]  = 0.0
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p3+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_3[ie2,il_2,0,g2]  #M^p2-1
                            db2  = basis_3[ie2,il_2,1,g2]  #M^p2-1

                            lvalues_u1[g1,g2]  += coeff_u*b1*b2
                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2
            lvalues_u2[ : , : ]  = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_w[ : , : ]   = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p2+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_4[ie1,il_1,0,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2                  = basis_2[ie2,il_2,0,g2] 
                            db2                 = basis_2[ie2,il_2,1,g2] 
                            lvalues_u2[g1,g2]  += coeff_w*b1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                    x    =  points_1[ie1, g1]
                    y    =  points_2[ie2, g2]

                    sx   = lvalues_u1[g1,g2]
                    sy   = lvalues_u2[g1,g2]
                    #.. Test 1
                    uhxx = lvalues_u1x[g1,g2]
                    uhyy = lvalues_u2y[g1,g2]
                    uhxy = lvalues_u1y[g1,g2]

                    w   += ((sx-x)**2+(sy-y)**2)/abs(uhxx*uhyy-uhxy**2) * wvol

            displacement += w
    rhs[p4,p3] = sqrt(displacement)
    # ...
