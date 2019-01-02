# to solve N.S. equations

import numpy as np
import matplotlib.pyplot as plt

# domain size, resolution in space and time
Lx = 50 # computational domain size in x direction
Ly = 50 # computational domain size in y direction
dt = 0.1 # step in time
dx = 1 # step in x direction
dy = 1 # step in y direction
Dx = int(Lx*0.5) # block size in x direction
Dy = 2 #int(Ly*0.05) # block size in y direction
Hx = int(Lx*0.2) # block starting position in x direction
Hy1 = int(Ly*0.3) # first block starting position in y direction starting from below
Hy2 = int(Ly*0.7) # second block starting position in y direction starting from below

# nondimensionlizing the problem: 
# length Lx, velocity U0, time Lx/U0, density with reference rho0, pressure: rho0 U0^2
# eq of state becomes: p=rho/M^2, where M=U0/c, Re=rho0 U0 Lx / nu
mu = 1000 # kg/s/m, nu m^2/s, nu = mu/rho
rho0 = 1000 # kg/m^3
U0 = 2 # m/s
Re = rho0*U0*Lx/mu
M = 0.1 # set to 0.1 to keep flow incompressible, U0/c, wherer c is the speed of sound

# variables - velocity, density, current density
rho = np.zeros( (Lx+1, Ly+1), float ) # density
u = np.zeros( (Lx+1, Ly+1), float ) # x component of the velocity
v = np.zeros( (Lx+1, Ly+1), float ) # y component of the velocity
ru = np.zeros( (Lx+1, Ly+1), float ) # x component of the current density
rv = np.zeros( (Lx+1, Ly+1), float ) # y component of the current density

# temporal variable
rho_star = np.zeros( (Lx+1, Ly+1), float ) # density at "half step"
ru_star = np.zeros( (Lx+1, Ly+1), float ) # x component of the current density at "half step"
rv_star = np.zeros( (Lx+1, Ly+1), float ) # y component of the current density at "half step"
u_star = np.zeros( (Lx+1, Ly+1), float ) # x component of the velocity at "half step"
v_star = np.zeros( (Lx+1, Ly+1), float ) # y component of the velocity at "half step"

top_left = np.zeros( 2, float ) # top-left corner element
top_right = np.zeros( 2, float ) # top-rigt corner element
bottom_left = np.zeros( 2, float ) # bottom-left corner element
bottom_right = np.zeros( 2, float ) # bottom-right corner element

def initial_field():
    rho[:,:] = rho0 # set initial density
    u[:,:] = U0 # set initial velocity in x direction
    v[:,:] = 0 # set initial velocity in y direction
    # set the velocity components to zero inside the blocks
    u[Hx:Hx+Dx, Hy1:Hy1+Dy] = 0
    u[Hx:Hx+Dx, Hy2:Hy2+Dy] = 0
    v[Hx:Hx+Dx, Hy1:Hy1+Dy] = 0
    v[Hx:Hx+Dx, Hy2:Hy2+Dy] = 0
    # set the density to zero inside the blocks
    rho[Hx:Hx+Dx, Hy1:Hy1+Dy] = 0
    rho[Hx:Hx+Dx, Hy2:Hy2+Dy] = 0
    # set the initial current densities
    ru[:,:] = rho0*u[:,:] # set initial current density in x direction
    rv[:,:] = rho0*v[:,:] # set initial current density in y direction

# -------------------------------------------------------------------------

def boundary_at_top( j, rho_in, rho_out, rv_in, ru_out, rv_out ): # j to specify last row, so Ly
    for i in range( 1, Lx ): # leaving out corner points
        rho_out[i,j] = rho_in[i,j]-U0*dt/2/dx*( rho_in[i+1,j]-rho_in[i-1,j] )+dt/2/dy*( -1*rv_in[i,j-2]+4*rv_in[i,j-1]-3*rv_in[i,j] ) 
        ru_out[i,j] = rho_out[i,j]*U0
        rv_out[i,j] = 0

def boundary_at_bottom( j, rho_in, rho_out, rv_in, ru_out, rv_out ): # j to specify first row, so 0
    for i in range( 1, Lx ): # leaving out corner points
        rho_out[i,j] = rho_in[i,j]-U0*dt/2/dx*( rho_in[i+1,j]-rho_in[i-1,j] )-dt/2/dy*( -1*rv_in[i,j+2]+4*rv_in[i,j+1]-3*rv_in[i,j] ) 
        ru_out[i,j] = rho_out[i,j]*U0
        rv_out[i,j] = 0

def boundary_at_left( i, rho_in, rho_out, rv_in, ru_out, rv_out ): # i to specify first column, so 0
    for j in range( 1, Ly ): # leaving out corner points
        #rho_out[i,j] = rho_in[i,j]-U0*dt/2/dx*( rho_in[i+2,j]+4*rho_in[i+1,j]-3*rho_in[i,j] )-dt/2/dy*( rv_in[i,j+1]-rv_in[i,j-1] ) 
        rho_out[i,j] = rho0 # inlet density specified (!)
        ru_out[i,j] = rho_out[i,j]*U0
        rv_out[i,j] = 0

def boundary_at_right( i, rho_in, u_in, v_in, rho_out, ru_out, rv_out ): # i to specify last column, so Lx
    for j in range( 1, Ly ): # leaving out corner points
        rho_out[i,j] = rho_in[i,j]
        ru_out[i,j] = rho_out[i,j]*u_in[i,j]
        rv_out[i,j] = rho_out[i,j]*v_in[i,j]

def boundary_at_corner( u_in, v_in, rho_out, ru_out, rv_out ):
    rho_out[0,0] = 0.5*(rho_out[1,0]+rho_out[0,1])
    rho_out[Lx,0] = 0.5*(rho_out[Lx-1,0]+rho_out[Lx,1])
    rho_out[0,Ly] = 0.5*(rho_out[1,Ly]+rho_out[0,Ly-1])
    rho_out[Lx,Ly] = 0.5*(rho_out[Lx-1,Ly]+rho_out[Lx,Ly-1])
    ru_out[0,0] = rho_out[0,0]*U0
    ru_out[Lx,0] = rho_out[Lx,0]*u_in[Lx,0]
    ru_out[0,Ly] = rho_out[0,Ly]*U0
    ru_out[Lx,Ly] = rho_out[Lx,Ly]*u_in[Lx,Ly]
    rv_out[0,0] = 0
    rv_out[Lx,0] = rho_out[Lx,0]*v_in[Lx,0]
    rv_out[0,Ly] = 0
    rv_out[Lx,Ly] = rho_out[Lx,Ly]*v_in[Lx,Ly]

#--------------------------------------------------------------------------

def block_top( j, rho_in, u_in, v_in ): # j to specify top position
    a1=8/9/dy*M**2/Re
    a2=1/18/dx*M**2/Re
    for i in range( Hx, Hx+Dx ):
        rho_in[i,j] = 1/3*(4*rho_in[i, j+1]-rho_in[i,j+2])-a1*(-5*v_in[i,j+1]+4*v_in[i,j+2]-v_in[i,j+3] )-a2*( -(u_in[i+1,j+2]-u_in[i-1,j+2])+4*(u_in[i+1,j+1]-u_in[i-1,j+1])-3*(u_in[i+1,j]-u_in[i-1,j]) )     
        u_in[i,j] = 0
        v_in[i,j] = 0
#        print("rho_top:", "i:", i, "j:", j, rho_in[i,j])
        if i == Hx:
            top_left[0] = rho_in[i,j]
        if i == Hx+Dx-1:
            top_right[0] = rho_in[i,j]

def block_bottom( j, rho_in, u_in, v_in ): # j to specify bottom position
    a1=8/9/dy*M**2/Re
    a2=1/18/dx*M**2/Re
    for i in range( Hx, Hx+Dx ): 
        rho_in[i,j] = 1/3*(4*rho_in[i, j-1]-rho_in[i,j-2])+a1*(-5*v_in[i,j-1]+4*v_in[i,j-2]-v_in[i,j-3] )-a2*( -(u_in[i+1,j-2]-u_in[i-1,j-2])+4*(u_in[i+1,j-1]-u_in[i-1,j-1])-3*(u_in[i+1,j]-u_in[i-1,j]) )     
        u_in[i,j] = 0
        v_in[i,j] = 0
#        print("rho_bottom:", "i:", i, "j:", j, rho_in[i,j])
        if i == Hx:
            bottom_left[0] = rho_in[i,j]
        if i == Hx+Dx-1:
            bottom_right[0] = rho_in[i,j]

def block_left( i, j1, j2, rho_in, u_in, v_in ): # i to specify left position, j1, j2 for block height
    a1=8/9/dx*M**2/Re
    a2=1/18/dy*M**2/Re
    for j in range( j1, j2 ):
        rho_in[i,j] = 1/3*(4*rho_in[i-1,j]-rho_in[i-2,j])+a1*(-5*u_in[i-1,j]+4*u_in[i-2,j]-u_in[i-3,j] )-a2*( -(v_in[i-2,j+1]-v_in[i-2,j-1])+4*(v_in[i-1,j+1]-v_in[i-1,j-1])-3*(v_in[i,j+1]-v_in[i,j-1]) )     
        u_in[i,j] = 0
        v_in[i,j] = 0
#        print("rho_left:", "i:", i, "j:", j, rho_in[i,j])
        if j == j1:
            bottom_left[1] = rho_in[i,j]
        if j == j2-1:
            top_left[1] = rho_in[i,j]

def block_right( i, j1, j2, rho_in, u_in, v_in ): # i to specify right position, j1, j2 for block height
    a1=8/9/dx*M**2/Re
    a2=1/18/dy*M**2/Re
    for j in range( j1, j2 ):
        rho_in[i,j] = 1/3*(4*rho_in[i+1,j]-rho_in[i+2,j])-a1*(-5*u_in[i+1,j]+4*u_in[i+2,j]-u_in[i+3,j] )-a2*( -(v_in[i+2,j+1]-v_in[i+2,j-1])+4*(v_in[i+1,j+1]-v_in[i+1,j-1])-3*(v_in[i,j+1]-v_in[i,j-1]) )     
        u_in[i,j] = 0
        v_in[i,j] = 0
#        print("rho_right:", "i:", i, "j:", j, rho_in[i,j])
        if j == j1:
            bottom_right[1] = rho_in[i,j]
        if j == j2-1:
            top_right[1] = rho_in[i,j]

def block_corner( j1, j2, rho_in ): # specify j1, j2 for block height
    rho_in[Hx,j1] = 0.5*(bottom_left[0]+bottom_left[1])
    rho_in[Hx,j2] = 0.5*(top_left[0]+top_left[1])
    rho_in[Hx+Dx-1,j1] = 0.5*(bottom_right[0]+bottom_right[1])
    rho_in[Hx+Dx-1,j2] = 0.5*(top_right[0]+top_right[1])
#    print("i:", Hx, "j:", j1, rho_in[Hx,j1])
#    print("i:", Hx, "j:", j2, rho_in[Hx,j2])
#    print("i:", Hx+Dx-1, "j:", j1, rho_in[Hx+Dx-1,j1])
#    print("i:", Hx+Dx-1, "j:", j2, rho_in[Hx+Dx-1,j2])

#--------------------------------------------------------------------------

def calc_velocity( rho_in, ru_in, rv_in, u_out, v_out ):
    for i in range(0, Lx+1):
        for j in range( 0, Ly+1):
            if ( np.abs(rho_in[i,j]) > 10):
                u_out[i,j] = ru_in[i,j] / rho_in[i,j]
                v_out[i,j] = rv_in[i,j] / rho_in[i,j]
            else:
                u_out[i,j] = 0
                v_out[i,j] = 0

def calc_equations_FF( rho_in, u_in, v_in, ru_in, rv_in, rho_out, ru_out, rv_out ):
    a1 = dt/dx
    a2 = dt/dy
    a3 = dt/dx/M**2
    a4 = dt/dy/M**2
    a5 = 4*dt/3/Re/dx**2
    a6 = dt/Re/dy**2
    a7 = dt/Re/dx**2
    a8 = 4*dt/3/Re/dy**2
    a9 = dt/12/Re/dx/dy
    a10 = 2*(a5+a6)
    a11 = 2*(a7+a8)
    k = 0
    for i in range( 1, Lx ):
        for j in range( 1, Ly ):
            if( i > Hx-1 and i < Hx+Dx ):
                k = 1
            else:
                k = 0
            if ( k and (j > Hy1-1 and j < Hy1+Dy) ) or ( k and (j > Hy2-1 and j < Hy2+Dy) ):
                continue
            # continuity:
            rho_out[i,j] = rho_in[i,j]-a1*(ru_in[i+1,j]-ru_in[i,j])-a2*(rv_in[i,j+1]-rv_in[i,j])
#            print("i:", i, "j:", j, rho_out[i,j])
            # momentum for x component:
            ru_out[i,j] = ru_in[i,j]-a3*(rho_in[i+1,j]-rho_in[i,j])-a1*( ru_in[i+1,j]*u_in[i+1,j]-ru_in[i,j]*u_in[i,j] )
            ru_out[i,j] -= a2*( ru_in[i,j+1]*v_in[i,j+1]-ru_in[i,j]*v_in[i,j] ) - a10*u_in[i,j] + a5*( u_in[i+1,j]+u_in[i-1,j] )
            ru_out[i,j] += a6*( u_in[i,j+1]+u_in[i,j-1] ) + a9*( v_in[i+1,j+1]+v_in[i-1,j-1]-v_in[i+1,j-1]-v_in[i-1,j+1] )
#            if ( ru_out[i,j] > 100*ru_out[i,j-1]):
#                ru_out[i,j] = ru_out[i,j-1]
            # momentum for y component:
            rv_out[i,j] = rv_in[i,j]-a4*(rho_in[i,j+1]-rho_in[i,j])-a2*( rv_in[i,j+1]*v_in[i,j+1]-rv_in[i,j]*v_in[i,j] )
            rv_out[i,j] -= a1*( ru_in[i+1,j]*v_in[i+1,j]-ru_in[i,j]*v_in[i,j] ) - a11*v_in[i,j] + a7*( v_in[i+1,j]+v_in[i-1,j] )
            rv_out[i,j] += a8*( v_in[i,j+1]+v_in[i,j-1] ) + a9*( u_in[i+1,j+1]+u_in[i-1,j-1]-u_in[i+1,j-1]-u_in[i-1,j+1] )
#            if ( rv_out[i,j] > 100*rv_out[i,j-1]):
#                rv_out[i,j] = rv_out[i,j-1]

def calc_equations_BB( rho_in, u_in, v_in, ru_in, rv_in, rho_out, ru_out, rv_out ):
    a1 = dt/dx
    a2 = dt/dy
    a3 = dt/dx/M**2
    a4 = dt/dy/M**2
    a5 = 4*dt/3/Re/dx**2
    a6 = dt/Re/dy**2
    a7 = dt/Re/dx**2
    a8 = 4*dt/3/Re/dy**2
    a9 = dt/12/Re/dx/dy
    a10 = 2*(a5+a6)
    a11 = 2*(a7+a8)
    k = 0
    for i in range( Lx-1, 0, -1 ):
        for j in range( Ly-1, 0, -1 ):
            if( i > Hx-1 and i < Hx+Dx ):
                k = 1
            else:
                k = 0
            if ( k and (j == Hy2+Dy-1) ) or ( k and (j == Hy1+Dy-1) ):
                continue
            if ( k and (j == Hy2+Dy-2) ) or ( k and (j == Hy1+Dy-2) ):
                continue
            # continuity:
            rho_out[i,j] = 0.5*( rho[i,j] + rho_in[i,j]-a1*(ru_in[i,j]-ru_in[i-1,j])-a2*(rv_in[i,j]-rv_in[i,j-1]) )
#            print("i:", i, "j:", j, rho_out[i,j])
            # momentum for x component:
            ru_out[i,j] = ru[i,j] + ru_in[i,j]-a3*(rho_in[i,j]-rho_in[i-1,j])-a1*( ru_in[i,j]*u_in[i,j]-ru_in[i-1,j]*u_in[i-1,j] )
            ru_out[i,j] -= a2*( ru_in[i,j]*v_in[i,j]-ru_in[i,j-1]*v_in[i,j-1] ) - a10*u_in[i,j] + a5*( u_in[i+1,j]+u_in[i-1,j] )
            ru_out[i,j] += a6*( u_in[i,j+1]+u_in[i,j-1] ) + a9*( v_in[i+1,j+1]+v_in[i-1,j-1]-v_in[i+1,j-1]-v_in[i-1,j+1] )
            ru_out[i,j] *= 0.5
#            if ( ru_out[i,j] > 100*ru_out[i,j-1]):
#                ru_out[i,j] = ru_out[i,j-1]
            # momentum for y component:
            rv_out[i,j] = rv[i,j] + rv_in[i,j]-a4*(rho_in[i,j]-rho_in[i,j-1])-a2*( rv_in[i,j]*v_in[i,j]-rv_in[i,j-1]*v_in[i,j-1] )
            rv_out[i,j] -= a1*( ru_in[i,j]*v_in[i,j]-ru_in[i-1,j]*v_in[i-1,j] ) - a11*v_in[i,j] + a7*( v_in[i+1,j]+v_in[i-1,j] )
            rv_out[i,j] += a8*( v_in[i,j+1]+v_in[i,j-1] ) + a9*( u_in[i+1,j+1]+u_in[i-1,j-1]-u_in[i+1,j-1]-u_in[i-1,j+1] )
            rv_out[i,j] *= 0.5
#            if ( rv_out[i,j] > 100*rv_out[i,j-1]):
#                rv_out[i,j] = rv_out[i,j-1]

# -------------------------------------------------------------------------

# boundary conditions star
def calc_boundaries_star():
    outer_boundaries_star()
    inner_boundaries_star()

def outer_boundaries_star():
    boundary_at_top(Ly, rho, rho_star, rv, ru_star, rv_star) # specify last row, so Ly
    boundary_at_bottom(0, rho, rho_star, rv, ru_star, rv_star) # specify first row, so 0
    boundary_at_left(0, rho, rho_star, rv, ru_star, rv_star) # specify first column, so 0
    boundary_at_right(Lx, rho, u, v, rho_star, ru_star, rv_star) # pecify last column, so Lx
    boundary_at_corner( u, v, rho_star, ru_star, rv_star )

def inner_boundaries_star():
    block_bottom(Hy1, rho_star, u_star, v_star)
    block_top(Hy1+Dy-1, rho_star, u_star, v_star)
    block_left(Hx, Hy1, Hy1+Dy, rho_star, u_star, v_star)
    block_right(Hx+Dx-1, Hy1, Hy1+Dy, rho_star, u_star, v_star)
    block_corner(Hy1, Hy1+Dy-1, rho_star)
    
    block_bottom(Hy2, rho_star, u_star, v_star)
    block_top(Hy2+Dy-1, rho_star, u_star, v_star)
    block_left(Hx, Hy2, Hy2+Dy, rho_star, u_star, v_star)
    block_right(Hx+Dx-1, Hy2, Hy2+Dy, rho_star, u_star, v_star)
    block_corner(Hy2, Hy2+Dy-1, rho_star)

# boundary conditions next
def calc_boundaries_next():
    outer_boundaries_next()
    inner_boundaries_next()

def outer_boundaries_next():
    boundary_at_top(Ly, rho_star, rho, rv_star, ru, rv) # specify last row, so Ly
    boundary_at_bottom(0, rho_star, rho, rv_star, ru, rv) # specify first row, so 0
    boundary_at_left(0, rho_star, rho, rv_star, ru, rv) # specify first column, so 0
    boundary_at_right(Lx, rho_star, u_star, v_star, rho, ru, rv) # pecify last column, so Lx
    boundary_at_corner(u_star, v_star, rho, ru, rv )

def inner_boundaries_next():
    block_bottom(Hy1, rho, u, v)
    block_top(Hy1+Dy-1, rho, u, v)
    block_left(Hx, Hy1, Hy1+Dy, rho, u, v)
    block_right(Hx+Dx-1, Hy1, Hy1+Dy, rho, u, v)
    block_corner(Hy1, Hy1+Dy-1, rho)
    
    block_bottom(Hy2, rho, u, v)
    block_top(Hy2+Dy-1, rho, u, v)
    block_left(Hx, Hy2, Hy2+Dy, rho, u, v)
    block_right(Hx+Dx-1, Hy2, Hy2+Dy, rho, u, v)
    block_corner(Hy2, Hy2+Dy-1, rho)

# boundary conditions first
def calc_boundaries_first():
    outer_boundaries_first()
    inner_boundaries_first()

def outer_boundaries_first():
    boundary_at_top(Ly, rho, rho, rv, ru, rv) # specify last row, so Ly
    boundary_at_bottom(0, rho, rho, rv, ru, rv) # specify first row, so 0
    boundary_at_left(0, rho, rho, rv, ru, rv) # specify first column, so 0
    boundary_at_right(Lx, rho, u, v, rho, ru, rv) # pecify last column, so Lx
    boundary_at_corner( u, v, rho, ru, rv )

def inner_boundaries_first():
    block_bottom(Hy1, rho, u, v)
    block_top(Hy1+Dy-1, rho, u, v)
    block_left(Hx, Hy1, Hy1+Dy, rho, u, v)
    block_right(Hx+Dx-1, Hy1, Hy1+Dy, rho, u, v)
    block_corner(Hy1, Hy1+Dy-1, rho)
    
    block_bottom(Hy2, rho, u, v)
    block_top(Hy2+Dy-1, rho, u, v)
    block_left(Hx, Hy2, Hy2+Dy, rho, u, v)
    block_right(Hx+Dx-1, Hy2, Hy2+Dy, rho, u, v)
    block_corner(Hy2, Hy2+Dy-1, rho)

def MacCormackScheme():
    calc_velocity( rho, ru, rv, u, v ) # calculate velocity components over all points
    calc_equations_FF( rho, u, v, ru, rv, rho_star, ru_star, rv_star ) # calculate rho_star, ru_star, rv_star
    calc_boundaries_star()
    calc_velocity( rho_star, ru_star, rv_star, u_star, v_star ) # calculate star velocity components over all points
    calc_equations_BB( rho_star, u_star, v_star, ru_star, rv_star, rho, ru, rv ) # calculate rho_star, ru_star, rv_star
    calc_boundaries_next()

# running the code
initial_field()
print("initial")
calc_boundaries_first()
for k in range( 0, 1):
    print("running")
    MacCormackScheme()

#while blabla
#MacCormackScheme()

plt.pcolor( rho.T )
plt.show

# Output to file - velocity and density distribution

