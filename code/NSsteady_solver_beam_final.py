# to solve N.S. equation for stacionary case

import numpy as np
import matplotlib.pylab as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Nxmax = 200
Nymax = 100
IL = 50 # starting position of the beam in x direction
H = 50 # height of the beam
T = 50 # lenght of the beam in x direction
h = 1
u = np.zeros( (Nxmax+1, Nymax+1), float ) # stream
w = np.zeros( (Nxmax+1, Nymax+1), float ) # vorticity
V0 = 1.0 # initial velocity field
omega = 0.005 # relaxation parameter
nu = 0.25 # viscosity
it = 0
R = V0*h/nu # Reynolds number
vx = np.zeros( (Nxmax+1, Nymax+1), float ) # velocity in x direction
vy = np.zeros( (Nxmax+1, Nymax+1), float ) # velocity in y direction

maxit = 5000
u1_top =  np.zeros( (maxit), float )
u1_front =  np.zeros( (maxit), float )
u1_back =  np.zeros( (maxit), float )
w2_top =  np.zeros( (maxit), float )
w2_front =  np.zeros( (maxit), float )
w2_back =  np.zeros( (maxit), float )

def borders(): # initialize and applying boundary conditions
    for i in range( 0, Nxmax+1 ): # initialize stream function
        for j in range( 0, Nymax+1 ): # and vorticity
            w[i,j] = 0.
            u[i,j] = j*V0
    for i in range( 0, Nxmax+1): # fluid surface
        u[i, Nymax] = u[i, Nymax-1] + V0*h
        w[i, Nymax-1] = 0.                                  # ez miért Nymax-1?
    for j in range( 0, Nymax+1 ): # inlet
        u[1, j] = u[0, j]                                   # ez miért 1?
        w[0, j] = 0
    for i in range( 0, Nxmax+1 ): # centerline
        if i <= IL and i >= IL + T:                         # kihagyja a beam alját!
            u[i, 0] = 0.
            w[i, 0] = 0.
    for j in range( 1, Nymax ): # outlet                    # kihagyja a sarkokat!
        w[Nxmax, j] = w[Nxmax-1, j]
        u[Nxmax, j] = u[Nxmax-1, j] # boundary conditions

def beam(): # method for beam; boundary conditions for beam: beam sides
    for j in range( 0, H+1 ):
        w[IL, j] = -2*u[IL-1, j] / (h*h) # front side # w[IL, j] = -2*( u[IL-1, j]-u[IL, j] ) / (h*h) # front side
        w[IL+T, j] = -2*u[IL+T+1, j] / (h*h) # back side # w[IL+T, j] = -2*( u[IL+T+1, j]-u[IL+T, j] ) / (h*h) # back side
    for i in range( IL, IL+T+1 ):
        w[i, H-1] = -2*u[i, H] / (h*h) # top side
    for i in range( IL, IL+T+1 ):
        for j in range( 0, H+1 ):
            u[IL, j] = 0. # front
            u[IL+T, j] = 0. # back
            u[i, H] = 0. # top
    u[IL+1:IL+T-1,0:H] = 0 # stream at the beam
    w[IL+1:IL+T-1,0:H] = 0 # vorticity at the beam

def relax(it): # method to relax system
    beam() # reset conditions at beam
    for i in range( 1, Nxmax ):
        for j in range( 1, Nymax ):
            r1 = omega*( ( u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]+h*h*w[i,j] )*0.25 - u[i,j] )
            u[i,j] += r1
            if ( i == IL+int(T*0.5) and j == H+3): # check conv at middle of beam top
                u1_top[it] = u[i,j]
            if ( i == IL-3 and j == int(H*0.6667) ): # check conv at beam front at 2/3 height
                u1_front[it] = u[i,j]
            if ( i == IL+T+3 and j == int(H*0.5) ): # check conv at beam back at half height
                u1_back[it] = u[i,j]
    for i in range( 1, Nxmax ):
        for j in range( 1, Nymax ):
            a1 = w[i+1, j]+w[i-1, j]+w[i,j+1]+w[i,j-1]
            a2 = ( u[i,j+1]-u[i,j-1] ) * ( w[i+1,j]-w[i-1,j] )
            a3 = ( u[i+1,j]-u[i-1,j] ) * ( w[i,j+1]-w[i,j-1] )
            r2 = omega*( ( a1-(R/4.)*(a2-a3) ) / 4.0 - w[i,j] )
            w[i,j] += r2
            if ( i == IL+int(T*0.5) and j == H+3): # check conv at middle of beam top
                w2_top[it] = w[i,j]
            if ( i == IL-3 and j == int(H*0.6667) ): # check conv at beam front at 2/3 height
                w2_front[it] = w[i,j]
            if ( i == IL+T+3 and j == int(H*0.5) ): # check conv at beam back at half height
                w2_back[it] = w[i,j]

borders()
while ( it < maxit ):
    relax(it)
    if it%10 == 0:
        print( it )
    it += 1
for i in range( 0, Nxmax+1 ):
    for j in range( 0, Nymax+1 ):
        u[i,j] = u[i,j] / (V0*h) # V0 h units

x = range( 0, Nxmax+1 )
y = range( 0, Nymax+1 )
X, Y = p.meshgrid(x, y)

def functz( u ): # returns stream flow to plot
    z = u[X, Y] # for several iterations
    return z

# calculate velocity of stream function:
def velocity_field( u ):
    for i in range( 1, Nxmax-1 ):
        for j in range( 1, Nymax-1 ):
            vx[i,j] = 1/(2*h)*( u[i,j+1]-u[i,j-1] )
            vy[i,j] = -1/(2*h)*( u[i+1,j]-u[i-1,j] )

str1 = str(omega/10)
str1 = str1[2:]
str2 = str(maxit)
str3 = str(R*10)
str3 = str3[:-2]
str4 = str(Nxmax)
str5 = str(Nymax)

velocity_field(u)
fig0 = plt.figure( num=2, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k' )
ax0 = fig0.add_subplot(111)
ax0.set_xlabel('X', fontsize=20 )
ax0.set_ylabel('Y', fontsize=20 )
ax0.set_title('Streamplot of the velocity field', fontsize=20)
ax0.set_xlim([-3,Nxmax+3])
ax0.set_ylim([-3,Nymax+3])
plt.streamplot(X,Y, vx.T,vy.T, density=2.5,zorder=5)
plt.savefig('figures/velocity_field_w'+str1+'_it'+str2+'_R'+str3+'_Nx'+str4+'_Ny'+str5+'_.png', dpi=200)

Z = functz( u ) # here the function is called
fig1 = plt.figure( num=2, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k' )
ax1 = Axes3D( fig1 )
ax1.plot_wireframe( X, Y, Z, color = 'r' ) # surface of wireframe in red
ax1.set_xlabel('X', fontsize=20)
ax1.set_ylabel('Y', fontsize=20)
ax1.set_zlabel('Stream function', fontsize=20)
ax1.set_title('Surface plot of the stream function', fontsize=20)
plt.savefig('figures/stream_surf_w'+str1+'_it'+str2+'_R'+str3+'_Nx'+str4+'_Ny'+str5+'_.png', dpi=200)

Z2 = functz( w ) # here the function is called
fig2 = plt.figure( num=2, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k') # creates the figure
ax2 = Axes3D( fig2 ) # plots the axis for the figure
ax2.plot_wireframe( X, Y, Z2, color = 'b' ) # surface of wireframe in blue
ax2.set_xlabel('X', fontsize=20)
ax2.set_ylabel('Y', fontsize=20)
ax2.set_zlabel('Vorticity', fontsize=20)
ax2.set_title('Surface plot of the vorticity', fontsize=20)
plt.savefig('figures/vorticity_surf_w'+str1+'_it'+str2+'_R'+str3+'_Nx'+str4+'_Ny'+str5+'_.png', dpi=200)

itnum = np.linspace(1, maxit, maxit)

fig3 = plt.figure( num=1, figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k' )
ax3 = fig3.add_subplot(111)
ax3.plot( itnum, u1_front, 'b-')
ax3.plot( itnum, u1_back, 'k-')
ax3.plot( itnum, u1_top, 'g-')
ax3.set_xlabel('Iteration', fontsize=20 )
ax3.set_ylabel('u (front: blue / back: black / top: green)', fontsize=16 )
ax3.set_title('Convergence of the streamfunction at three specific points', fontsize=20)
plt.savefig('figures/three_conv_w'+str1+'_it'+str2+'_R'+str3+'_Nx'+str4+'_Ny'+str5+'_.png', dpi=200)

#ax1.plot( t, y1, 'k-')
#ax1.plot( t_1, y0_1, 'r--')
#ax1.plot( t_1, y1_1, 'c--')
#ax1.plot( itnum, u1_back, 'k-')
#ax2.plot( y1_1, y0_1, 'r--')
#ax2.plot( y1_2, y0_2, 'b-')
