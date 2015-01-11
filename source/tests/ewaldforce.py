# Run this file to make a test of the Ewald method implemented in ewald.py.
# Note that this file has to be run in pure Python mode!

# Include the actual code directory in the searched paths
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))[:-1]))

# Check for pure Python mode
if any([filename.endswith('.so') for filename in os.listdir('.')]):
    raise Exception('You must run the ewaldforce test in pure Python mode!')

# General Python imports
from pylab import *
import subprocess
from time import sleep
# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
# Nbody imports
from _params_active import *
import ewald

x = boxsize*0.464
y = boxsize*0.213
z = boxsize*0.165
i = round(x*ewald_gridsize/boxsize *2)
j = round(y*ewald_gridsize/boxsize *2)
k = round(z*ewald_gridsize/boxsize *2)

def test():
    print('\nTests of the Ewald force in ewald.py. Parameters:')
    print('boxsize =', boxsize)
    print('ewald_gridsize =', ewald_gridsize)
    print('x, y, z =', x, y, z)
    print('i, j, k =', i, '(' + str(x*ewald_gridsize/boxsize *2) + ')',
                       j, '(' + str(y*ewald_gridsize/boxsize *2) + ')',
                       k, '(' + str(z*ewald_gridsize/boxsize *2) + ')')

    # Compile and run the gadget2 code written below
    filename = 'tests/ewaldforce_gadget.c'
    with open(filename, 'w') as gadget_file:
        # Print the code in the variable gadget_code to file filename
        print(gadget_code, end="", file=gadget_file)
    # Compile the code
    subprocess.Popen(['gcc', filename, '-lm', '-o', 'tests/ewaldforce_gadget'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
    sleep(1)
    # Run the code
    subprocess.Popen('tests/ewaldforce_gadget')
    sleep(1)
    # Cleanup
    subprocess.Popen(['rm', 'tests/ewaldforce_gadget.c'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
    subprocess.Popen(['rm', 'tests/ewaldforce_gadget'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
    # The ewald.summation function
    force = ewald.summation(x/boxsize, y/boxsize, z/boxsize)
    print('The ewald.summation function:                   ', force)
    print('-------------------------------------------------')
    # The ewald.summation function with manual scaling and removal of the direct force
    force = ewald.summation(x/boxsize, y/boxsize, z/boxsize)
    force /= boxsize**2
    r3 = (x**2 + y**2 + z**2 + softening2)**1.5
    force[0] += x/r3
    force[1] += y/r3
    force[2] += z/r3
    print('The ewald.summation function with manual scaling and removal of the direct force :', force)
    # The ewald.ewald function
    force = ewald.ewald(x, y, z)
    print('The ewald.ewald function:                                                         ', force)
    # The ewald.CIC function and manual scaling
    force = ewald.CIC(x/boxsize, y/boxsize, z/boxsize)
    force /= boxsize**2
    force[0] += x/r3
    force[1] += y/r3
    force[2] += z/r3
    print('The ewald.CIC function and manual scaling:                                        ', force)
    # Manual lookup in grid and scaling (no CIC)
    force = ewald.ewald(x, y, z)
    grid = load(ewald_file)
    print('Manual lookup in grid and scaling (no CIC):                                       ', grid[i, j, k, :] / boxsize**2 + array([x/r3, y/r3, z/r3]))

    # Periodicity tests with the ewald.CIC function (the highest level function, ewald.ewald,
    # cannot be used as it is not quite periodic due to removal of the force from the physical particle) 
    print('\nPeriodicity tests with ewald.CIC ...')
    force = ewald.CIC(x/boxsize, y/boxsize, z/boxsize)
    force /= boxsize**2
    force[0] += x/r3
    force[1] += y/r3
    force[2] += z/r3
    print('total_force(x, y, z):          ', force)
    force = ewald.CIC((x - boxsize)/boxsize, y/boxsize, z/boxsize)
    force /= boxsize**2
    force[0] += x/r3
    force[1] += y/r3
    force[2] += z/r3
    print('total_force(x - boxsize, y, z):', force)
    force = ewald.CIC((x + boxsize)/boxsize, y/boxsize, z/boxsize)
    force /= boxsize**2
    force[0] += x/r3
    force[1] += y/r3
    force[2] += z/r3
    print('total_force(x + boxsize, y, z):', force)

    # Plot of Ewald correction
    print('\nPlotting the Ewald correction field. Saving to ewaldforce_correction.pdf')
    meshrange = 0.5*boxsize*arange(ewald_gridsize)/ewald_gridsize
    Y, X = meshgrid(meshrange, meshrange)
    _, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, sharex='col', sharey='row')
    all_axes = [ax1, ax2, ax3, ax4]
    for Z, ax in zip([0, boxsize/8, boxsize/4, boxsize/2], all_axes):
        U = grid[:, :, int(2*Z/boxsize*(ewald_gridsize - 1)), 0]
        V = grid[:, :, int(2*Z/boxsize*(ewald_gridsize - 1)), 1]
        ax.quiver( X, Y, U, V, units='width')
        ax.set_title('z =' + str(Z/boxsize))
        axis([0, boxsize/2, 0, boxsize/2])
    xlabel('x')
    ylabel('y')
    savefig('tests/ewaldforce_correction.pdf')

    # Plot of total gravitational force
    print('\nPlotting the total force field. Saving to ewaldforce_total-force.pdf')
    for ii in range(ewald_gridsize):
        for jj in range(ewald_gridsize):
            for kk in range(ewald_gridsize):
                # Rescaling of coordinates. The factor 0.5 ensures
                # that only the first octant of the box is tabulated
                xx = 0.5*ii/ewald_gridsize
                yy = 0.5*jj/ewald_gridsize
                zz = 0.5*kk/ewald_gridsize
                r2 = xx**2 + yy**2 + zz**2
                if r2 == 0:
                    continue
                denum = (r2 + 0)**1.5
                grid[ii, jj, kk, 0] -= xx/denum
                grid[ii, jj, kk, 1] -= yy/denum
                grid[ii, jj, kk, 2] -= zz/denum
    _, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, sharex='col', sharey='row')
    all_axes = [ax1, ax2, ax3, ax4]
    for Z, ax in zip([0, boxsize/8, boxsize/4, boxsize/2], all_axes):
        U = grid[:, :, int(2*Z/boxsize*(ewald_gridsize - 1)), 0]
        V = grid[:, :, int(2*Z/boxsize*(ewald_gridsize - 1)), 1]
        ax.quiver( X, Y, U, V, units='width')
        ax.set_title('z =' + str(Z/boxsize))
        axis([0, boxsize/2, 0, boxsize/2])
    xlabel('x')
    ylabel('y')
    savefig('tests/ewaldforce_total-force.pdf')

    # Test in the Newtonian limit (particles are close --> very small Ewald correction)!
    # Create initial particles
    print('\nTesting the total force with N = 2 in the Newtonian limit')
    print('(interparticle distance << boxsize). Saving to ewaldforce_Newtonian-2-body.pdf.')
    start_vel = 3
    particles = {'pos': array([[boxsize*0.4, boxsize*0.4, boxsize*0.4],
                               [boxsize*0.4, boxsize*0.42, boxsize*0.4]]), 
                 'vel': array([[0, 0, -start_vel], [0, 0, start_vel]], dtype='float64') + array([[30, 20, 10], [30, 20, 10]])}
    # Direct summation + Ewald
    pos = particles['pos']
    vel = particles['vel']
    Y_path = []
    Z_path = []
    T = 5000
    for t in range(T):
        X = pos[1, 0] - pos[0, 0]
        Y = pos[1, 1] - pos[0, 1]
        Z = pos[1, 2] - pos[0, 2]
        Y_path.append(Y)
        Z_path.append(Z)
        # Using the high level ewald.ewald method
        force = ewald.ewald(X, Y, Z)
        # Add in the force from the actual particle
        r3 = (X**2 + Y**2 + Z**2 + softening2)**1.5
        force[0] -= X/r3
        force[1] -= Y/r3
        force[2] -= Z/r3
        # Update velocities
        dv = zeros((N, 3), dtype='float')
        dvx = dt*G_Newton*force[0]
        dvy = dt*G_Newton*force[1]
        dvz = dt*G_Newton*force[2]
        dv[0, 0] -= dvx
        dv[0, 1] -= dvy
        dv[0, 2] -= dvz
        dv[1, 0] += dvx
        dv[1, 1] += dvy
        dv[1, 2] += dvz
        for t2 in range(N):
            vel[t2, 0] += dv[t2, 0]
            vel[t2, 1] += dv[t2, 1]
            vel[t2, 2] += dv[t2, 2]
        # Update positions
        for t2 in range(N):
            pos[t2, 0] += vel[t2, 0]*dt
            pos[t2, 1] += vel[t2, 1]*dt
            pos[t2, 2] += vel[t2, 2]*dt
            # Stay inside the box
            pos[t2, 0] = pos[t2, 0] % boxsize
            pos[t2, 1] = pos[t2, 1] % boxsize
            pos[t2, 2] = pos[t2, 2] % boxsize
    # Plot the orbit viewed from particle 0.
    # Blue stars indicicate times at which the particles where "far away from each other" due to boundary effects.
    # Hopefully these do not behave differently than the other points!
    Y_path = array(Y_path)
    Z_path = array(Z_path)
    toroidal_effect = array(list(where(Y_path < -boxsize/2)[0]) + list(where(Y_path > boxsize/2)[0]) + list(where(Z_path < -boxsize/2)[0]) + list(where(Z_path > boxsize/2)[0]))
    Y_path[Y_path < -boxsize/2] += boxsize
    Y_path[Y_path > boxsize/2] -= boxsize
    Z_path[Z_path < -boxsize/2] += boxsize
    Z_path[Z_path > boxsize/2] -= boxsize
    figure()
    plot(Y_path, Z_path, 'r', linewidth=0.0001)
    plot(Y_path[toroidal_effect], Z_path[toroidal_effect], 'b*')
    savefig('tests/ewaldforce_Newtonian-2-body.pdf')

# The Ewald force code taken from Gadget2
gadget_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void ewald_force(double x[3], double force[3])
{
  double alpha, r2;
  double r, val, hdotx, dx[3];
  int i, h[3], n[3], h2;

  alpha = 2.0;

  for(i = 0; i < 3; i++)
    force[i] = 0;

  if(x[0] == 0.0 && x[1] == 0.0 && x[2] == 0.0)
    return;

  r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
"""
normal_gravity = r"""
  for(i = 0; i < 3; i++)
    force[i] += x[i] / (r2 * sqrt(r2));
"""

# Remove the normal gravity, leaving only the Ewald correction
#gadget_code += normal_gravity

gadget_code +=r"""
  for(n[0] = -4; n[0] <= 4; n[0]++)
    for(n[1] = -4; n[1] <= 4; n[1]++)
      for(n[2] = -4; n[2] <= 4; n[2]++)
	{
	  for(i = 0; i < 3; i++)
	    dx[i] = x[i] - n[i];

	  r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);

	  val = erfc(alpha * r) + 2 * alpha * r / sqrt(M_PI) * exp(-alpha * alpha * r * r);

	  for(i = 0; i < 3; i++)
	    force[i] -= dx[i] / (r * r * r) * val;
	}

  for(h[0] = -4; h[0] <= 4; h[0]++)
    for(h[1] = -4; h[1] <= 4; h[1]++)
      for(h[2] = -4; h[2] <= 4; h[2]++)
	{
	  hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
	  h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];

	  if(h2 > 0)
	    {
	      val = 2.0 / ((double) h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * sin(2 * M_PI * hdotx);

	      for(i = 0; i < 3; i++)
		force[i] -= h[i] * val;
	    }
	}
}

int main()
{

int iii;
int jjj;
int kkk;
double x[3];
double force[3];
force[0] = 0.0;
force[1] = 0.0;
force[2] = 0.0;
"""
gadget_code += 'x[0] = ' + str(x/boxsize) + ';\n'
gadget_code += 'x[1] = ' + str(y/boxsize) + ';\n'
gadget_code += 'x[2] = ' + str(z/boxsize) + ';\n'
gadget_code += r"""
ewald_force(x, force);


printf("Ewald from Gadget2 (no boxsize scaling!):        [ %g  %g  %g]\n", force[0], force[1], force[2]);
return 0;
}
"""

if __name__ == "__main__":
    test()

