
# Include the code directory in the searched paths
import sys, os
concept_dir = os.path.realpath(__file__)
this_dir = os.path.dirname(concept_dir)
while True:
    if concept_dir == '/':
        raise Exception('Cannot find the .paths file!')
    if '.paths' in os.listdir(os.path.dirname(concept_dir)):
        break
    concept_dir = os.path.dirname(concept_dir)
sys.path.append(concept_dir)

from commons import *
from IO import load_standard, Gadget_snapshot
import graphics
from graphics import animate

import struct

def plot(filename):
    # Detect whether the snapshot is of the standard type or of the Gadget 2 type
    input_type = 'standard'
    with open(filename, 'rb') as f:
        try:
            f.seek(4)
            if struct.unpack('4s', f.read(struct.calcsize('4s')))[0] == b'HEAD':
                input_type = 'GADGET 2'
        except:
            pass
    # Load a, boxsize and the particles
    if input_type == 'GADGET 2':
        snapshot = Gadget_snapshot()
        snapshot.load(filename, write_msg=False)
        a = snapshot.header['Time']
        boxsize = snapshot.header['BoxSize']*units.kpc/snapshot.header['HubbleParam']
        particles = snapshot.particles
    elif input_type == 'standard':
        with h5py.File(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
            a = hdf5_file.attrs['a']
            boxsize = hdf5_file.attrs['boxsize']
        particles = load_standard(filename, write_msg=False)
    # Plot
    graphics.boxsize = boxsize
    animate(particles, 0, a, a, filename=(os.path.basename(filename) + '.' + image_format))

if output_type == 'file':
    # Plot the single snapshot file
    filename = IC_file
    plot(filename)
elif output_type == 'dir':
    # Plot every snapshot file in the directory
    dirname = IC_file
    files = os.listdir(dirname)
    for filename in files:
        try:
            plot(dirname + '/' + filename)
        except:
            pass
