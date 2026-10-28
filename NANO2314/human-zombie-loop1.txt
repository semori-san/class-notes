# -*- coding: utf-8 -*-
"""

human & zombie populations

demonstrating linear algebra:
using a "for loop" to make a series of plots

"""

import matplotlib.pyplot as plt
import numpy as np

# These are the only parameters you need to modify
h_0 = 15  # initial human population
z_0 = 0   # initial zombie population
a_zh = 0.2 # coefficient: z gain from h
a_hz = 0.1 # coefficient: h gain from z
num_steps = 20  # number of time steps to simulate

a_hh = 1-a_zh  # Coefficient: h from h 
a_zz = 1-a_hz # Coefficient: z from z

pop_tot = h_0 + z_0 #total population, conserved

# This is how we input a matrix in Python
# A = np.array([[a11, a12],  <-- 1st row
#              [a21, a22]]) <-- 2nd row

# Build matrix that represents population evolution 
A = np.array([[a_hh, a_hz],
              [a_zh, a_zz]])

# Lists to store evolution over time
human_pop = [h_0]
zombie_pop = [z_0]

h_current = h_0
z_current = z_0

# Start a new plot
plt.figure()

# Plot at 0th hour
plt.quiver(0, 0, h_current, z_current, angles='xy', scale_units='xy', scale=1, 
           color='green')

# Set display ranges
plt.xlim(0, pop_tot)
plt.ylim(0, pop_tot)

# Set the plot's aspect ratio to be equal (square)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel("human population")
plt.ylabel("zombie population")
#plt.title(f"Hour 0: h = {h_current}, z = {z_current}")
plt.grid(True)
plt.savefig(f"HZvector-zh{a_zh}-hz{a_hz}-h0{h_0}-z0{z_0}-0hr.png")  # Saves as a PNG file
plt.show()

# Iterative simulation - repeat "num_steps" times
for n in range(num_steps):
    hour = n+1
    v = A @ np.array([h_current, z_current])  # Compute next state
    h_current, z_current = v  # Update for next iteration
    
    # Store results
    human_pop.append(h_current)
    zombie_pop.append(z_current)
    
    # Round numbers, to show limited number of digits
    h_disp = round(h_current, 2)
    z_disp = round(z_current, 2)
    
    # Start a new figure
    plt.figure()
    
    # Plot the vector at this iteration
    plt.quiver(0, 0, h_current, z_current, angles='xy', scale_units='xy', scale=1, 
               color='green')
    
    # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set display ranges
    plt.xlim(0, pop_tot)
    plt.ylim(0, pop_tot)
    
    plt.xlabel("human population")
    plt.ylabel("zombie population")
    plt.title(f"z from h: {a_zh}; h from z: {a_hz}; Hour {hour}")
    plt.grid(True)
    plt.savefig(f"HZvector-zh{a_zh}-hz{a_hz}-h0{h_0}-z0{z_0}-{hour}hr.png")  # Saves as a PNG file
    plt.show()