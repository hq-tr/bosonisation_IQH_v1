import numpy as np
import FQH_states as FQH
from bosonisation import bosonisation_mapping, b2f
from two_qh_state import two_particle_state_boson, two_qh_IQH_state
from plot_line import lineplot_disk_density
import matplotlib.pyplot as plt

Ne = int(input("Input N_e: "))
No = Ne+2

try:
	with open("best_gauge") as f:
		best_gauge = f.read().split()
except FileNotFoundError:
	print("Best gauge not found. Initialize the routine with default gauge.")
	print("Alternatively, you may terminate the program and run get_2qh_state_2.py to get the best gauge.")
	cntyn = str(input("Continue (y/n)? "))
	if cntyn in ["n", "N"]:
		exit()

mapping = bosonisation_mapping(Ne, best_gauge).get()

# Get a state to bosonise:
fname = str(input("Bosonic state file name: "))
NS_state = FQH.fqh_state_boson(fname)
if NS_state.format == "decimal":
	NS_state.format_convert()

# Convert the bosonic state to fermion basis (IQH) using the given gauge
fermion_state = b2f(NS_state,mapping)

# Convert basis format to decimal
fermion_state.format_convert()

# Save to file
fermion_state.printwf(f"{fname}_boson")