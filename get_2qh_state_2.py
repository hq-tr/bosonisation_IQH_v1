import misc
import numpy as np
from FQH_states import *
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool
import time

Ne = int(input("Input N_e: "))
No = Ne+2 

theta_list = np.linspace(0,np.pi,200)
z0_list    = np.linspace(0,np.sqrt(2*No),200)
state = two_qh_state(Ne,No)

sign_list  = [1,1,1,1,1,1]

def gauge_list_get(dim): # just doing the retarded way now because im tired
	if dim == 1:
		return ["0"]
	elif dim == 2:
		return ["00","01","10","11"]
	elif dim == 3:
		return ["000","001","010","011","100","101","110","111"]
	else:
		return ["".join(x) for x in product("01", repeat=dim)]

def glist(x): 
	return product([gauge_list_get(5)[x[0]]],[gauge_list_get(4)[x[1]]],gauge_list_get(4),gauge_list_get(3),gauge_list_get(3), gauge_list_get(2),gauge_list_get(2),gauge_list_get(1),gauge_list_get(1))

#glist = [["000","00","00","0","0"]]

def scale_factor(m):
	return ((m+1)/(m+2))**(m/2)

#with open("braiding_phase") as f:
#	phase = list(map(float, f.readlines()))

def get_electronic_state(theta, gauge_list, phi=0, fname = None, Lz = None, disk=True, correction_term=0):
	z_true = theta-correction_term
	if phi==0:
		if disk:
			state.z_0 = z_true
		else:
			state.z_0 = np.tan(theta/2)
	else:
		if disk:
			state.z_0 = z_true
		else:
			state.z_0 - np.tan(theta/2)*np.exp(1j*phi)
	coef_boson = state.get_bosonic_coef()
	wavefunction = fqh_state()
	#gauge_list = glist[gauge_index]
	#print(f"Gauge choice: {gauge_list}")
	if Lz == None:
		normalize=True
		for i in range(len(coef_boson)):
			Lz = Ne-i 
			#print(Lz, end = "\t")
			with open(f"Lz_{Lz}/basis_fermion") as f:
				b = f.readlines()

			dim = len(b)
			#print(dim, end = "\t")
			#with open(f"root_list_{Lz}_(1,3)/bosons/basis") as f:
			#	print(f.readline())

			if "1" in gauge_list[Lz]:
				note = gauge_list[Lz]+"_"
			else:
				note = ""

			with open(f"Lz_{Lz}/fermionic_coef_{note}0") as f:
				coef_list = list(map(float, f.readlines()))
			#print(coef_list)
			

			for j in range(dim):
				bas = b[j].replace("\n","")[::-1]
				get_wf = fqh_state(([bas], [1.0]))
				if i == 1:
					wavefunction += (-1)*coef_boson[i]*coef_list[j]*get_wf
				else:
					wavefunction += coef_boson[i]*coef_list[j]*get_wf
	else:
		normalize=False
		with open(f"Lz_{Lz}/basis_fermion") as f:
			b = f.readlines()
		dim = len(b)
		if "1" in gauge_list:
			note = f"{gauge_list}_"
		else:
			note = ""

		with open(f"Lz_{Lz}/fermionic_coef_{note}0") as f:
			coef_list = list(map(float, f.readlines()))
		for j in range(dim):
				bas = b[j].replace("\n","")[::-1]
				get_wf = fqh_state(([bas], [1.0]))
				if Lz == Ne-1:
					wavefunction += (-1)*coef_boson[Ne-Lz]*coef_list[j]*get_wf
				else:
					wavefunction += coef_boson[Ne-Lz]*coef_list[j]*get_wf
	if normalize:
		if disk:
			wavefunction.normalize()
			wavefunction.printwf("aa")
		else:
			wavefunction.normalize()
			#print(wavefunction.dim())
	if fname == None:
		return wavefunction 
	else:
		wavefunction.printwf(fname)
		return

def get_IQH_state(theta, phi = 0, fname=None, disk=True):
	z_true = theta/2
	if phi==0:
		if disk:
			state.z_0 = z_true
		else:
			state.z_0 = np.tan(theta/2)
	else:
		if disk:
			state.z_0 = z_true*np.exp(1j*phi)
		else:
			state.z_0 - np.tan(theta/2)*np.exp(1j*phi)
	coef_fermion = state.get_fermionic_polynomial_coef()
	Laughlin_wf  = fqh_state()
		#print(coef_fermion)
	for i in range(len(coef_fermion)):
		Lz = Ne-i
		with open(f"Lz_{Lz}/basis_fermion") as f:
			a = f.readlines()
		bas = a[-1].replace("\n","")[::-1]
		#print(bas)
		get_wf = fqh_state(([bas], [1.0]))
		Laughlin_wf += coef_fermion[i]*get_wf
	
	Laughlin_wf.printwf("test_state")
	if disk:
		Laughlin_wf.disk_normalize()
		Laughlin_wf.printwf("test_state_normalized")
	else:
		Laughlin_wf.sphere_normalize()

	if fname == None:
		return Laughlin_wf
	else:
		Laughlin_wf.printwf(fname)
		return

def get_dim(Lz):
	return (Ne-Lz)//2 + 1

def get_overlap(gauge_list, custom_theta_list = [],fname=None):
	ov = []
	if len(custom_theta_list) == 0:
		z0list = z0_list
	else:
		z0list = np.tan(custom_theta_list/2)
	for i in range(200):
		z = z0list[i]
		#r_phase = np.sqrt(2*phase[i]+(z**2))
		wf = get_electronic_state(z, gauge_list)
		Laughlin_wf = get_IQH_state(z)
		ov.append(wf.overlap(Laughlin_wf)**2)
	if fname == None: 
		fname = "overlap_data_all/overlap_"+"_".join(gauge_list)
	if len(custom_theta_list) == 0:
		with open(fname,"w+") as f:
			f.write("\n".join(list(map(str, ov))))
	else:
		with open("overlap_check", "w+") as f:
			f.write("\n".join(list(map(str, ov))))
	return

def get_overlap_one(gauge):
	theta = np.sqrt(No)
	#print(gauge,end="\r")
	a, b = get_electronic_state(theta, gauge)
	return a.overlap(b)**2

def get_best_gauge(glist_it):
	return max(glist_it, key=get_overlap_one)

if __name__ == "__main__":
	st = time.time()
	theta = np.sqrt(No)
	best_gauge = []
	IQH_state = get_IQH_state(theta)
	for Lz in range(Ne+1):
		def get_overlap_lz(gauge):
			all_gauge = ["0"*get_dim(i) if i!=Lz else gauge for i in range(Ne+1)]
			#print(all_gauge)
			wf = get_electronic_state(theta, gauge, Lz=Lz)
			return wf.overlap(IQH_state)
		dim = get_dim(Lz)
		print(Lz, end="\t")
		print(dim)
		best_gauge_lz = max(gauge_list_get(dim), key=get_overlap_lz)
		best_gauge.append(best_gauge_lz)
	print(time.time()-st)
	print(best_gauge)

	with open("best_gauge","w+") as f:
		f.write(" ".join(best_gauge))

	#get_overlap(best_gauge, fname=f"best_overlap_{Ne}e.dat")