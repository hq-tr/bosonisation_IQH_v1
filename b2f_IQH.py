import misc
import task1
import angular_momentum as am 
import FQH_states as FQH 
from scipy.linalg import eigh, inv
import numpy as np
from subprocess import call
from itertools import product
from multiprocessing import Pool

def gauge_list(dim): # just doing the retarded way now because im tired
	if dim == 1:
		return ["0", "1"]
	elif dim == 2:
		return ["00","01","10","11"]
	elif dim == 3:
		return ["000","001","010","011","100","101","110","111"]
	else:
		return ["".join(x) for x in product("01", repeat=dim)]

def gauge_pick(option):
	dim = len(option)
	return np.array([np.ones(dim)-2*(x=="1") for x in option])

def get_index(a,b,tol=1e-10):
	index = [i for i in range(len(a)) if abs(a[i]-b)<tol]
	if len(index)==1:
		return index[0]
	else:
		return index

def get_note(g):
	if "1" in g:
		note = g+"_"
	else:
		note = ""
	return note

Ne = int(input("Input N_e: "))
No = Ne+2

print(f"Finding bosonization mapping for two-quasihole IQH state, {Ne} electrons")

def get_states(Lz):
	print(Lz)

	call(["mkdir",f"Lz_{Lz}"])
	# Fermionic states
	b = task1.findBasis_brute(Ne,No,True,-Lz)
	dim = len(b)

	with open(f"Lz_{Lz}/basis_fermion", "w+") as f:
		f.write("\n".join([misc.index_to_binary(vec, No, get_string=True) for vec in b]))

	LL = task1.LplusLminus_2(b,Ne,No)
	E, V = eigh(LL.toarray())
	V = V.T 

	#print("V =")
	#print(V)

	# Bosonic states
	b2 = task1.findBasis_brute(2,Ne+1,True, Lz, bosonic=True)

	with open(f"Lz_{Lz}/basis_boson", "w+") as f:
		f.write("\n".join([misc.index_to_binary_boson(vec, Ne+1, inc_space=True) for vec in b2]))

	LL2 = am.LminusLplus_boson(b2, Ne+1)
	E2, V2 = eigh(LL2.toarray())
	V2 = V2.T
	#for i in range(len(b)):
	#	print(E2[i], end = "\t")
	#	print(V2[i])

	index = [get_index(E2, En) for En in E]
	#print(index)

	W = np.array([V2[i] for i in index])
	#W[1] = -W[1]
	#print("W = ")
	#print(W)

	#print("***")
	if dim > 1:
		for g in gauge_list(dim):
			# Express bosonic state as a linear coefficient of fermionic states
			#print(g)
			#print(gauge_pick(g)*V)
			C = np.dot(inv(W), gauge_pick(g)*V)
			for i in range(dim):
				with open(f"Lz_{Lz}/fermionic_coef_{get_note(g)}{i}", "w+") as f:
					f.write("\n".join(list(map(str, C[i]))))
	else:
		with open(f"Lz_{Lz}/fermionic_coef_0", "w+") as f:
			f.write("1.0")
	return

with Pool(Ne+1) as p:
	p.map(get_states, range(Ne+1))

