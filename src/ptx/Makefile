PTXXX=ptxas
FBXX=fatbin
CXX=gcc

PTXFLAGS=-arch=sm_70

preprocess: mmult.ptx
	$(CXX) -E - < $< > mmult_proc.ptx

ptx: preprocess
	$(PTXXX) -arch=sm_70 -c -o mmult.gpu.o mmult_proc.ptx
