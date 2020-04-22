PTXXX=ptxas
FBXX=fatbinary
CXX=gcc

PTXFLAGS=-arch=sm_70
FBXXFLAGS=-64
preprocess: mmult.ptx
	$(CXX) -E - < $< > mmult_proc.ptx

ptx: preprocess
	$(PTXXX) $(PTXFLAGS) -c -o mmult.sm_70.cubin mmult_proc.ptx

fatbin: ptx
	$(FBXX) -create mmult.fatbin "--image=profile=sm_70,file=mmult.sm_70.cubin" $(FBXXFLAGS)
