PTXXX=ptxas
FBXX=fatbinary
CXX=gcc
NVXX=nvcc

ARCH=sm_70
PTXFLAGS=-arch=$(ARCH)
FBXXFLAGS=-64

all: fatbin
	$(NVXX) -arch=$(ARCH) -Xptxas="-v" -dlink -o mmult.o mmult.fatbin mmult.cu

preprocess: mmult.ptx
	$(CXX) -E - < $< > mmult_proc.ptx

ptx: preprocess
	$(PTXXX) $(PTXFLAGS) -c -o mmult.sm_70.cubin mmult_proc.ptx

fatbin: ptx
	$(FBXX) -create mmult.fatbin "--image=profile=sm_70,file=mmult.sm_70.cubin" $(FBXXFLAGS)

clean:
	rm mmult.o mmult.fatbin mmult_proc.ptx mmult.sm_70.cubin
