# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall -lm -funroll-loops -Wno-unused-result
PAR_FLAG = -fopenmp -fcilkplus -pthread -DCILK

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = pr #classification
SUITE = $(KERNELS) converter

.PHONY: all
all: $(SUITE)

% : src/%.cc src/*.h
	$(CXX) $(CXX_FLAGS) $< -o $@

# # Testing
# include test/test.mk

# # Benchmark Automation
# include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*