# Build with:  make            (portfolio + convergence)
#              make test       (unit tests)
#              make data N=100000   (generate bonds.csv)
#              make clean
#
# macOS needs libomp:  brew install libomp
# Linux uses gcc with -fopenmp.

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  CC = clang
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS = -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS = -L$(OMP_PREFIX)/lib -lomp -Wl,-rpath,$(OMP_PREFIX)/lib
else
  CC = gcc
  OMP_CFLAGS = -fopenmp
  OMP_LDFLAGS =
endif

# -DUNIT_TEST removes the interactive main() inside 1.ytm.c so it can be linked.
CFLAGS = -O2 -DUNIT_TEST
N ?= 100000

.PHONY: all test data clean

all: portfolio convergence

portfolio: 3.portfolio.c 1.ytm.c 2.duration_convexity.c 1.ytm.h 2.duration_convexity.h
	$(CC) $(CFLAGS) $(OMP_CFLAGS) 3.portfolio.c 1.ytm.c 2.duration_convexity.c -o $@ $(OMP_LDFLAGS) -lm

convergence: 5.convergence.c 1.ytm.c 1.ytm.h
	$(CC) $(CFLAGS) 5.convergence.c 1.ytm.c -o $@ -lm

test_project: test_project.c 1.ytm.c 2.duration_convexity.c 1.ytm.h 2.duration_convexity.h
	$(CC) $(CFLAGS) test_project.c 1.ytm.c 2.duration_convexity.c -o $@ -lm

test: test_project
	./test_project

data:
	python3 3.gen_portfolio.py $(N)

clean:
	rm -f portfolio convergence test_project results.csv