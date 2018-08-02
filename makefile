TEST_DIR := test
BENCHMARK_DIR := benchmark

CXX := g++
CXXFLAGS := -O3 -I./include -I./external -Wall -Wextra -fmax-errors=5 -std=c++11 -MMD -g 
LINK := g++
LINKFLAGS := -g

TEMPORARY_PATTERNS := *.o *~ *.d
TEMPORARIES := $(foreach DIR, $(TEST_DIR) $(BENCHMARK_DIR),$(addprefix $(DIR)/,$(TEMPORARY_PATTERNS)))

TEST_SRC := $(wildcard $(TEST_DIR)/*test.cc) 
TEST_OBJ := $(TEST_SRC:.cc=.o)

BENCHMARK_SRC := $(wildcard $(BENCHMARK_DIR)/*benchmark.cc) 
BENCHMARK_OBJ := $(BENCHMARK_SRC:.cc=.o)

.PHONY : valgrind
valgrind : CXXFLAGS += -D_TEST=1
valgrind : run_test
	valgrind --leak-check=full --show-leak-kinds=all ./run_test
	rm run_test

.PHONY : test 
test : CXXFLAGS += -D_TEST=1
test : run_test
	./run_test
	rm run_test

run_test : ${TEST_OBJ}
	$(LINK) $^ -o $@ $(LINKFLAGS) 

-include $(TEST_OBJ:.o=.d)

.PHONY : benchmark
benchmark : LINKFLAGS += -lbenchmark -lpthread
benchmark : run_benchmark
	./run_benchmark --benchmark_out=benchmark.log --benchmark_out_format=json
	rm run_benchmark

run_benchmark : $(BENCHMARK_OBJ)
	$(LINK) $^ -o $@ $(LINKFLAGS)

-include $(BENCHMARK_OBJ.o=.d)

.PHONY : documentation
documentation :
	doxygen Doxyfile

.PHONY : clean

clean :
	rm -f $(TEMPORARIES)
