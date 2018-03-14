CXX = g++ -std=c++14
CXXFLAGS = -O3 -I../ -Wall -Wextra -fmax-errors=3
LINK = g++
LINKFLAGS =

.PHONY : all

all : test

test : test.o
	$(LINK) $(LINKFLAGS) $^ -o $@

test.o : test.cc ../tensor.hh
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY : clean

clean : 
	rm -f *.o test *~
