CXX = g++
CXXFLAGS = -O3 -I../ -Wall -Wextra
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
	rm -f *.o test
