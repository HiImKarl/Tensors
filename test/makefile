CXX = g++
CXXFLAGS = -O3
LINK = g++
LINKFLAGS =

.PHONY : all

all : test

test : test.o
	$(LINK) $(LINKFLAGS) $^ -o $@

test.o : test.cpp tensor.hh
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY : clean

clean : 
	rm -f *.o test
