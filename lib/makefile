CXX = g++
CXXFLAGS = -O3
LINK = g++
LINKFLAGS =

.PHONY : all

all : tensor

tensor : tensor.o
	$(LINK) $(LINKFLAGS) $^ -o $@

tensor.o : tensor.cpp tensor.hh
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY : clean

clean : 
	rm -f *.o tensor
