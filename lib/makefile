CXX = g++
CXXFLAGS = -O3 -c
LINK = g++
LINKFLAGS =

.PHONY : all

all : tensor.exe

tensor.exe : tensor.o
	$(LINK) $(LINKFLAGS) $< -o $@

tensor.o : tensor.cpp tensor.hh
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY : clean

clean : 
	rm -f *.o tensor.exe
