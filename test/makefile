CXX = g++
CXXFLAGS = -O3 -I../ -Wall -Wextra -fmax-errors=3 -std=c++14 -MMD
LINK = g++
LINKFLAGS =

SRC := $(wildcard *.cc)
OBJ := $(SRC:.cc=.o)

.PHONY : all

all : test

test : $(OBJ)
	$(LINK) $(LINKFLAGS) $^ -o $@

-include $(OBJ:.o=.d)

.PHONY : clean

clean :
	rm -f *.o test *~
