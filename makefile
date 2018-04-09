SUBDIRS := test
CXX := g++
CXXFLAGS := -O3 -I./ -I./include -Wall -Wextra -fmax-errors=5 -std=c++11 -MMD
LINK := g++
LINKFLAGS :=

TEMPORARY_PATTERNS := *.o *~ *.d
TEMPORARIES := $(foreach DIR,$(SUBDIRS),$(addprefix $(DIR)/,$(TEMPORARY_PATTERNS)))

SRC := $(wildcard $(SUBDIRS)/*.cc)
OBJ := $(SRC:.cc=.o)

.PHONY : all

all : run_tests

run_tests : $(OBJ)
	$(LINK) $(LINKFLAGS) $^ -o $@

-include $(OBJ:.o=.d)

.PHONY : clean

clean :
	rm -f $(TEMPORARIES) run_tests
