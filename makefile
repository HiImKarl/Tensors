SUBDIRS := $(wildcard */.)

all:
	cp lib/tensor.hh test/tensor.hh
	$(MAKE) -C $@
clean:
	$(MAKE) -C $(SUBDIRS) clean

.PHONY: all clean

