SUBDIRS := test

all:
	cp lib/tensor.hh test/tensor.hh
	$(MAKE) -C $(SUBDIRS)

clean:
	$(MAKE) -C $(SUBDIRS) clean

.PHONY: all clean

