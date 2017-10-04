SUBDIRS := test

all:
	$(MAKE) -C $(SUBDIRS)

clean:
	$(MAKE) -C $(SUBDIRS) clean

.PHONY: all clean

