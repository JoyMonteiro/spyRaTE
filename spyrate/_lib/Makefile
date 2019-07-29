TARGET_LIB=librte.a
RRTMGP_DIR=rte-rrtmgp

FC=gfortran
FCFLAGS="-ffree-line-length-none -m64 -std=f2003 -march=native -DUSE_CBOOL"

all: $(TARGET_LIB)

$(TARGET_LIB): $(RRTMGP_DIR)
	cd $(RRTMGP_DIR)/build; make; cp librte.a ../../;

$(RRTMGP_DIR):
	git clone https://github.com/RobertPincus/rte-rrtmgp.git
