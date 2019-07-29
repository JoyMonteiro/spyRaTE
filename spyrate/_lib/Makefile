TARGET_LIB=librte.a
RRTMGP_DIR=rte-rrtmgp

all: $(TARGET_LIB)

$(TARGET_LIB): $(RRTMGP_DIR)
	cd $(RRTMGP_DIR)/build; make; cp librte.a ../../;

$(RRTMGP_DIR):
	git clone https://github.com/RobertPincus/rte-rrtmgp.git
