# Optional Signal integration wiring.

# Signal integration is optional: only buildable when ../signal is present.
ifeq ($(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/Makefile),)
WITH_SIGNAL = 0
else
WITH_SIGNAL = 1
endif

SIGNAL_FLIGHT_INCLUDE_FLAGS := \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/server \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/shared \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/raticross \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/fastfilter \
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/connectome
SIGNAL_FLIGHT_SIM_SOURCES := \
	$(filter-out \
		$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/main.c \
		$(SIGNAL_FLIGHT_SIGNAL_DIR)/server/mongoose.c, \
		$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/*.c)) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/connectome/*.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/*.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/*.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/raticross/*.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/fastfilter/*.c)

SIGNAL_FLIGHT_TRACE_EXTRA :=
ifeq ($(SIGNAL_FLIGHT_COUNTERFACTUAL),1)
SIGNAL_FLIGHT_TRACE_EXTRA += --counterfactual --sample-stride $(SIGNAL_FLIGHT_SAMPLE_STRIDE) --horizon-ticks $(SIGNAL_FLIGHT_HORIZON_TICKS) --explore-random $(SIGNAL_FLIGHT_EXPLORE_RANDOM)
endif
ifeq ($(SIGNAL_FLIGHT_PAIN_PROBE),1)
SIGNAL_FLIGHT_TRACE_EXTRA += --pain-probe
endif
ifeq ($(SIGNAL_FLIGHT_WALL_PROBE),1)
SIGNAL_FLIGHT_TRACE_EXTRA += --wall-probe
endif
