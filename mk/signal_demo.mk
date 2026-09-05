# Native orchestration for the pinned connected demonstration.
SIGNAL_DEMO := $(BUILD_DIR)/signal_demo
SIGNAL_DEMO_TEST := $(BUILD_DIR)/test_signal_demo
SIGNAL_DEMO_SANITIZE_TEST := $(BUILD_DIR)/test_signal_demo_sanitize
SIGNAL_DEMO_SOURCES := examples/signal_demo.c examples/signal_demo_runtime.inc
SIGNAL_DEMO_FLAGS := -DSIGNAL_DEMO_ROOT='"$(CURDIR)"' -fno-fast-math -ffp-contract=off

.PHONY: signal-demo-contract-check signal-demo-contract-sanitize
signal-demo-contract-check: $(SIGNAL_DEMO_TEST)
	./$(SIGNAL_DEMO_TEST)

check: signal-demo-contract-check
sanitize: signal-demo-contract-sanitize

signal-demo-contract-sanitize: $(SIGNAL_DEMO_SANITIZE_TEST)
	./$(SIGNAL_DEMO_SANITIZE_TEST)

$(SIGNAL_DEMO): $(SIGNAL_DEMO_SOURCES) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SIGNAL_DEMO_FLAGS) examples/signal_demo.c $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_DEMO_TEST): tests/test_signal_demo.c $(SIGNAL_DEMO_SOURCES) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SIGNAL_DEMO_FLAGS) -Wno-unused-function tests/test_signal_demo.c $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) $(LDFLAGS) $(LDLIBS) -o $@

$(SIGNAL_DEMO_SANITIZE_TEST): tests/test_signal_demo.c $(SIGNAL_DEMO_SOURCES) $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SIGNAL_DEMO_FLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wno-unused-function tests/test_signal_demo.c $(NN_OBJ) $(NN_BLAS_OBJ) $(NN_JSON_OBJ) $(NN_TEXT_OBJ) $(LDFLAGS) $(LDLIBS) -o $@
