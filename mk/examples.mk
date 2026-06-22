.PHONY: run benchmark
.PHONY: bitwise-recursive tile-recursive tile-feature-entropy life-curriculum layer-discovery

run: $(XOR_EXAMPLE)
	./$(XOR_EXAMPLE)

bitwise-recursive: $(BITWISE_RECURSIVE)
	./$(BITWISE_RECURSIVE)

tile-recursive: $(TILE_RECURSIVE)
	./$(TILE_RECURSIVE)

tile-feature-entropy: $(TILE_FEATURE_ENTROPY)
	./$(TILE_FEATURE_ENTROPY)

life-curriculum: $(LIFE_CURRICULUM)
	./$(LIFE_CURRICULUM)

layer-discovery: $(LAYER_DISCOVERY)
	./$(LAYER_DISCOVERY)

benchmark: $(BENCHMARK)
	./$(BENCHMARK)
