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
	-I$(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/raticross
SIGNAL_FLIGHT_SIM_SOURCES := \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/game_sim.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/state_digest.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/actor_principal.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/public_actor_resolver.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/contract_ownership.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/ownership_quarantine.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/player_identity_registry.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_legacy_classify.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_legacy_inventory.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_craft_provenance.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_smelt_provenance.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/ship_birth_reservation.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_ai.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/signal_brain.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/signal_intelligence.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/signal_contract_brain.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/signal_npc_worker_brain.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/gossip.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_autopilot.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_flight.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_nav.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_save.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_catalog.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/station_authority.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/chain_log.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/chain_log_verify.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_receipt_issue.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/cargo_receipt_trust.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/handoff_flow.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_asteroid.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_physics.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_production.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_construction.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_mining.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_anchor.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/server/sim_ship.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/actor_principal.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/public_actor_id.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/cell_geometry.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/cell_stress.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/cargo_package.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/commodity.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/manifest.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/ship.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/economy.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/asteroid.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/rng.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/belt.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/module_schema.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/station_util.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/cargo_receipt.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/pubkey_proof.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/signal_memzero.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/rx_envelope.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/handoff_ticket.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/tractor.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/laser.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/trade_paging.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/holographic_nn.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/holographic_nn_backend.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/holographic_nn_confidence.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/signal_field.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/npc_radio.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/raticross/raticross.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/tweetnacl.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/randombytes.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/vendor/tweetnacl/signal_crypto_tweetnacl.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/fixpoint.c) \
	$(wildcard $(SIGNAL_FLIGHT_SIGNAL_DIR)/shared/settlement_engine.c) \

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
