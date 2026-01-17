#pragma once
#include "types.h"

int encode_move(const GameState& state, nshogi::core::Move32 move);
EncodedState encode_state(const GameState &state);
