#pragma once
#include "types.h"

struct GameState;
using EncodedState = std::array<float, 48 * 9 * 9>;

int encode_move(const GameState& state, nshogi::core::Move32 move);
EncodedState encode_state(const GameState &state);
