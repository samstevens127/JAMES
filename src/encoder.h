#pragma once
#include "types.h"
#include <torch/torch.h>

struct GameState;
using EncodedState = torch::Tensor;

int encode_move(const GameState& state, nshogi::core::Move32 move);
EncodedState encode_state(const GameState &state);

// encoded both state and mirror for training
std::pair<EncodedState, EncodedState> encode_state_mirror(const GameState &state);
