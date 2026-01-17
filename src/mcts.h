#pragma once
#include "types.h"
#include "nn.h"



class MCTS {
        public:
                struct SearchResult {
                        nshogi::core::Move32 best_move;
                        std::vector<std::pair<int, uint32_t>> visit_counts; // <move_index, visits>
                        float root_value;
                };
                MCTS(NodePool& pool);
                
                MCTS::SearchResult search(const GameState &root, std::shared_ptr<NeuralNetwork> nn, const size_t iterations);
                void start_new_game();
                void update_root(const GameState& state, nshogi::core::Move32 move);


        private:
                void perform_iteration(MCTSNode* root_node, GameState state, std::shared_ptr<NeuralNetwork> nn);
                NodePool& pool;
                MCTSNode* root_node = nullptr;
};
