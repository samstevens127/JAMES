#pragma once
#include "types.h"
#include "nn.h"



template<bool training>
class MCTS {
        public:
                struct SearchResult {
                        nshogi::core::Move32 best_move;
                        std::vector<std::pair<int, uint32_t>> visit_counts; // <move_index, visits>
                        float root_value;
                };
                MCTS(NodePool<training>& pool, const int n_threads = 1);
                
                MCTS::SearchResult search(const GameState &root, std::shared_ptr<NeuralNetwork> nn, const size_t iterations);
                void start_new_game();
                void update_root(const GameState& state, nshogi::core::Move32 move);


        private:
                bool expand(MCTSNode<training>* node, GameState &state, std::shared_ptr<NeuralNetwork> nn, float &value);
                bool perform_iteration(MCTSNode<training>* root_node, GameState state, std::shared_ptr<NeuralNetwork> nn);

                std::mt19937 gen{std::random_device{}()};


                NodePool<training>& pool;
                int num_threads = 1;
                MCTSNode<training>* root_node = nullptr;

                uint8_t search_depth = 0;
};
