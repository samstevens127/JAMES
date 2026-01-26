#include "mcts.h"
#include "encoder.h"
#include "types.h"
#include <limits>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

template<bool training>
MCTS<training>::MCTS(NodePool<training>& p, int n_threads) : pool(p), num_threads(n_threads)
{
        start_new_game();
}

template<bool training>
void MCTS<training>::start_new_game() 
{
        pool.reset();
        root_node = pool.allocate_single();
        root_node->reset();
}

template<bool training>
void MCTS<training>::update_root(const GameState& state, nshogi::core::Move32 move) {
        MCTSNode<training>* next_root = nullptr;
        for (auto& child : root_node->children()) {
                if (child.move == move) {
                        next_root = &child;
                        break;
                }
        }
        
        if (next_root) {
                root_node = next_root;
                root_node->parent_ptr = nullptr; 
        } else {
                start_new_game();
        }
}

// Helper for atomic float addition
static void add_to_atomic_float(std::atomic<float>& atomic_float, float delta) 
{
        float expected = atomic_float.load(std::memory_order_relaxed);
        float desired;
        do {
                desired = expected + delta;
        } while (!atomic_float.compare_exchange_weak(expected, desired, std::memory_order_relaxed));
}

// Static helper: picks the best child of the current node
template<bool training>
static MCTSNode<training>* select_child_puct(MCTSNode<training>* node) 
{
        float max_puct = -std::numeric_limits<float>::infinity();
        MCTSNode<training>* best_child = nullptr;
        uint32_t parent_visits = 0;

        if constexpr (!training)
                parent_visits = node->visits.load(std::memory_order_relaxed);
        else 
                parent_visits = node->visits;
        
        for (auto& child : node->children()) {
                float curr_puct = child.puct(parent_visits);
                
                if (curr_puct > max_puct) {
                        best_child = &child;
                        max_puct = curr_puct;
                }
        }
        return best_child;
}

/* @brief helper function to expand and evaluate node
 * 
 */
template<bool training>
bool MCTS<training>::expand(MCTSNode<training>* node, GameState &state, std::shared_ptr<NeuralNetwork> nn, float &value)
{
        if constexpr (!training) {
                bool expected = false;
                if (!node->is_expanding.compare_exchange_strong(expected, true)) {
                        try{
                                auto future = nn->evaluate_async(state);
                                auto result = future.get(); 
                                node->expand_with_policy(state, result.policy, pool);
                                value = result.value;
                                node->expanded.store(true, std::memory_order_release);
                                return true;
                        } catch(...) {
                                node->is_expanding.store(false, std::memory_order_release);
                                node->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                                return true;
                        }
                        
                } else {
                        node->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                        MCTSNode<training>* curr = node->parent_ptr;
                        while (curr) {
                                curr->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                                curr = node->parent_ptr;
                        }
                        return false;
                }
                return false;
        } else {
                auto future = nn->evaluate_async(state);
                auto result = future.get(); 
                node->expand_with_policy(state, result.policy, pool);
                value = result.value;
                node->expanded = true;
        }
        return false;
}

template<bool training>
bool MCTS<training>::perform_iteration(MCTSNode<training>* root_node, GameState state, std::shared_ptr<NeuralNetwork> nn) 
{
        MCTSNode<training>* node = root_node;
        
        // select 
        if constexpr (!training) {
                while (node->expanded.load(std::memory_order_acquire) &&
                                !(node->children().size()) == 0) {
                        node->virtual_loss.fetch_add(1, std::memory_order_relaxed); 
                        node = select_child_puct<training>(node);
                        state.do_move(node->move);
                }
                node->virtual_loss.fetch_add(1, std::memory_order_relaxed);
        } else {
                while (node->expanded &&
                                !(node->children().size()) == 0) {
                        node = select_child_puct<training>(node);
                        state.do_move(node->move);
                }
        }
        
        
        float value = 0.0f;
        bool is_terminal = state.is_terminal();
        bool exit_code = false;
        
        if (is_terminal) {
                value = state.result();
        } else {
                if constexpr (!training) {
                        // evaluate
                        if (!node->expanded.load(std::memory_order_acquire))
                                exit_code = expand(node, state, nn, value); 
                
                } else {
                        if (!node->expanded)
                                exit_code = expand(node, state, nn, value); // evaluation handled here
                }
        }
                // backprop & Remove Virtual Loss
        if constexpr(!training){
                while (node != nullptr) {
                        node->virtual_loss.fetch_sub(1, std::memory_order_relaxed); 
                        
                        if (!is_terminal && !node->expanded) {
                                node = node->parent_ptr;
                                continue;
                        }
                        
                        node->visits.fetch_add(1, std::memory_order_relaxed);
                        add_to_atomic_float(node->value_sum, value);
                        value = -value;
                        node = node->parent_ptr;
                }
        } else {
                while (node != nullptr) {
                        node->visits++;
                        node->value_sum += value;
                        value = -value;
                        node = node->parent_ptr;

                }
        }
        return exit_code;
}
/* @brief performs search from root
 *
 * @params
 * GameState& root
 * shared_ptr<NeuralNetwork> nn
 * size_t iterations
 */


template<bool training>
typename MCTS<training>::SearchResult MCTS<training>::search(const GameState &root_state,
                                std::shared_ptr<NeuralNetwork> nn,
                                size_t iterations) 
{
        //
        if constexpr (!training){
                int num_threads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
                if (num_threads < 1)
                        num_threads = 1;
                if ((size_t) num_threads > iterations)
                        num_threads = (int) iterations;

                std::atomic<int64_t> remaining_iters{static_cast<int64_t>(iterations)};
                std::vector<std::thread> workers;
                
                for (int t = 0; t < num_threads; t++) {
                        workers.emplace_back([this, nn, &remaining_iters, s = root_state.clone()]() mutable {
                                while (remaining_iters.fetch_sub(1, std::memory_order_relaxed) > 0)
                                        while(!perform_iteration(root_node, s.clone(), nn));
                        });
                }
                
                for (auto &t : workers) t.join();

        } else {
                if (!root_node->expanded ) {
                        perform_iteration(root_node, root_state.clone(), nn);
                }
                root_node->apply_dirichlet_noise(0.25f, 0.15f, gen);

                for (size_t i = 0; i < iterations; i++)
                        perform_iteration(root_node, root_state.clone(), nn);
        }
        
        // Selection of best move based on visit count
        MCTS<training>::SearchResult result;
        uint32_t max_visits = 0;
        bool move_found = false;
        
        auto children = root_node->children();

        if (children.empty())
                return result;
                        
        for (auto& child : children) {
                uint32_t v;
                if constexpr (!training)
                        v = child.visits.load();
                else
                        v = child.visits;
                int move_idx = encode_move(root_state, child.move); 
                result.visit_counts.push_back({move_idx, v});

                if (v > max_visits) {
                        max_visits = v;
                        result.best_move = child.move;
                        move_found = true;
                }
        }

        if (!move_found) {
                float max_policy = -1.0f;
                for (auto& child : children) {
                        if (child.prior > max_policy) {
                                max_policy = child.prior;
                                result.best_move = child.move;
                                move_found = true;
                        }
        }
        
                if (!move_found) {
                        std::cout << "CRITICAL: Search failed completely." << std::endl;
                        if (!children.empty()) result.best_move = children[0].move;
                }
        }

        if constexpr (!training)
                result.root_value = root_node->visits.load() > 0 ? 
                                      root_node->value_sum.load() / root_node->visits.load() : 0.0f;
        else 
                result.root_value = root_node->visits > 0 ? 
                                      root_node->value_sum / root_node->visits : 0.0f;
        return result;
}

template class MCTS<true>;
template class MCTS<false>;
