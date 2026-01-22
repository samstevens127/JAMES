#include "mcts.h"
#include "encoder.h"
#include "types.h"
#include <limits>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

MCTS::MCTS(NodePool& p) : pool(p) 
{
        start_new_game();
}

void MCTS::start_new_game() 
{
        pool.reset();
        root_node = pool.allocate_single();
        root_node->reset();
}

void MCTS::update_root(const GameState& state, nshogi::core::Move32 move) {
        MCTSNode* next_root = nullptr;
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

// Static helper: Just picks the best child of the CURRENT node
static MCTSNode* select_child_puct(MCTSNode* node) 
{
        float max_puct = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        uint32_t parent_visits = node->visits.load(std::memory_order_relaxed);
        
        for (auto& child : node->children()) {
                float curr_puct = child.puct(parent_visits);
                
                if (curr_puct > max_puct) {
                        best_child = &child;
                        max_puct = curr_puct;
                }
        }
        return best_child;
}

void MCTS::perform_iteration(MCTSNode* root_node, GameState state, std::shared_ptr<NeuralNetwork> nn) 
{
        MCTSNode* node = root_node;
        
        // select with Virtual Loss
        while (node->expanded.load(std::memory_order_acquire) &&
                        !(node->children().size()) == 0) {
                node->virtual_loss.fetch_add(1, std::memory_order_relaxed); 
                node = select_child_puct(node);
                state.do_move(node->move);
        }
        
        // Add virtual loss before evaluating
        node->virtual_loss.fetch_add(1, std::memory_order_relaxed);
        
        float value = 0.0f;
        bool is_terminal = state.is_terminal();
        
        if (is_terminal) {
                value = state.result();
        } else {
                // evaluate
                if (!node->expanded.load(std::memory_order_acquire)) {
                        if (!node->expansion_lock.test_and_set(std::memory_order_acquire)) {
                                try{
                                        auto future = nn->evaluate_async(state);
                                        auto result = future.get(); 
                                        node->expand_with_policy(state, result.policy, pool);
                                        value = result.value;
                                        node->expanded.store(true, std::memory_order_release);
                                } catch(...) {
                                        node->expansion_lock.clear(std::memory_order_release);
                                        node->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                                        return;
                                }
                                
                } else {
                        while (!node->expanded.load(std::memory_order_acquire)) {
                    std::this_thread::yield(); 
                        }
                        while (node) {
                                node->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
                                node = node->parent_ptr;
                        }
                        return;
                        }
                } 
        
        // backprop & Remove Virtual Loss
        while (node != nullptr) {
                node->virtual_loss.fetch_sub(1, std::memory_order_relaxed); // Remove virtual loss
                
                if (!is_terminal && !node->expanded) {
                        // If we didn't expand (lost race), don't update stats
                        node = node->parent_ptr;
                        continue;
                }
                
                node->visits.fetch_add(1, std::memory_order_relaxed);
                add_to_atomic_float(node->value_sum, value);
                value = -value;
                node = node->parent_ptr;
        }
}
}
/* @brief performs search from root
 *
 * @params
 * GameState& root
 * shared_ptr<NeuralNetwork> nn
 * size_t iterations
 */

MCTS::SearchResult MCTS::search(const GameState &root_state,
                                std::shared_ptr<NeuralNetwork> nn,
                                size_t iterations) 
{
        //std::thread::hardware_concurrency()
        int num_threads = std::max(1, 128 - 1);
        if (num_threads < 1)
                num_threads = 1;
        if ((size_t) num_threads > iterations)
                num_threads = (int) iterations;

        std::atomic<int64_t> remaining_iters{static_cast<int64_t>(iterations)};
        std::vector<std::thread> workers;
        
        for (int t = 0; t < num_threads; t++) {
                workers.emplace_back([this, nn, &remaining_iters, s = root_state.clone()]() mutable {
                        while (remaining_iters.fetch_sub(1, std::memory_order_relaxed) > 0)
                                perform_iteration(root_node, s.clone(), nn);
                });
        }
        
        for (auto &t : workers) t.join();
        
        // Selection of best move based on visit count
        SearchResult result;
        uint32_t max_visits = 0;
        bool move_found = false;
        
        auto children = root_node->children();

        if (children.empty())
                return result;
                        
        for (auto& child : children) {
                uint32_t v = child.visits.load();
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

        result.root_value = root_node->visits.load() > 0 ? 
                              root_node->value_sum.load() / root_node->visits.load() : 0.0f;
        return result;
}
