#pragma once
#include <nshogi/core/position.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/types.h>
#include <nshogi/core/movelist.h>
#include <nshogi/io/sfen.h>
#include <vector>
#include <deque>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <iostream>


using EncodedState = std::array<float, 48 * 9 * 9>;

struct GameState {
        
        
        explicit GameState(nshogi::core::State&& s)
          : state(std::make_unique<nshogi::core::State>(std::move(s))) {}
        
        GameState clone() const 
        {
                if (!state) std::cout << "ERROR NULLPTR IN CLONE" << std::endl;

                auto cloned_raw = state->clone(); 
                return GameState(std::move(cloned_raw));
        }
        
        
        GameState(const GameState&) = delete;
        GameState& operator=(const GameState&) = delete;
        
        GameState(GameState&&) = default;
        GameState& operator=(GameState&& other) = default;
        
        std::unique_ptr<nshogi::core::State> state;
        bool is_terminal() const;
        
        float result() const; 
        std::vector<nshogi::core::Move32> legal_moves() const;
        
        void do_move(nshogi::core::Move32 m);
        void undo_move();
};

class NodePool;

struct alignas(64) MCTSNode {
        nshogi::core::Move32 move;
        MCTSNode* parent_ptr = nullptr;
        std::vector<MCTSNode*> children;
        
        std::atomic<int> virtual_loss{0};
        std::atomic<uint32_t> visits{0};
        std::atomic<bool> expanded{false};
        std::atomic<float> value_sum{0.0f}; // total value
        float prior = 0.0f;
        std::mutex expansion_mtx;
        inline float value() 
        { 
                return visits == 0 ? 0.0f : value_sum / visits;
        };
        
        template <float cpuct = 2.5f>
        float puct(uint32_t &parent_visits) const 
        {
                float n = static_cast<float>(visits.load(std::memory_order_relaxed));
                float vl = static_cast<float>(virtual_loss.load(std::memory_order_relaxed));
                float s = value_sum.load(std::memory_order_relaxed);
                float q = (n == 0.0) ? 0.0f : s / n;
                float u = cpuct * prior *
                std::sqrt((float)parent_visits) / (1.0f + n + vl);
                return q + u;
        }
        
        void reset(nshogi::core::Move32 m = nshogi::core::Move32()) {
                move = m;
                parent_ptr = nullptr;
                children.clear(); 
                visits.store(0, std::memory_order_relaxed);
                value_sum.store(0.0f, std::memory_order_relaxed);
                expanded.store(false, std::memory_order_relaxed);
                virtual_loss.store(0, std::memory_order_relaxed);
        }
        
        void expand_with_policy(const GameState &state, const std::vector<float> &pi, NodePool &pool);
        
        bool is_leaf() { return children.empty(); }
};

class NodePool {
        public:
                static constexpr size_t BLOCK_SIZE = 100000000; 
                
                NodePool() {
                        add_block();
                }
                
                MCTSNode* allocate_single() 
                {
                        size_t idx = current_idx.fetch_add(1, std::memory_order_relaxed);
                        
                        if (idx >= BLOCK_SIZE) {
                                return nullptr; 
                        }
                        return &blocks.back()[idx];
                }
                
                MCTSNode* allocate_slab(const size_t num_nodes) 
                {
                        size_t idx = current_idx.fetch_add(num_nodes, std::memory_order_relaxed);
                        if (idx + num_nodes > BLOCK_SIZE) return nullptr;
                        MCTSNode* start = &blocks.back()[idx];
                        
                        for (size_t i = 0; i < num_nodes; i++) {
                                start[i].reset();
                        }
                        
                        return start;
                }
                
                void reset() 
                {
                        current_idx.store(0, std::memory_order_relaxed);
                }
        private:
                std::deque<std::vector<MCTSNode>> blocks;
                std::atomic<size_t> current_idx{0};
                std::mutex expansion_mutex;
        
                void add_block() 
                {
                        std::lock_guard<std::mutex> lock(expansion_mutex);
                        blocks.emplace_back(BLOCK_SIZE);
                }
};
