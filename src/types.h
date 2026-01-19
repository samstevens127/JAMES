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
#include <span>
#include "encoder.h"



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
        
        std::atomic<int> virtual_loss{0};
        std::atomic<uint32_t> visits{0};
        std::atomic<bool> expanded{false};
        std::atomic<float> value_sum{0.0f}; // total value
        float prior = 0.0f;

        std::atomic_flag expansion_lock = ATOMIC_FLAG_INIT;

        std::span<MCTSNode> children() const 
        {
                return {children_data, num_children};
        }

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
                children_data = nullptr;
                num_children = 0;
                visits.store(0, std::memory_order_relaxed);
                value_sum.store(0.0f, std::memory_order_relaxed);
                expanded.store(false, std::memory_order_relaxed);
                virtual_loss.store(0, std::memory_order_relaxed);
        }
        
        void expand_with_policy(const GameState &gamestate, const std::vector<float> &pi, NodePool &pool);
        
        bool is_leaf() { return children().size() == 0; }

        private:
                MCTSNode* children_data;
                uint32_t num_children;
};


class NodePool {
        public:
                static constexpr size_t BLOCK_SIZE = 2e6; 
                
                NodePool() {
                        add_block();
                }
                
                MCTSNode* allocate_single();
                
                MCTSNode* allocate_slab(const GameState &gamestate, 
                                        const std::vector<nshogi::core::Move32> &moves,
                                        const std::vector<float> &policy, 
                                        MCTSNode *parent_node);
                
                
                void reset();
        private:
                std::deque<std::vector<MCTSNode>> blocks;
                size_t current_block_idx = 0;
                size_t current_offset = 0;

                std::mutex pool_mutex;
        
                void add_block() 
                {
                        blocks.emplace_back(BLOCK_SIZE);
                }
};
