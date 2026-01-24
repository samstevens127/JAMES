#pragma once
#include <nshogi/core/position.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/types.h>
#include <nshogi/core/movelist.h>
#include <nshogi/io/sfen.h>
#include <torch/torch.h>
#include <vector>
#include <type_traits>
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
        std::vector<nshogi::core::Move32> legal_moves() const;

        bool is_terminal() const;
        float result() const; 
        void do_move(nshogi::core::Move32 m);
        void undo_move();

};

template<bool training>
class NodePool;

template<typename T, bool training>
using AtomicIfSearch = std::conditional_t<training, T, std::atomic<T>>;

template<bool training>
struct alignas(64) MCTSNode {
        nshogi::core::Move32 move;
        MCTSNode<training>* parent_ptr = nullptr;
        AtomicIfSearch<float, training> value_sum{0.0f}; // total value
        AtomicIfSearch<uint32_t, training> visits{0};

        float prior = 0.0f;
        uint8_t depth = 0;

        AtomicIfSearch<bool, training> expanded{false};

        AtomicIfSearch<int, training> virtual_loss{0};
        AtomicIfSearch<bool, training> is_expanding{false};

        std::span<MCTSNode<training>> children() const 
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
                float s,n,vl;
                if constexpr (!training) {
                        n = static_cast<float>(visits.load(std::memory_order_relaxed));
                        vl = static_cast<float>(virtual_loss.load(std::memory_order_relaxed));
                        s = value_sum.load(std::memory_order_relaxed);
                } else {
                        n = static_cast<float>(visits);
                        s = value_sum;
                        vl = 0.0f;
                }
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
                if constexpr (!training) {
                        visits.store(0, std::memory_order_relaxed);
                        value_sum.store(0.0f, std::memory_order_relaxed);
                        expanded.store(false, std::memory_order_relaxed);
                        is_expanding.store(false, std::memory_order_relaxed);
                        virtual_loss.store(0, std::memory_order_relaxed);
                } else {
                        visits = 0;
                        value_sum = 0.0;
                        expanded = false;
                }
        }
        
        void expand_with_policy(const GameState &gamestate, const at::Tensor &pi, NodePool<training> &pool);
        
        bool is_leaf() { return children().size() == 0; }

        private:
                MCTSNode<training>* children_data;
                uint32_t num_children;
};


template<bool training>
class NodePool {
        public:
                static constexpr size_t BLOCK_SIZE = 2e6; 
                
                NodePool() {
                        add_block();
                }
                
                MCTSNode<training>* allocate_single();
                MCTSNode<training>* allocate_slab(const GameState &gamestate, 
                                        const std::vector<nshogi::core::Move32> &moves,
                                        const at::Tensor &pi, 
                                        MCTSNode<training> *parent_node);
                
                
                void reset();
        private:
                std::deque<std::vector<MCTSNode<training>>> blocks;
                size_t current_block_idx = 0;
                size_t current_offset = 0;


                std::conditional_t<!training, std::mutex, bool> pool_mutex;
        
                void add_block() 
                {
                        blocks.emplace_back(BLOCK_SIZE);
                }
};
