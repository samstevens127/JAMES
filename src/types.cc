#include <algorithm>
#include <iterator>
#include "types.h"
#include "encoder.h"


bool GameState::is_terminal() const 
{
        auto ml = nshogi::core::MoveGenerator::generateLegalMoves(*state);
        return ml.size() == 0;
}

// TODO deal with tie
float GameState::result() const 
{
        auto ml = nshogi::core::MoveGenerator::generateLegalMoves(*state);
        
        if (ml.size() == 0)
                return -1.0f; 
        
        return 0.0f;
}

std::vector<nshogi::core::Move32> GameState::legal_moves() const 
{
        if (!state) return {};
        auto ml = nshogi::core::MoveGenerator::generateLegalMoves(*state);
        
        std::vector<nshogi::core::Move32> moves;
        moves.reserve(ml.size());
        for (auto m : ml) {
                moves.push_back(m);
        }
        return moves;
}


void GameState::do_move(nshogi::core::Move32 m) 
{
        if (m == nshogi::core::Move32()) {
        // We throw a runtime error so Python catches it, rather than segfaulting C++
        throw std::invalid_argument("Attempted to apply an invalid (empty) move in do_move");
    }
        state->doMove(m);
}

void GameState::undo_move() 
{
        state->undoMove();
}

void MCTSNode::expand_with_policy(const GameState &gamestate, 
                                  const std::vector<float> &policy, 
                                  NodePool &pool)
{
        auto moves = gamestate.legal_moves();
        if (moves.empty()) 
                return;
        
        MCTSNode* slab = pool.allocate_slab(gamestate, moves, policy, this);

        if (!slab) {
                std::cerr << "MCTSNodePool exhausted! Cannot expand node." << std::endl;
                return; 
        }

        children_data = slab;
        num_children = moves.size();

}

void NodePool::reset()
{
        std::lock_guard<std::mutex> lock(pool_mutex);
        current_block_idx = 0;
        current_offset = 0;
}

MCTSNode* NodePool::allocate_single() 
{
        std::lock_guard<std::mutex> lock(pool_mutex);

        if (current_offset + 1 > BLOCK_SIZE) {
                current_block_idx++;
                current_offset = 0;
                
                if (current_block_idx >= blocks.size()) {
                        add_block();
                }
        }

        MCTSNode* node = &blocks[current_block_idx][current_offset++];
        return node;
}

MCTSNode* NodePool::allocate_slab(const GameState &gamestate, 
                                        const std::vector<nshogi::core::Move32> &moves,
                                        const std::vector<float> &policy, 
                                        MCTSNode *parent_node) 
{
        size_t num_nodes = moves.size();
    
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        if (current_offset + num_nodes > BLOCK_SIZE) {
            current_block_idx++;
            current_offset = 0;
            
            if (current_block_idx >= blocks.size()) {
                add_block();
            }
        }
        
        
        MCTSNode* start = &blocks[current_block_idx][current_offset];
        current_offset += num_nodes;
        
        for (size_t i = 0; i < num_nodes; i++) {
            start[i].reset(moves[i]);
            start[i].parent_ptr = parent_node;
            int move_idx = encode_move(gamestate, moves[i]);
            start[i].prior = policy[move_idx];
        }
        
        return start;
}
