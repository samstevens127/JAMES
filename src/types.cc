#include "types.h"
#include "encoder.h"

std::mutex global_nshogi_mutex;

bool GameState::is_terminal() const 
{
        std::lock_guard<std::mutex> lock(global_nshogi_mutex); // Temporary test
        auto ml = nshogi::core::MoveGenerator::generateLegalMoves(*state);
        return ml.size() == 0;
}

// TODO deal with tie
float GameState::result() const 
{
        std::lock_guard<std::mutex> lock(global_nshogi_mutex);
        auto ml = nshogi::core::MoveGenerator::generateLegalMoves(*state);
        
        if (ml.size() == 0)
                return -1.0f; // side to move is checkmated
        
        return 0.0f;
}

std::vector<nshogi::core::Move32> GameState::legal_moves() const 
{
        std::lock_guard<std::mutex> lock(global_nshogi_mutex);
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
        std::lock_guard<std::mutex> lock(global_nshogi_mutex);
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
        
        MCTSNode* slab = pool.allocate_slab(moves.size());
        
        for (size_t i = 0; i < moves.size(); ++i) {
                MCTSNode* child = &slab[i];
                // Reset child state 
                child->visits.store(0);
                child->value_sum.store(0.0f);
                child->expanded.store(false);
                child->children.clear();
                
                child->move = moves[i];
                child->parent_ptr = this;
                
                int move_idx = encode_move(gamestate, moves[i]);
                child->prior = policy[move_idx];
                
                this->children.push_back(child);
        }
}
