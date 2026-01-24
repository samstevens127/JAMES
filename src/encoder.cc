#include "encoder.h"
#include "types.h"

static constexpr int MOVE_OFFSET = 0;
static constexpr int DROP_OFFSET = 81 * 81 * 2;  // 13122

static inline int piece_type_index(nshogi::core::PieceKind p) 
{
        using PK = nshogi::core::PieceKind;
        
        switch (p) {
                case PK::PK_BlackPawn:
                case PK::PK_WhitePawn:        return 0;
                case PK::PK_BlackLance:
                case PK::PK_WhiteLance:       return 1;
                case PK::PK_BlackKnight:
                case PK::PK_WhiteKnight:      return 2;
                case PK::PK_BlackSilver:
                case PK::PK_WhiteSilver:      return 3;
                case PK::PK_BlackGold:
                case PK::PK_WhiteGold:        return 4;
                case PK::PK_BlackBishop:
                case PK::PK_WhiteBishop:      return 5;
                case PK::PK_BlackRook:
                case PK::PK_WhiteRook:        return 6;
                case PK::PK_BlackProPawn:
                case PK::PK_WhiteProPawn:     return 7;
                case PK::PK_BlackProLance:
                case PK::PK_WhiteProLance:    return 8;
                case PK::PK_BlackProKnight:
                case PK::PK_WhiteProKnight:   return 9;
                case PK::PK_BlackProSilver:
                case PK::PK_WhiteProSilver:   return 10;
                case PK::PK_BlackProBishop:
                case PK::PK_WhiteProBishop:       return 11;
                case PK::PK_BlackProRook:
                case PK::PK_WhiteProRook:      return 12;
                case PK::PK_BlackKing:
                case PK::PK_WhiteKing:        return 13;
                default: return -1;
        }
}

int encode_move(const GameState& state, nshogi::core::Move32 move) 
{
        if (move.drop()) {
                int piece = move.pieceType();   // 0..6
                int to = move.to();              // 0..80
                return DROP_OFFSET + piece * 81 + to;
        } else {
                int from = move.from();          // 0..80
                int to = move.to();              // 0..80
                int promo = move.promote() ? 1 : 0;
                return MOVE_OFFSET + ((from * 81 + to) * 2 + promo);
        }
}

static inline int mirror_sq(int sq) 
{
        int x = sq % 9;
        int y = sq / 9;
        return y * 9 + (8 - x);
}

static void encode_hands_ptr(const GameState &gamestate,
                  float* ptr)
{
        using namespace nshogi::core;
        
        Color stm = gamestate.state->getSideToMove();
        Color opp = (stm == Black ? White : Black);
        
        const Stands &my_hand  = gamestate.state->getPosition().getStand(stm);
        const Stands &opp_hand = gamestate.state->getPosition().getStand(opp);
        
        auto encode_one = [&](const Stands &hand, int base_plane) {
                for (int pt = 0; pt < 7; ++pt) {
                    PieceTypeKind piece = static_cast<PieceTypeKind>(pt);
                    int count = getStandCount(hand,piece);
                    if (count == 0) continue;
                
                    int plane = base_plane + pt;
                    float value = std::min(1.0f, count / 4.0f);

                    float* plane_ptr = ptr + (plane * 81);
                
                    for (int i = 0; i < 81; ++i) {
                        plane_ptr[i] = value;
                    }
        }
        };
        
        encode_one(my_hand,  28);
        encode_one(opp_hand, 35);
}

int get_mirrored_move_index(int move_idx) 
{
        if (move_idx >= 13122) { 
                int drop_data = move_idx - 13122;
                int piece = drop_data / 81;
                int to_sq = drop_data % 81;
                return 13122 + (piece * 81) + mirror_sq(to_sq);
        } else { 
                int move_data = move_idx / 2;
                int promo = move_idx % 2;
                int from_sq = move_data / 81;
                int to_sq = move_data % 81;
                return ((mirror_sq(from_sq) * 81 + mirror_sq(to_sq)) * 2) + promo;
        }
}

template<bool mirrored>
void fill_tensor_data(const GameState &gamestate, float* ptr) 
{
        auto player_color = gamestate.state->getSideToMove();
        bool flip = (player_color == nshogi::core::White);
        
        for (nshogi::core::Square sq_ = nshogi::core::Sq1I; sq_ < nshogi::core::NumSquares; ++sq_) {
        	int sq = flip ? (80 - sq_) : sq_;
        	
        	if constexpr (mirrored) {
        	    int x = sq % 9;
        	    int y = sq / 9;
        	    sq = (8 - x) + (y * 9);
        	}
        
        	auto p = gamestate.state->getPosition().pieceOn(sq_);
        	if (p == nshogi::core::PK_Empty) continue;
        	
                bool is_white = (p >= nshogi::core::PK_WhitePawn && p <= nshogi::core::PK_WhiteProRook);
                bool is_opponent = (gamestate.state->getSideToMove() == nshogi::core::Black) ? is_white : !is_white;
                
                int type = piece_type_index(p);
                if (type == -1) continue;
                int plane = type + (is_opponent ? 14 : 0);
                
                ptr[plane * 81 + sq] = 1.0f;
        }

        encode_hands_ptr(gamestate, ptr);
}

EncodedState encode_state(const GameState &gamestate) 
{
        auto out = torch::zeros({48, 9, 9}, torch::kFloat32);
        // Don't mirror the board
        fill_tensor_data<false>(gamestate, out.data_ptr<float>());
        return out;
}

std::pair<EncodedState, EncodedState> encode_state_mirror(const GameState &gamestate) {
        auto normal = torch::zeros({48, 9, 9}, torch::kFloat32);
        auto mirrored = torch::zeros({48, 9, 9}, torch::kFloat32);
        
        fill_tensor_data<false>(gamestate, normal.data_ptr<float>());
        fill_tensor_data<true>(gamestate, mirrored.data_ptr<float>());
        
        return {normal, mirrored};
}
