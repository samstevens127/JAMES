#include "encoder.h"
#include "types.h"

static constexpr int MOVE_OFFSET = 0;
static constexpr int DROP_OFFSET = 81 * 81 * 2;  // 13122

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


// 0..13, independent of color
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

// 0..6
static inline int hand_piece_index(nshogi::core::PieceKind p) 
{
        using PK = nshogi::core::PieceKind;
        switch (p) {
                case PK::PK_BlackPawn:
                case PK::PK_WhitePawn:   return 0;
                case PK::PK_BlackLance:
                case PK::PK_WhiteLance:  return 1;
                case PK::PK_BlackKnight:
                case PK::PK_WhiteKnight: return 2;
                case PK::PK_BlackSilver:
                case PK::PK_WhiteSilver: return 3;
                case PK::PK_BlackGold:
                case PK::PK_WhiteGold:   return 4;
                case PK::PK_BlackBishop:
                case PK::PK_WhiteBishop: return 5;
                case PK::PK_BlackRook:
                case PK::PK_WhiteRook:   return 6;
                default: return -1;
        }
}


static void encode_hands(const GameState &gamestate,
                  EncodedState &out,
                  bool flip)
{
        using namespace nshogi::core;
        
        Color stm = gamestate.state->getSideToMove();
        Color opp = (stm == Black ? White : Black);
        
        const Stands &my_hand  = gamestate.state->getPosition().getStand(stm);
        const Stands &opp_hand = gamestate.state->getPosition().getStand(opp);
        
        auto encode_one = [&](const Stands &hand, bool opponent, int base_plane) {
                for (int pt = 0; pt < 7; ++pt) {
                    PieceTypeKind piece = static_cast<PieceTypeKind>(pt);
                    int count = getStandCount(hand,piece);
                    if (count == 0) continue;
                
                    int plane = base_plane + pt;
                    float value = std::min(1.0f, count / 4.0f);
                
                    for (int i = 0; i < 81; ++i) {
                        out[plane * 81 + i] = value;
                    }
        }
        };
        
        // planes:
        // 28–34 : side-to-move
        // 35–41 : opponent
        encode_one(my_hand,  false, 28);
        encode_one(opp_hand, true,  35);
}

EncodedState encode_state(const GameState &gamestate) 
{
        auto player_color = gamestate.state->getSideToMove();
        bool flip = (player_color == nshogi::core::White);
        
        EncodedState out{};
        out.fill(0.0f);
        
        for (nshogi::core::Square sq_ = nshogi::core::Sq1I; sq_ < nshogi::core::NumSquares; ++sq_) {
                int sq = flip ? (80 - sq_) : sq_;
                
                auto p = gamestate.state->getPosition().pieceOn(sq_);
                if (p == nshogi::core::PK_Empty) continue;
                
                bool is_white =
                        (p >= nshogi::core::PK_WhitePawn &&
                         p <= nshogi::core::PK_WhiteProRook);
                
                bool is_opponent =
                        (gamestate.state->getSideToMove() == nshogi::core::Black)
                                ? is_white
                                : !is_white;
                
                int type = piece_type_index(p);
                int plane = type + (is_opponent ? 14 : 0);
                
                int x = sq % 9;
                int y = sq / 9;
                
                out[plane * 81 + y * 9 + x] = 1.0f;
        }
        
        encode_hands(gamestate, out, flip);
        return out;
}
