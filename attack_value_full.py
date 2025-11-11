from typing import Any, Tuple
import chess
from collections import deque
import numpy as np
import hashlib
import json

# individual peice weights
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
KING_VICINITY_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1), (0, 0), (0, 1),
                         (1, -1), (1, 0), (1, 1)]

# weights
WEIGHTS = {
    "attacked_pieces": 0.6,      
    "pinned_pieces": 2.0,
    "center_control": 0.3,
    "king_vicinity": 3.0,
    "open_file_pieces": 0.7,
    "multi_attacked": 1.5,
    "knight_pressure": 1.2,     
    "bishop_pressure": 0.8,     
    "rook_pressure": 0.9,      
    "check_bonus": 1.2
}

HISTORY_LENGTH = 5
DECAY_BASE = 0.65
CHECK_DECAY_BASE = 0.85
EMPTY_SQUARE_CONTROL_VALUE = 0.15  


def _hash_feature(obj):
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def calculate_check_bonus(board: chess.Board, color: bool) -> Tuple[float, dict]:
    
    opp_color = not color
    opp_king_sq = board.king(opp_color)
    
    if opp_king_sq is None:
        return 0.0, {"in_check": False}
    
    if not board.is_check():
        return 0.0, {"in_check": False}
    
    bonus = 0.4
    details = {
        "in_check": True,
        "base_bonus": 0.4,
        "king_mobility_reduction": 0.0,
        "edge_penalty": 0.0,
        "escape_difficulty": 0.0,
        "double_check": False,
        "material_loss_to_escape": False
    }
    
    checkers = board.checkers()
    if len(checkers) >= 2:
        bonus += 0.8 
        details["double_check"] = True
    
    legal_king_moves = 0
    king_square = board.king(opp_color)
    
    for move in board.legal_moves:
        if move.from_square == king_square:
            legal_king_moves += 1
    
    max_king_moves = 8
    mobility_reduction = (max_king_moves - legal_king_moves) / max_king_moves
    mobility_bonus = (mobility_reduction ** 1.5) * 0.6
    bonus += mobility_bonus
    details["king_mobility_reduction"] = mobility_bonus
    details["legal_king_moves"] = legal_king_moves
    
    king_rank = chess.square_rank(opp_king_sq)
    king_file = chess.square_file(opp_king_sq)
    
    on_edge = king_rank in [0, 7] or king_file in [0, 7]
    in_corner = (king_rank in [0, 7]) and (king_file in [0, 7])
    
    if in_corner:
        edge_bonus = 0.5 
        details["edge_penalty"] = 0.5
        details["king_position"] = "corner"
    elif on_edge:
        edge_bonus = 0.25 
        details["edge_penalty"] = 0.25
        details["king_position"] = "edge"
    else:
        edge_bonus = 0.0
        details["king_position"] = "center"
    
    bonus += edge_bonus
    
    if legal_king_moves == 0:
        escape_bonus = 1.0
        details["escape_difficulty"] = 1.0
        details["escape_note"] = "checkmate"
    elif legal_king_moves == 1:
        escape_bonus = 0.5 
        details["escape_difficulty"] = 0.5
        details["escape_note"] = "very_limited"
    elif legal_king_moves == 2:
        escape_bonus = 0.25 
        details["escape_difficulty"] = 0.25
        details["escape_note"] = "limited"
    else:
        escape_bonus = 0.0
        details["escape_note"] = "multiple_options"
    
    bonus += escape_bonus
    
    if legal_king_moves == 0 and board.legal_moves.count() > 0:
        bonus += 0.3
        details["forced_block_or_capture"] = True
    
    bonus = min(bonus, 2.5)
    details["final_bonus"] = bonus
    
    return bonus, details

def extract_feature_state(board: chess.Board, color: bool):
    opp_color = not color
    piece_map = board.piece_map()

    # Track which squares are attacked by specific piece types to avoid double counting
    knight_attacked_squares = set()
    bishop_attacked_squares = set()
    rook_attacked_squares = set()

    # --- Knight Pressure (attacks by knights) ---
    knight_pressure = []
    for sq, piece in piece_map.items():
        if piece.color == color and piece.piece_type == chess.KNIGHT:
            for target_sq in board.attacks(sq):
                knight_attacked_squares.add(target_sq)
                target_piece = board.piece_at(target_sq)
                if target_piece and target_piece.color == opp_color:
                    # Enemy piece attacked by knight
                    knight_pressure.append((sq, target_sq, PIECE_VALUES[target_piece.piece_type]))
                else:
                    # Empty square control
                    knight_pressure.append((sq, target_sq, EMPTY_SQUARE_CONTROL_VALUE))
    knight_pressure = tuple(sorted(knight_pressure))

    # --- Bishop Pressure (diagonal attacks by bishops and queens) ---
    bishop_pressure = []
    for sq, piece in piece_map.items():
        if piece.color == color and piece.piece_type in (chess.BISHOP, chess.QUEEN):
            for target_sq in board.attacks(sq):
                # Check if it's a diagonal attack
                if chess.square_file(target_sq) != chess.square_file(sq) and \
                   chess.square_rank(target_sq) != chess.square_rank(sq):
                    bishop_attacked_squares.add(target_sq)
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == opp_color:
                        # Enemy piece attacked diagonally
                        bishop_pressure.append((sq, target_sq, PIECE_VALUES[target_piece.piece_type]))
                    else:
                        # Empty square diagonal control
                        bishop_pressure.append((sq, target_sq, EMPTY_SQUARE_CONTROL_VALUE))
    bishop_pressure = tuple(sorted(bishop_pressure))

    #Rook Pressure (file/rank attacks by rooks and queens)
    rook_pressure = []
    for sq, piece in piece_map.items():
        if piece.color == color and piece.piece_type in (chess.ROOK, chess.QUEEN):
            for target_sq in board.attacks(sq):
                # Check if it's a file/rank attack (not diagonal)
                if chess.square_file(target_sq) == chess.square_file(sq) or \
                   chess.square_rank(target_sq) == chess.square_rank(sq):
                    rook_attacked_squares.add(target_sq)
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == opp_color:
                        # Enemy piece attacked on file/rank
                        rook_pressure.append((sq, target_sq, PIECE_VALUES[target_piece.piece_type]))
                    else:
                        # Empty square file/rank control
                        rook_pressure.append((sq, target_sq, EMPTY_SQUARE_CONTROL_VALUE))
    rook_pressure = tuple(sorted(rook_pressure))

    # --- Attacked opponent pieces (EXCLUDING squares already in piece-specific pressure) ---
    # This avoids double-counting tactical threats
    attacked_pieces = []
    for sq, piece in piece_map.items():
        if piece.color == opp_color and board.is_attacked_by(color, sq):
            # Only count if NOT already counted in specific piece pressure
            if sq not in knight_attacked_squares and \
               sq not in bishop_attacked_squares and \
               sq not in rook_attacked_squares:
                attacked_pieces.append((sq, piece.piece_type))
    attacked_pieces = tuple(sorted(attacked_pieces))

    # Pinned opponent pieces
    pinned_pieces = []
    for sq, piece in piece_map.items():
        if piece.color == opp_color and board.is_pinned(opp_color, sq):
            pinned_pieces.append((sq, piece.piece_type))
    pinned_pieces = tuple(sorted(pinned_pieces))

    # Center control
    center_control = tuple(sorted([sq for sq in CENTER_SQUARES if board.is_attacked_by(color, sq)]))

    # King vicinity
    opp_king_sq = board.king(opp_color)
    king_vicinity = tuple()
    if opp_king_sq is not None:
        kr = chess.square_rank(opp_king_sq)
        kf = chess.square_file(opp_king_sq)
        vic = []
        for dr, df in KING_VICINITY_OFFSETS:
            r = kr + dr
            f = kf + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                sq = chess.square(f, r)
                if board.is_attacked_by(color, sq):
                    vic.append(sq)
        king_vicinity = tuple(sorted(vic))

    # Open files
    open_file_pieces = []
    for file in range(8):
        pawns_on_file = [sq for sq, piece in piece_map.items() if piece.piece_type == chess.PAWN and chess.square_file(sq) == file]
        if len(pawns_on_file) == 0:
            for rank in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                if piece and piece.color == color and piece.piece_type in (chess.ROOK, chess.QUEEN):
                    open_file_pieces.append((sq, piece.piece_type))
    open_file_pieces = tuple(sorted(open_file_pieces))

    # Multi-attacker
    multi_attacked = []
    for sq, piece in piece_map.items():
        if piece.color == opp_color:
            attackers = board.attackers(color, sq)
            if len(attackers) >= 2:
                multi_attacked.append((sq, piece.piece_type, len(attackers)))
    multi_attacked = tuple(sorted(multi_attacked))

    features = {
        "attacked_pieces": attacked_pieces,
        "pinned_pieces": pinned_pieces,
        "center_control": center_control,
        "king_vicinity": king_vicinity,
        "open_file_pieces": open_file_pieces,
        "multi_attacked": multi_attacked,
        "knight_pressure": knight_pressure,
        "bishop_pressure": bishop_pressure,
        "rook_pressure": rook_pressure,
        "check_pattern": tuple()
    }

    # raw numeric components
    raw = {
        "attacked_value": sum(PIECE_VALUES[pt] for (_, pt) in attacked_pieces),
        "pinned_value": sum(PIECE_VALUES[pt] for (_, pt) in pinned_pieces),
        "center_count": len(center_control),
        "king_vicinity_count": len(king_vicinity),
        "open_file_value": sum(PIECE_VALUES[pt] for (_, pt) in open_file_pieces) * 0.5,
        "multi_attacked_count": sum(PIECE_VALUES[pt] for (_, pt, cnt) in multi_attacked),
        "knight_pressure_count": sum(val for (_, _, val) in knight_pressure),
        "bishop_pressure_count": sum(val for (_, _, val) in bishop_pressure),
        "rook_pressure_count": sum(val for (_, _, val) in rook_pressure),
    }

    return features, raw


class AttackHistory:
    def __init__(self, history_length=HISTORY_LENGTH):
        self.history_length = history_length
        self.recent = deque(maxlen=history_length)

    def push(self, features_by_side):
        hashed = {}
        for color, feat in features_by_side.items():
            hashed[color] = {k: _hash_feature(feat[k]) for k in feat.keys()}
        self.recent.append(hashed)

    def count_feature_occurrences(self, color, feature_name, feature_obj):
        h = _hash_feature(feature_obj)
        count = 0
        for entry in self.recent:
            if color in entry and feature_name in entry[color] and entry[color][feature_name] == h:
                count += 1
        return count

def compute_attack_value_with_history(board: chess.Board, edit, color: bool, history: AttackHistory):
    features, raw = extract_feature_state(board, color)
    
    check_bonus_value, check_details = calculate_check_bonus(board, color)
    raw["check_bonus"] = check_bonus_value
    
    if check_details.get("in_check", False):
        check_pattern = (
            "check",
            check_details.get("legal_king_moves", 8),
            check_details.get("double_check", False),
            check_details.get("king_position", "center")
        )
    else:
        check_pattern = ("no_check",)
    
    features["check_pattern"] = check_pattern

    components = {}
    total = 0.0

    for fname, base_val in [
        ("attacked_pieces", raw["attacked_value"]),
        ("pinned_pieces", raw["pinned_value"]),
        ("center_control", raw["center_count"]),
        ("king_vicinity", raw["king_vicinity_count"]),
        ("open_file_pieces", raw["open_file_value"]),
        ("multi_attacked", raw["multi_attacked_count"]),
        ("knight_pressure", raw["knight_pressure_count"]),
        ("bishop_pressure", raw["bishop_pressure_count"]),
        ("rook_pressure", raw["rook_pressure_count"]),
    ]:
        feat_obj = features.get(fname)
        if feat_obj is None:
            continue
            
        occurrences = history.count_feature_occurrences(color, fname, feat_obj)
        decay_multiplier = (DECAY_BASE ** occurrences) if occurrences > 0 else 1.0
        contrib = WEIGHTS.get(fname, 1.0) * base_val * decay_multiplier
        components[fname] = {
            "raw": base_val,
            "occurrences_in_history": occurrences,
            "decay_multiplier": decay_multiplier,
            "contribution": contrib
        }
        total += contrib

    # Add check bonus
    check_occurrences = history.count_feature_occurrences(color, "check_pattern", check_pattern)
    check_decay = (CHECK_DECAY_BASE ** check_occurrences) if check_occurrences > 0 else 1.0
    check_contribution = WEIGHTS.get("check_bonus", 1.0) * check_bonus_value * check_decay
    
    components["check_bonus"] = {
        "raw": check_bonus_value,
        "occurrences_in_history": check_occurrences,
        "decay_multiplier": check_decay,
        "contribution": check_contribution,
        "details": check_details
    }
    total += check_contribution

    features_by_side = {color: features}
    opp_features, _ = extract_feature_state(board, not color)
    features_by_side[not color] = opp_features
    if edit == True:
        history.push(features_by_side)

    return total, components, raw, features

#tensor for later nural network
def board_to_tensor(board: chess.Board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane_idx = color_offset + {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}[piece_type]
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        planes[plane_idx, r, f] = 1.0

    stm = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)
    castling = np.array([
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    ], dtype=np.float32)

    return planes, stm, castling

#optional
def save_board_data(board, color, history, path):
    attack_val, components, raw, features = compute_attack_value_with_history(board, color, history)
    planes, stm, castling = board_to_tensor(board)
    np.savez_compressed(path, planes=planes, stm=stm, castling=castling,
                        attack_value=attack_val, raw=raw, features=features, components=components)