import chess
import pandas as pd
import os
from typing import List, Dict, Any
from attack_value_full import AttackHistory, compute_attack_value_with_history
import multiprocessing as mp
import re
from tqdm import tqdm


num_cores = mp.cpu_count()
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "CSV files", "games.csv")
my_data = pd.read_csv(csv_path)
my_data = my_data[my_data['moves'].notna() & (my_data['moves'].str.strip() != '')]

#WARNING IF YOU USE THIS DATASET USE ONLE PART OF IT
#csv_path = os.path.join(script_dir, "CSV files", "lichess-08-2014.csv")
#my_data = pd.read_csv(csv_path)
#my_data = my_data.rename(columns={'PGN': 'moves'})
#my_data = my_data[my_data['moves'].notna() & (my_data['moves'].str.strip() != '')]#.take(range(100000)) 
#my_data["id"] = my_data.index + 1

df = my_data

file_save_name = 'TESTatcData.csv'


def clean_chess_notation(moves_string):
    if pd.isna(moves_string) or moves_string == '':
        return ''
    cleaned = re.sub(r'\d+\.', '', moves_string)
    cleaned = ' '.join(cleaned.split())
    return cleaned

if csv_path == os.path.join(script_dir, "CSV files", "lichess-08-2014.csv"):
    # file_save_name = 'TESTatcDataLICHESSS.csv'
    df['moves'] = df['moves'].apply(clean_chess_notation)
    file_save_name = 'TESTatcDataLichess.csv'



def _label_single_game(moves_str, id, opening, white_rating, black_rating, avg_rating, winner): #took out engine
    
    labeled: List[Dict[str, Any]] = []
    
    board = chess.Board()
    
    # Create history object for the entire game
    white_history = AttackHistory(history_length=5)
    black_history = AttackHistory(history_length=5)
    
    move_number = 0
    moves = str(moves_str).strip().split()
    for move_str in moves:
        move_number += 1

        try:
            move = board.parse_san(move_str)
        except Exception as e:
            print(f"Failed to parse move '{move_str}': {e}")
            break

        board.push(move)
        # Determine whose perspective to evaluate from
        side_to_move = board.turn  # chess.WHITE or chess.BLACK
            # Calculate attack value for WHITE
        if side_to_move == chess.BLACK:
            white_attack_value, white_components, white_raw, white_features = compute_attack_value_with_history(
            board, True, color=chess.WHITE, history=white_history
            )
            black_attack_value, black_components, black_raw, black_features = compute_attack_value_with_history(
                board, False, color=chess.BLACK, history=black_history
            )
        else:
            # Calculate attack value for BLACK
            black_attack_value, black_components, black_raw, black_features = compute_attack_value_with_history(
                board, True, color=chess.BLACK, history=black_history
            )
            white_attack_value, white_components, white_raw, white_features = compute_attack_value_with_history(
            board, False, color=chess.WHITE, history=white_history
            )
        
        fen = board.fen()
        if white_rating > 0:
            labeled.append({
                "game_id": id,
                "fen": fen,
                "move_number": move_number,
                "side_to_move": "white" if side_to_move == chess.WHITE else "black",
                "white_attack_value": white_attack_value,
                "black_attack_value": black_attack_value,
                "attack_differential": white_attack_value - black_attack_value,  # Net attack advantage
                "opening_name": opening,
                'white_rating': white_rating,
                "black_rating": black_rating,
                'result': winner

            })
        else:
            labeled.append({
                "game_id": id,
                "fen": fen,
                "move_number": move_number,
                "side_to_move": "white" if side_to_move == chess.WHITE else "black",
                "white_attack_value": white_attack_value,
                "black_attack_value": black_attack_value,
                "attack_differential": white_attack_value - black_attack_value,  # Net attack advantage
                'average_rating': avg_rating,
                'result': winner.split()[0].lower() if isinstance(winner, str) else winner
            })
    
    return labeled




def process_game(args):
    if len(args) == 6:
        game_id, moves_str, opening, white_rating, black_rating, winner = args
        return _label_single_game(moves_str, game_id, opening, white_rating, black_rating, 0, winner)
    elif len(args) == 4:
        game_id, moves_str, avg_rating, winner = args
        return _label_single_game(moves_str, game_id, 0, 0, 0, avg_rating, winner)
    else:
        print('Unexpected amout of arguments')

def process_data(df):
    try: 
        games_data = list[tuple](zip(df['id'].tolist(), df['moves'].tolist(), df['opening_name'].tolist(), df['white_rating'].tolist(), df['black_rating'].tolist(), df['winner'].tolist()))
    except:
        games_data = list[tuple](zip(df['id'].tolist(), df['moves'].tolist(), df['Average Rating'].tolist(), df['Result'].tolist()))

    
    if __name__ == "__main__":
        mp.freeze_support()  # required on Windows

        num_cores = mp.cpu_count()
        print(f"Starting multiprocessing with {num_cores} cores...")

        with mp.Pool(processes=num_cores) as pool:
            # results = pool.map(process_game, games_data)
            results = list(tqdm(pool.imap(process_game, games_data), total=len(games_data), desc="Processing games"))
        
        all_rows = [row for game_rows in results for row in game_rows]
        df_final = pd.DataFrame(all_rows)

    # Calculate incremental attack changes
        white_attack_val_inc = []
        black_attack_val_inc = []

        for i in range(df_final.shape[0]):
            if i == 0 or df_final.at[i, 'game_id'] != df_final.at[i-1, 'game_id']:
                white_attack_val_inc.append(df_final.at[i, 'white_attack_value'] - 2.745)
                black_attack_val_inc.append(df_final.at[i, 'black_attack_value'] - 2.745)
            elif df_final.at[i, 'side_to_move'] == 'black':
                white_attack_val_inc.append(df_final.at[i, 'white_attack_value'] - df_final.at[i-1, 'white_attack_value'])
                black_attack_val_inc.append(df_final.at[i, 'black_attack_value'] - df_final.at[i-1, 'black_attack_value'])
            else:
                black_attack_val_inc.append(df_final.at[i, 'black_attack_value'] - df_final.at[i-1, 'black_attack_value'])
                white_attack_val_inc.append(df_final.at[i, 'white_attack_value'] - df_final.at[i-1, 'white_attack_value'])
                insert_position = len(df_final.columns) - 1

# Insert new columns at the desired position
        df_final.insert(insert_position, "white_value_change", white_attack_val_inc)
        df_final.insert(insert_position + 1, "black_value_change", black_attack_val_inc)
    

# label = attack value for the player who just moved
        df_final["attack_value"] = df_final.apply(
            lambda row: row["white_attack_value"] if row["side_to_move"] == "black" else row["black_attack_value"],
            axis=1
        )
        output_path = os.path.join(script_dir, "CSV files", file_save_name)
        df_final.to_csv(output_path)
        
        print("Saved to " + output_path + ", "  + "if " + output_path + " already existed it was replaced")

process_data(df)
