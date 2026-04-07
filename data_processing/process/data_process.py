# Method to process the data for tokenizing later
import chess
import json
from files.chess_moves import CHESS_MOVES

TOKEN_MAP = {
class ChessGame:
    def __init__(self, moves, move_times, white_elo, black_elo, term, time_ctrl, outcome):
        self.moves = moves
        self.move_times = move_times
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.term = term
        self.time_ctrl = time_ctrl
        self.outcome = outcome

def load_data(file_path):
    games = []
    with open(file_path, 'r') as file:
        for line in file:
            games.append(json.loads(line))
    return games

def process_data():
    data = load_data('data-processing/files/2022-01-test.jsonl')
    for item in data:
        game = ChessGame(
            moves=item['moves-uci'].split(),
            move_times=item['moves-seconds'],
            white_elo=int(item['white-elo']),
            black_elo=int(item['black-elo']),
            term=item['termination'],
            time_ctrl=item['time-control'],
            outcome=item['result'],
        )
        games.append(game)
    return games