import chess
import chess.pgn
import pickle
import os

GAME_LEN = 256

def read_pgn_games(pgn_path):
    with open(pgn_path, "r") as f:
        game = chess.pgn.read_game(f)
        while game is not None:
            yield game
            game = chess.pgn.read_game(f)


def _stoi_itos():
    itof = lambda i: chr(ord('a') + i % 8) + f"{1 + int(i / 8)}"
    stoi = {itof(i): i for i in range(64)}
    stoi["q"] = len(stoi)
    stoi["n"] = len(stoi)
    stoi["b"] = len(stoi)
    stoi["r"] = len(stoi)
    stoi["<beg>"] = len(stoi)
    stoi["<nop>"] = len(stoi)
    itos = {i: s for s, i in stoi.items()}

    return stoi, itos


_STOI, _ITOS = _stoi_itos()


def write_meta(path):
    meta_path = os.path.join(path, 'meta.pkl')
    print(f"Writing meta {meta_path}")
    with open(meta_path, 'wb') as f:
        pickle.dump({ 'vocab_size': len(_STOI)}, f)


def encode(moves):
    tokens = list()
    for move in moves:
        move = str(move)
        if move[0] == "<":
            tokens.append(_STOI[move])
        else:
            tokens.append(_STOI[move[:2]])
            tokens.append(_STOI[move[2:4]])
            if len(move) == 5:
                tokens.append(_STOI[move[4]])
    return tokens


def decode(tokens):
    moves = list()
    buff = list()
    for token in tokens:
        decoded = _ITOS[token]
        if decoded[0] == "<":
            if len(buff) != 0:
                moves.append("".join(buff))
                buff = list()
            moves.append(decoded)
        elif len(decoded) == 1:
            buff.append(decoded)
            moves.append("".join(buff))
            buff = list()
        elif len(buff) == 2 and len(decoded) == 2:
            moves.append("".join(buff))
            buff = [decoded]
        else:
            buff.append(decoded)

    if len(buff) != 0:
        moves.append("".join(buff))
        buff = list()
    return moves


def _do_arrays_match(a, b):
    assert len(a) == len(b)
    for i, a_e in enumerate(a):
        if a_e != b[i]:
            return False
    return True


if __name__ == "__main__":
    game1 = ["<beg>", "e2e4", "e6e7", "a7a8r", "<nop>"]
    assert _do_arrays_match(game1, decode(encode(game1)))

    assert len(encode(100 * ["<nop>"])) == 100

    print("Tests passed")