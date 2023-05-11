#!/usr/bin/env python3

from tokenizer import encode, decode, read_pgn_games, write_meta, GAME_LEN
import numpy as np
import sys
import os


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <in> <out>")
        return
    if not os.path.isdir(sys.argv[2]):
        os.mkdir(sys.argv[2])

    encoded = list()
    for n, game in enumerate(read_pgn_games(sys.argv[1])):
        if n % 2000 == 0:
            print(f"Reading {n} games")
        if n > 100:
            break

        moves = ["<beg>"] + [str(move) for move in game.mainline_moves()]
        encoded.append(encode(moves))

    ids = np.array(
        [e[:GAME_LEN + 1] + encode(max(0, GAME_LEN + 1 - len(e)) * ["<nop>"]) for e in encoded],
        dtype=np.uint16)

    print("Shape", ids.shape)
    n = len(encoded)
    train_ids = ids[:int(0.9*n)]
    train_path = os.path.join(sys.argv[2], 'train.bin')
    print(f"Writing shape {train_ids.shape} path {train_path}")
    train_ids.tofile(train_path)

    test_ids = ids[int(0.9 * n):]
    test_path = os.path.join(sys.argv[2], 'val.bin')
    print(f"Writing shape {test_ids.shape} path {test_path}")
    test_ids.tofile(test_path)

    write_meta(sys.argv[2])


if __name__ == "__main__":
    main()