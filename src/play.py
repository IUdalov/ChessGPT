#!/usr/bin/env python3
import chess.engine

from chess_gpt import ChessGPT
from tokenizer import encode, GAME_LEN
import click



@click.command()
@click.option('-c', '--checkpoint', help='Path to checkpoint', type=click.Path(exists=True, dir_okay=False), required=True  )
@click.option('-s', '--start', help="Start of game.", required=True)
@click.option('--num-samples', default=10, type=click.INT)
@click.option('-n', '--num-moves', default=GAME_LEN, type=click.INT)
def main(checkpoint, start, num_samples, num_moves):
    chess_gpt = ChessGPT(checkpoint)

    games = chess_gpt.next_moves(start.split(" "), num_samples=num_samples, n_moves=num_moves)

    if games is None:
        print("Unable to continue the game")
        return

    for n, game in enumerate(games):
        print(f"Game #{n}", start + " " + " ".join(game))


if __name__ == "__main__":
    main()
    chess.IllegalMoveError