"""Game theory engine sub-package — equilibria, dynamics, auctions, quantum games."""

from quant_gameth.games.normal_form import NormalFormGame, find_nash_equilibria
from quant_gameth.games.extensive_form import ExtensiveFormGame
from quant_gameth.games.minimax import minimax_solve
from quant_gameth.games.evolutionary import replicator_dynamics, moran_process
from quant_gameth.games.mechanism import Auction
from quant_gameth.games.quantum_games import ewl_quantum_game
from quant_gameth.games.repeated import iterated_game, tournament
from quant_gameth.games.cooperative import shapley_value
