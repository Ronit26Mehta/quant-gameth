"""
Extensive-form games — game trees with backward induction and subgame-perfect equilibrium.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import EquilibriumResult, SolverMethod


class GameNode:
    """A node in an extensive-form game tree.

    Parameters
    ----------
    player : int or None
        Which player acts at this node (``None`` for terminal nodes).
    actions : list of str
        Available actions at this node.
    children : dict
        ``{action: child_node}``
    payoffs : np.ndarray or None
        Payoff vector at terminal nodes, shape ``(n_players,)``.
    name : str
        Identifier for this node.
    info_set : str or None
        Information set label (for imperfect information games).
    """

    def __init__(
        self,
        player: Optional[int] = None,
        actions: Optional[List[str]] = None,
        children: Optional[Dict[str, "GameNode"]] = None,
        payoffs: Optional[np.ndarray] = None,
        name: str = "",
        info_set: Optional[str] = None,
    ):
        self.player = player
        self.actions = actions or []
        self.children = children or {}
        self.payoffs = payoffs
        self.name = name
        self.info_set = info_set

    @property
    def is_terminal(self) -> bool:
        return self.payoffs is not None

    def add_child(self, action: str, child: "GameNode") -> None:
        self.actions.append(action)
        self.children[action] = child

    def __repr__(self) -> str:
        if self.is_terminal:
            return f"Terminal({self.name}, payoffs={self.payoffs})"
        return f"GameNode({self.name}, player={self.player}, actions={self.actions})"


class ExtensiveFormGame:
    """Extensive-form game with tree representation.

    Parameters
    ----------
    root : GameNode
    n_players : int
    name : str
    """

    def __init__(
        self,
        root: GameNode,
        n_players: int = 2,
        name: str = "extensive_game",
    ):
        self.root = root
        self.n_players = n_players
        self.name = name

    @classmethod
    def ultimatum_game(cls, total: float = 10.0, n_offers: int = 11) -> "ExtensiveFormGame":
        """Classic ultimatum game: P1 offers, P2 accepts/rejects."""
        root = GameNode(player=0, name="proposer")
        offers = np.linspace(0, total, n_offers)
        for offer in offers:
            action = f"offer_{offer:.1f}"
            p2_node = GameNode(player=1, name=f"responder_{offer:.1f}")
            accept_payoff = np.array([total - offer, offer])
            reject_payoff = np.array([0.0, 0.0])
            p2_node.add_child("accept", GameNode(payoffs=accept_payoff, name="accept"))
            p2_node.add_child("reject", GameNode(payoffs=reject_payoff, name="reject"))
            root.add_child(action, p2_node)
        return cls(root, n_players=2, name="ultimatum")

    @classmethod
    def centipede_game(cls, n_rounds: int = 6, pot: float = 1.0, growth: float = 2.0) -> "ExtensiveFormGame":
        """Centipede game: alternating take/pass with growing pot."""
        current_pot = pot

        def build_node(round_idx: int, pot_size: float) -> GameNode:
            player = round_idx % 2
            if round_idx >= n_rounds:
                # Final node: split the pot
                return GameNode(payoffs=np.array([pot_size / 2, pot_size / 2]),
                                name=f"split_{round_idx}")
            node = GameNode(player=player, name=f"round_{round_idx}")
            # Take: current player gets more
            take_payoff = np.zeros(2)
            take_payoff[player] = pot_size * 0.6
            take_payoff[1 - player] = pot_size * 0.4
            node.add_child("take", GameNode(payoffs=take_payoff, name=f"take_{round_idx}"))
            # Pass: pot grows
            node.add_child("pass", build_node(round_idx + 1, pot_size * growth))
            return node

        root = build_node(0, pot)
        return cls(root, n_players=2, name="centipede")

    def backward_induction(self) -> EquilibriumResult:
        """Find subgame-perfect equilibrium via backward induction."""
        t0 = time.perf_counter()
        strategies: Dict[str, str] = {}

        def solve(node: GameNode) -> np.ndarray:
            if node.is_terminal:
                return node.payoffs

            best_action = None
            best_payoffs = None
            for action in node.actions:
                child = node.children[action]
                child_payoffs = solve(child)
                if best_payoffs is None or child_payoffs[node.player] > best_payoffs[node.player]:
                    best_payoffs = child_payoffs
                    best_action = action

            strategies[node.name] = best_action
            return best_payoffs

        equilibrium_payoffs = solve(self.root)
        elapsed = time.perf_counter() - t0

        # Convert strategies to arrays (one-hot per info set)
        strat_arrays = []
        for p in range(self.n_players):
            player_strats = {k: v for k, v in strategies.items() if k.startswith(f"round_{p}") or (p == 0 and "proposer" in k) or (p == 1 and "responder" in k)}
            if player_strats:
                strat_arrays.append(np.array([hash(str(player_strats)) % 1000]))
            else:
                strat_arrays.append(np.array([0.0]))

        return EquilibriumResult(
            strategies=strat_arrays,
            payoffs=equilibrium_payoffs,
            equilibrium_type="subgame_perfect",
            method=SolverMethod.BACKTRACKING,
            time_seconds=elapsed,
            metadata={"strategy_profile": strategies},
        )

    def count_nodes(self) -> int:
        """Total number of nodes in the game tree."""
        count = 0

        def visit(node: GameNode) -> None:
            nonlocal count
            count += 1
            for child in node.children.values():
                visit(child)

        visit(self.root)
        return count

    def tree_depth(self) -> int:
        """Maximum depth of the game tree."""
        def depth(node: GameNode) -> int:
            if node.is_terminal:
                return 0
            return 1 + max(depth(c) for c in node.children.values())
        return depth(self.root)

    def __repr__(self) -> str:
        return f"ExtensiveFormGame({self.name}, nodes={self.count_nodes()})"
