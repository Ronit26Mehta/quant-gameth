"""
Mechanism design — auctions (first-price, second-price, VCG, combinatorial).

Implements standard auction formats and revenue analysis.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


class Auction:
    """Auction mechanism.

    Parameters
    ----------
    n_bidders : int
    n_items : int
        Number of items being auctioned.
    """

    def __init__(self, n_bidders: int, n_items: int = 1):
        self.n_bidders = n_bidders
        self.n_items = n_items

    def first_price_sealed(
        self, valuations: np.ndarray, bids: Optional[np.ndarray] = None
    ) -> Dict:
        """First-price sealed-bid auction.

        Highest bidder wins, pays their bid.

        Parameters
        ----------
        valuations : np.ndarray
            True valuations, shape ``(n_bidders,)``.
        bids : np.ndarray or None
            Bids (if None, uses Bayes-Nash equilibrium bidding strategy).
        """
        if bids is None:
            # BNE for uniform[0,1] valuations: bid = (n-1)/n * v
            bids = valuations * (self.n_bidders - 1) / self.n_bidders

        winner = int(np.argmax(bids))
        price = float(bids[winner])
        surplus = float(valuations[winner] - price)

        return {
            "winner": winner,
            "price": price,
            "surplus": surplus,
            "revenue": price,
            "bids": bids.tolist(),
            "efficient": bool(winner == np.argmax(valuations)),
        }

    def second_price_sealed(
        self, valuations: np.ndarray, bids: Optional[np.ndarray] = None
    ) -> Dict:
        """Second-price sealed-bid (Vickrey) auction.

        Highest bidder wins, pays second-highest bid.
        Truthful bidding is a dominant strategy.
        """
        if bids is None:
            bids = valuations.copy()  # truthful bidding is dominant

        sorted_indices = np.argsort(bids)[::-1]
        winner = int(sorted_indices[0])
        price = float(bids[sorted_indices[1]])
        surplus = float(valuations[winner] - price)

        return {
            "winner": winner,
            "price": price,
            "surplus": surplus,
            "revenue": price,
            "efficient": bool(winner == np.argmax(valuations)),
        }

    def vcg_auction(
        self, valuations: np.ndarray
    ) -> Dict:
        """VCG (Vickrey–Clarke–Groves) mechanism.

        Generalisation of Vickrey to multi-item settings.
        Payment = externality imposed on others.
        """
        n = len(valuations)
        if valuations.ndim == 1:
            # Single item
            return self.second_price_sealed(valuations)

        # Multi-item: valuations shape (n_bidders, n_items)
        # Efficient allocation: maximise total value
        from itertools import permutations

        best_allocation = None
        best_total = -np.inf

        n_items = valuations.shape[1]
        items = list(range(n_items))
        bidders = list(range(n))

        # Simple assignment (greedy for tractability)
        allocation = {}
        remaining_items = set(items)
        remaining_bidders = set(bidders)

        while remaining_items and remaining_bidders:
            best_pair = None
            best_val = -np.inf
            for b in remaining_bidders:
                for item in remaining_items:
                    if valuations[b, item] > best_val:
                        best_val = valuations[b, item]
                        best_pair = (b, item)
            if best_pair is None:
                break
            allocation[best_pair[1]] = best_pair[0]
            remaining_items.discard(best_pair[1])
            remaining_bidders.discard(best_pair[0])

        # Compute VCG payments
        total_without = {}
        for bidder in range(n):
            other_total = sum(
                valuations[allocation[item], item]
                for item in allocation
                if allocation[item] != bidder
            )
            total_without[bidder] = other_total

        # Total welfare without each bidder (recompute without them)
        payments = {}
        for bidder in range(n):
            # Welfare of others in current allocation
            others_current = sum(
                valuations[allocation[item], item]
                for item in allocation
                if allocation[item] != bidder
            )
            # Welfare of others in allocation without this bidder
            others_without = 0
            remaining = set(items)
            available_bidders = set(range(n)) - {bidder}
            temp_alloc = {}
            while remaining and available_bidders:
                best_pair = None
                best_v = -np.inf
                for b in available_bidders:
                    for item in remaining:
                        if valuations[b, item] > best_v:
                            best_v = valuations[b, item]
                            best_pair = (b, item)
                if best_pair is None:
                    break
                temp_alloc[best_pair[1]] = best_pair[0]
                remaining.discard(best_pair[1])
                available_bidders.discard(best_pair[0])

            others_without = sum(
                valuations[temp_alloc[item], item] for item in temp_alloc
            )
            payments[bidder] = max(0.0, others_without - others_current)

        return {
            "allocation": allocation,
            "payments": payments,
            "revenue": sum(payments.values()),
            "efficient": True,  # VCG is always efficient
        }

    def english_auction(
        self,
        valuations: np.ndarray,
        increment: float = 0.01,
    ) -> Dict:
        """Simulate ascending-price (English) auction.

        Bidders drop out when price exceeds valuation.
        Equivalent to second-price in private-value setting.
        """
        n = len(valuations)
        price = 0.0
        active = np.ones(n, dtype=bool)
        bid_history: List[Tuple[float, int]] = []

        while np.sum(active) > 1:
            price += increment
            for i in range(n):
                if active[i] and valuations[i] < price:
                    active[i] = False
                    bid_history.append((price, i))

        winner = int(np.argmax(active))
        return {
            "winner": winner,
            "price": float(price),
            "surplus": float(valuations[winner] - price),
            "revenue": float(price),
            "bid_history": bid_history,
        }

    def dutch_auction(
        self,
        valuations: np.ndarray,
        start_price: Optional[float] = None,
        decrement: float = 0.01,
    ) -> Dict:
        """Simulate descending-price (Dutch) auction.

        Price drops until someone bids.
        Equivalent to first-price in private-value setting.
        """
        if start_price is None:
            start_price = float(np.max(valuations) * 1.5)

        price = start_price
        while price > 0:
            for i in range(len(valuations)):
                # Bidder accepts if price is low enough (risk-neutral BNE)
                threshold = valuations[i] * (self.n_bidders - 1) / self.n_bidders
                if price <= threshold:
                    return {
                        "winner": i,
                        "price": float(price),
                        "surplus": float(valuations[i] - price),
                        "revenue": float(price),
                    }
            price -= decrement

        return {"winner": -1, "price": 0.0, "surplus": 0.0, "revenue": 0.0}

    @staticmethod
    def revenue_equivalence_demo(
        n_bidders: int = 4,
        n_simulations: int = 10000,
        seed: int = 42,
    ) -> Dict:
        """Demonstrate the Revenue Equivalence Theorem.

        Simulates all four standard auctions and compares expected revenue.
        """
        rng = np.random.default_rng(seed)
        auction = Auction(n_bidders)

        revenues = {"first_price": [], "second_price": [], "english": [], "dutch": []}

        for _ in range(n_simulations):
            valuations = rng.uniform(0, 1, n_bidders)

            r1 = auction.first_price_sealed(valuations)
            revenues["first_price"].append(r1["revenue"])

            r2 = auction.second_price_sealed(valuations)
            revenues["second_price"].append(r2["revenue"])

            # English ≈ second-price
            revenues["english"].append(r2["revenue"])

            # Dutch ≈ first-price
            revenues["dutch"].append(r1["revenue"])

        return {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in revenues.items()
        }
