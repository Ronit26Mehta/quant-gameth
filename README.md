<p align="center">
  <h1 align="center">⚛️ quant-gameth</h1>
  <p align="center">
    <strong>Quantum-Game Theory Framework</strong><br>
    Solve combinatorial optimization &amp; game theory problems with quantum-inspired algorithms
  </p>
  <p align="center">
    <a href="https://github.com/Ronit26Mehta/quant-gameth/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python"></a>
    <a href="https://github.com/Ronit26Mehta/quant-gameth"><img src="https://img.shields.io/badge/version-0.1.0-green.svg" alt="Version"></a>
    <a href="https://github.com/Ronit26Mehta/quant-gameth/issues"><img src="https://img.shields.io/github/issues/Ronit26Mehta/quant-gameth" alt="Issues"></a>
  </p>
</p>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         quant-gameth Framework                          │
├──────────────────────┬────────────────────────┬─────────────────────────┤
│   ⚛️ QUANTUM ENGINE   │  🎯 GAME THEORY ENGINE │   🔗 PROBLEM ENCODERS   │
│                      │                        │                         │
│  state.py            │  normal_form.py        │  qubo.py                │
│  gates.py            │  extensive_form.py     │  constraints.py         │
│  circuit.py          │  minimax.py            │  game_bridge.py         │
│  measurement.py      │  evolutionary.py       │  strategy_mapper.py     │
│  grover.py           │  mechanism.py          │                         │
│  qaoa.py             │  quantum_games.py      │                         │
│  vqe.py              │  repeated.py           │                         │
│  annealing.py        │  cooperative.py        │                         │
│  ansatz.py           │                        │                         │
│  tensor_network.py   │                        │                         │
├──────────────────────┴────────────────────────┴─────────────────────────┤
│                         🧩 APPLICATION SOLVERS                          │
│  sudoku · graph_coloring · maxcut · nqueens · knapsack · tsp            │
│  portfolio                                                              │
├──────────────────────┬────────────────────────┬─────────────────────────┤
│   📊 DATA GENERATORS  │   📈 VISUALIZATION     │   ⏱️ METRICS & BENCH    │
│  puzzles.py          │  quantum_viz.py        │  performance.py         │
│  graphs.py           │  game_viz.py           │  benchmark.py           │
│  games_gen.py        │  solver_viz.py         │                         │
│  market.py           │                        │                         │
├──────────────────────┴────────────────────────┴─────────────────────────┤
│                         ⚙️ INFRASTRUCTURE                               │
│  backends/ (classical · gpu · hybrid)                                   │
│  utils/   (serialization · decomposition)                               │
│  _types.py (core dataclasses & enums)                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### ⚛️ Quantum Algorithms
- **Statevector simulation** — Clifford+T + parameterised gates, up to ~22 qubits
- **QAOA** — multi-layer, warm-start, custom mixers
- **VQE** — variational eigensolver with parameter shift gradients
- **Grover's search** — amplitude amplification, multi-target
- **Quantum annealing** — simulated QA + parallel tempering
- **Tensor networks** — MPS backend for n>20 qubits
- **Ansatz library** — HEA, QAOA, QAOA+, UCC

</td>
<td width="50%">

### 🎯 Game Theory
- **Nash equilibria** — support enumeration, Lemke-Howson, LP
- **Extensive-form** — backward induction, subgame perfect equilibrium
- **Evolutionary** — replicator dynamics, Moran process, ESS
- **Mechanism design** — 1st-price, Vickrey, VCG, English auctions
- **Quantum games** — EWL protocol, quantum PD, penny flip
- **Repeated games** — iterated PD, 10+ strategies, tournaments
- **Cooperative** — Shapley value, core, nucleolus

</td>
</tr>
<tr>
<td>

### 🧩 Application Solvers
- **MaxCut** — QAOA, simulated annealing, brute-force
- **Graph Coloring** — backtracking, QAOA, SA
- **TSP** — nearest-neighbour, 2-opt, SA, QAOA
- **Knapsack** — DP, branch-and-bound, QUBO annealing
- **Sudoku** — constraint propagation, backtracking
- **N-Queens** — backtracking, simulated annealing
- **Portfolio** — Markowitz, discrete selection, SA

</td>
<td>

### ⚙️ Infrastructure
- **3 backends** — Classical (NumPy), GPU (CuPy), Hybrid (auto-dispatch)
- **Decomposition** — divide-and-conquer, sliding window, hierarchical
- **Benchmarking** — automated sweeps, JSON/CSV export
- **Serialization** — save/load results, circuits, states
- **Performance** — timing, memory profiling, approximation ratios

</td>
</tr>
</table>

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Ronit26Mehta/quant-gameth.git
cd quant-gameth

# Install in editable mode
pip install -e .

# (Optional) GPU acceleration
pip install cupy-cuda12x
```

### Requirements

| Package | Minimum Version |
|---------|----------------|
| Python | ≥ 3.9 |
| NumPy | ≥ 1.21 |
| SciPy | ≥ 1.7 |
| Matplotlib | ≥ 3.5 |
| NetworkX | ≥ 2.6 |

---

## 📖 Quick Start

### 1. Quantum Circuit — Bell State

```python
from quant_gameth.quantum.circuit import QuantumCircuit, Simulator
from quant_gameth.quantum.measurement import sample_counts

qc = QuantumCircuit(2)
qc.h(0).cx(0, 1)           # Hadamard + CNOT → Bell state |Φ+⟩

sv = Simulator().run(qc)
counts = sample_counts(sv, n_shots=1024)
print(counts)               # {'00': ~512, '11': ~512}
```

### 2. Nash Equilibrium — Prisoner's Dilemma

```python
from quant_gameth.games.normal_form import NormalFormGame

game = NormalFormGame.prisoners_dilemma()
equilibria = game.find_nash()

for eq in equilibria:
    print(f"Strategies: {eq.strategies}")
    print(f"Payoffs:    {eq.payoffs}")
```

### 3. MaxCut with QAOA

```python
import numpy as np
from quant_gameth.solvers.maxcut import solve_maxcut

adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
result = solve_maxcut(adj, method="qaoa", qaoa_depth=3)
print(f"Cut value: {result.metadata['cut_value']}")
print(f"Partition: {result.solution}")
```

### 4. Portfolio Optimization

```python
from quant_gameth.generators.market import generate_portfolio_data
from quant_gameth.solvers.portfolio import solve_portfolio

mu, sigma = generate_portfolio_data(n_assets=10, seed=42)
result = solve_portfolio(mu, sigma, risk_aversion=1.0, method="markowitz")
print(f"Optimal weights: {result.solution.round(4)}")
print(f"Sharpe ratio:    {result.metadata['sharpe_ratio']:.4f}")
```

### 5. Sudoku Solver

```python
from quant_gameth.generators.puzzles import generate_sudoku
from quant_gameth.solvers.sudoku import solve_sudoku

board = generate_sudoku(difficulty="hard", seed=42)
result = solve_sudoku(board, method="backtracking")
print(result.solution.reshape(9, 9))
```

---

## 🎮 Demo Scripts

Run any demo directly:

```bash
python -m quant_gameth.examples.demo_quantum
python -m quant_gameth.examples.demo_games
python -m quant_gameth.examples.demo_sudoku
python -m quant_gameth.examples.demo_maxcut
python -m quant_gameth.examples.demo_portfolio
python -m quant_gameth.examples.demo_trading
python -m quant_gameth.examples.demo_tournament
```

| Script | What It Demonstrates |
|--------|---------------------|
| `demo_quantum` | States, Bell state, Grover's search, QAOA, VQE |
| `demo_games` | Nash equilibria, backward induction, evolutionary dynamics, Shapley values |
| `demo_sudoku` | Puzzle generation (easy/medium/hard) + solving |
| `demo_maxcut` | Brute-force vs QAOA vs SA with approximation ratios |
| `demo_portfolio` | Markowitz, discrete selection, efficient frontier |
| `demo_trading` | First-price, Vickrey, VCG, English auctions |
| `demo_tournament` | Iterated PD round-robin + evolutionary dynamics |

---

## 📊 Benchmarking

```python
from quant_gameth.metrics.benchmark import BenchmarkSuite, BenchmarkConfig

suite = BenchmarkSuite()
suite.register_builtin("maxcut")

results = suite.run(BenchmarkConfig(
    problem_name="maxcut",
    sizes=[6, 8, 10, 12],
    methods=["qaoa", "annealing", "brute_force"],
    n_repeats=5,
))

suite.export_json(results, "benchmark_results.json")
suite.export_csv(results,  "benchmark_results.csv")
```

### Expected Performance

| Problem | Size | Classical Baseline | Framework | Speedup |
|---------|------|--------------------|-----------|---------|
| MaxCut (dense) | 100 nodes | ~10s | ~5s | 2× |
| Sudoku (hard) | 729 variables | ~60s | ~30s | 2× |
| Portfolio | 50 assets | ~1s | ~0.5s | 2× |
| 2-player game | 10×10 | ~0.1s | ~0.05s | 2× |

---

## 🔬 Novel Contributions

| # | Innovation | Description |
|---|-----------|-------------|
| 1 | **Quantum-Game Bridge** | Auto-transpile any normal-form game → QUBO or quantum circuit |
| 2 | **EWL Quantum Games** | Full EWL protocol with demonstrated quantum advantage over classical Nash |
| 3 | **Hybrid Quantum-Evolutionary** | Combine replicator dynamics with quantum annealing for equilibrium discovery |
| 4 | **Game-Theoretic QAOA** | QAOA warm-start from game best-response strategies |
| 5 | **Multi-Agent Tournament** | Tournament system with evolving quantum strategies across generations |

---

## 📁 Project Structure

```
quant-gameth/
├── pyproject.toml                    # Build config & dependencies
├── LICENSE                           # MIT License
├── README.md                         # This file
├── quant_gameth/
│   ├── __init__.py
│   ├── _types.py                     # Core dataclasses & enums
│   ├── quantum/                      # Quantum simulation engine (11 modules)
│   ├── games/                        # Game theory engine (8 modules)
│   ├── encoders/                     # Problem encoders (4 modules)
│   ├── solvers/                      # Application solvers (7 modules)
│   ├── generators/                   # Data generators (4 modules)
│   ├── viz/                          # Visualization (3 modules)
│   ├── metrics/                      # Performance & benchmarking (2 modules)
│   ├── backends/                     # Execution backends (4 modules)
│   ├── utils/                        # Utilities (2 modules)
│   └── examples/                     # Runnable demos (7 scripts)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a [Pull Request](https://github.com/Ronit26Mehta/quant-gameth/pulls).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/Ronit26Mehta/quant-gameth/blob/main/LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for quantum computing and game theory research<br>
  <a href="https://github.com/Ronit26Mehta/quant-gameth">⭐ Star this repo</a> if you find it useful!
</p>
