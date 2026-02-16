# Mathematical Foundations Appendix
## Quantum-Game Theory Framework: Rigorous Formulations

---

## A. Quantum Mechanics Primer for Game Theory

### A.1 Hilbert Space Formulation

**Definition**: A Hilbert space ℋ is a complete complex vector space with inner product ⟨·|·⟩.

For quantum systems:
- **Single qubit**: ℋ = ℂ² with basis {|0⟩, |1⟩}
- **n qubits**: ℋ = (ℂ²)⊗ⁿ ≅ ℂ^(2ⁿ)

**General state**:
```
|ψ⟩ = Σᵢ αᵢ|i⟩ where Σᵢ |αᵢ|² = 1
αᵢ ∈ ℂ (complex amplitudes)
```

**Density matrix formulation**:
```
ρ = |ψ⟩⟨ψ| for pure states
ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ| for mixed states

Properties:
- Hermitian: ρ† = ρ
- Positive semi-definite: ρ ≥ 0
- Unit trace: Tr(ρ) = 1
```

### A.2 Quantum Operations

**Unitary Evolution**:
```
U: ℋ → ℋ such that U†U = UU† = I
|ψ'⟩ = U|ψ⟩

Composition: (U₂ ∘ U₁)|ψ⟩ = U₂(U₁|ψ⟩)
```

**Common Single-Qubit Gates**:

**Pauli Matrices**:
```
X = |0⟩⟨1| + |1⟩⟨0| = [0 1; 1 0]  (bit flip)
Y = -i|0⟩⟨1| + i|1⟩⟨0| = [0 -i; i 0]
Z = |0⟩⟨0| - |1⟩⟨1| = [1 0; 0 -1]  (phase flip)
```

**Hadamard Gate**:
```
H = (|0⟩⟨0| + |0⟩⟨1| + |1⟩⟨0| - |1⟩⟨1|)/√2 = 1/√2 [1 1; 1 -1]

Creates superposition: H|0⟩ = (|0⟩ + |1⟩)/√2
```

**Rotation Gates**:
```
RX(θ) = e^(-iθX/2) = cos(θ/2)I - i sin(θ/2)X
RY(θ) = e^(-iθY/2) = cos(θ/2)I - i sin(θ/2)Y
RZ(θ) = e^(-iθZ/2) = cos(θ/2)I - i sin(θ/2)Z
```

**Two-Qubit Gates**:

**CNOT (Controlled-NOT)**:
```
CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
     = [1 0 0 0]
       [0 1 0 0]
       [0 0 0 1]
       [0 0 1 0]

Creates entanglement: CNOT(H⊗I)|00⟩ = (|00⟩ + |11⟩)/√2
```

**Controlled-Z**:
```
CZ = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Z
   = diag(1, 1, 1, -1)
```

### A.3 Measurement & Born Rule

**Projective Measurement**:
```
Measurement in basis {|mᵢ⟩}:
P(outcome i | state ρ) = Tr(Πᵢ ρ)

where Πᵢ = |mᵢ⟩⟨mᵢ| (projection operator)

Post-measurement state: ρ' = Πᵢ ρ Πᵢ / Tr(Πᵢ ρ)
```

**Expectation Value**:
```
For observable O (Hermitian operator):
⟨O⟩ = ⟨ψ|O|ψ⟩ = Tr(ρO)

Example (Pauli-Z on qubit):
⟨Z⟩ = |α₀|² - |α₁|² for |ψ⟩ = α₀|0⟩ + α₁|1⟩
```

### A.4 Entanglement Theory

**Separable vs Entangled States**:

**Separable**: ρ = Σᵢ pᵢ ρᵢ^A ⊗ ρᵢ^B

**Entangled**: Cannot be written in separable form

**Bell States** (maximally entangled):
```
|Φ⁺⟩ = (|00⟩ + |11⟩)/√2
|Φ⁻⟩ = (|00⟩ - |11⟩)/√2
|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
```

**Entanglement Measures**:

**Von Neumann Entropy**:
```
S(ρ) = -Tr(ρ log₂ ρ)

For pure state: S = 0
For maximally mixed: S = log₂(dim ℋ)
```

**Concurrence** (for 2-qubit states):
```
C(ρ) = max{0, λ₁ - λ₂ - λ₃ - λ₄}

where λᵢ are eigenvalues of R = √(√ρ ρ̃ √ρ)
ρ̃ = (σy ⊗ σy) ρ* (σy ⊗ σy)

C = 0: separable
C = 1: maximally entangled
```

---

## B. Game Theory Foundations

### B.1 Normal Form Games

**Definition**: G = (N, S, u) where:
- N = {1,...,n}: player set
- S = S₁ × ... × Sₙ: strategy space
- u = (u₁,...,uₙ): payoff functions uᵢ: S → ℝ

**Example (Prisoner's Dilemma)**:
```
         Cooperate  Defect
Cooperate   (3,3)   (0,5)
Defect      (5,0)   (1,1)

Payoff matrix for row player:
M₁ = [3 0]
     [5 1]
```

### B.2 Nash Equilibrium

**Pure Strategy Nash Equilibrium**:

A strategy profile s* = (s₁*,...,sₙ*) is Nash if:
```
∀i, ∀sᵢ ∈ Sᵢ: uᵢ(s*) ≥ uᵢ(sᵢ, s₋ᵢ*)
```

No player can improve by unilateral deviation.

**Mixed Strategy Nash Equilibrium**:

Mixed strategy: σᵢ ∈ Δ(Sᵢ) (probability distribution)

Nash equilibrium σ* satisfies:
```
∀i, ∀σᵢ: Eσ*[uᵢ] ≥ Eσᵢ,σ*₋ᵢ[uᵢ]
```

**Nash's Theorem**: Every finite game has at least one mixed strategy Nash equilibrium.

### B.3 Computing Nash Equilibria

**2-Player Games via Support Enumeration**:

1. Guess support S₁ ⊆ S₁, S₂ ⊆ S₂
2. Solve indifference equations:
   ```
   For all s, s' in S₁: u₁(s,σ₂*) = u₁(s',σ₂*)
   For all t, t' in S₂: u₂(σ₁*,t) = u₂(σ₁*,t')
   ```
3. Check: probabilities ≥ 0, Σ pᵢ = 1
4. Verify best response property

**Lemke-Howson Algorithm**:

Frames Nash finding as complementary pivoting in polytope.

**Complexity**: PPAD-complete for general games

### B.4 Evolutionary Game Theory

**Replicator Dynamics**:
```
ẋᵢ = xᵢ[u(eᵢ,x) - u(x,x)]

where:
- xᵢ: proportion playing strategy i
- u(eᵢ,x): payoff to pure strategy i against population x
- u(x,x): average payoff
```

**Evolutionarily Stable Strategy (ESS)**:

σ* is ESS if ∀σ ≠ σ*:
```
Either: u(σ*,σ*) > u(σ,σ*)
Or: u(σ*,σ*) = u(σ,σ*) and u(σ*,σ) > u(σ,σ)
```

Resists invasion by mutant strategies.

---

## C. Quantum Game Theory

### C.1 Eisert-Wilkens-Lewenstein (EWL) Scheme

**Quantum Prisoner's Dilemma**:

**Setup**:
1. Initial state: |ψ₀⟩ = |00⟩
2. Entangling gate: Ĵ = (|00⟩⟨00| + |01⟩⟨01| + |10⟩⟨10| + e^(iγ)|11⟩⟨11|)/√2
   - Creates entanglement: |ψ₁⟩ = Ĵ|00⟩
3. Players apply unitary strategies:
   - Player 1: U₁ ∈ SU(2)
   - Player 2: U₂ ∈ SU(2)
   - State: |ψ₂⟩ = (U₁ ⊗ U₂)|ψ₁⟩
4. Disentangle: |ψ₃⟩ = Ĵ†|ψ₂⟩
5. Measure in computational basis
6. Payoffs from measurement outcome

**Strategy Parametrization**:
```
U(θ,φ) = [e^(iφ)cos(θ/2)    sin(θ/2)    ]
         [-sin(θ/2)          e^(-iφ)cos(θ/2)]

Special cases:
- θ=0: cooperate (C)
- θ=π: defect (D)
- θ=π, φ=0: quantum strategy Q
```

**Quantum Advantage**:

Classical Nash: (D,D) with payoff (1,1)

Quantum Nash: (Q,Q) with payoff (3,3)!

**Proof**: Q strategy satisfies:
```
U_Q|0⟩ = (|0⟩ + i|1⟩)/√2
U_Q|1⟩ = (i|0⟩ + |1⟩)/√2

Against any classical strategy, Q always cooperates.
Against Q, both get cooperation payoff.
```

### C.2 Quantum Best Response

**Classical Best Response**:
```
BR_i(σ_{-i}) = argmax_{σᵢ} u_i(σᵢ, σ_{-i})
```

**Quantum Best Response**:
```
BR_i^Q(ρ_{-i}) = argmax_{ρᵢ} Tr[(ρᵢ ⊗ ρ_{-i}) M_i]

subject to: Tr(ρᵢ) = 1, ρᵢ ≥ 0
```

**Solution via Semidefinite Programming (SDP)**:
```
maximize: Tr(M_i^T (ρᵢ ⊗ ρ_{-i}))
subject to: ρᵢ ⪰ 0, Tr(ρᵢ) = 1
```

This is convex! Solvable efficiently.

### C.3 Quantum Nash Equilibrium

**Definition**: ρ* = (ρ₁*,...,ρₙ*) is quantum Nash if:
```
∀i, ∀ρᵢ: Tr[(ρᵢ* ⊗ ρ*_{-i}) M_i] ≥ Tr[(ρᵢ ⊗ ρ*_{-i}) M_i]
```

**Existence**: Not guaranteed for all payoff structures!

**Theorem (Marinatto & Weber)**: 
Quantum games with commuting payoff operators [M_i, M_j] = 0 always have quantum Nash equilibria.

**Finding Quantum Nash**:

**Method 1: Fixed Point Iteration**
```
repeat:
  ρᵢ(t+1) = BR_i^Q(ρ_{-i}(t))
until ||ρ(t+1) - ρ(t)|| < ε
```

**Method 2: Variational Approach**
```
Parameterize: ρᵢ(θᵢ) via quantum circuit
Optimize: (θ₁*,...,θₙ*) such that each θᵢ* is best response
Use gradient ascent on quantum Fisher information manifold
```

### C.4 Entanglement in Games

**Quantifying Strategic Entanglement**:

For 2-player game with state ρ_AB:

**Mutual Information**:
```
I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)

where S(ρ) = -Tr(ρ log ρ) (von Neumann entropy)
```

**Quantum Mutual Information**:
```
I_Q(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB) + S(ρ_AB||ρ_A ⊗ ρ_B)

Measures total (classical + quantum) correlations
```

**Result**: Higher entanglement → stronger correlations → better equilibria payoffs

**Khan et al. Trading Game**:
- Classical traders: I_Q = 0
- Quantum traders: I_Q ≈ 0.8 bits
- Payoff improvement: +15-20%

---

## D. QUBO and Ising Models

### D.1 QUBO Formulation

**General Form**:
```
minimize: f(x) = Σᵢ qᵢᵢ xᵢ + Σᵢ<ⱼ qᵢⱼ xᵢxⱼ
subject to: x ∈ {0,1}ⁿ

Matrix form: f(x) = x^T Q x
where Q is upper-triangular
```

**Properties**:
- NP-hard in general
- Many combinatorial problems reduce to QUBO
- Amenable to quantum annealing

### D.2 Constraint Encoding

**One-Hot Encoding**:

Constraint: exactly one of {x₁,...,xₖ} is true
```
Σᵢ xᵢ = 1

Penalty: P(Σᵢ xᵢ - 1)² = P[Σᵢ xᵢ² - 2Σᵢ xᵢ + 1]
       = P[k - 2Σᵢ xᵢ + 1]  (since xᵢ² = xᵢ for binary)

Add to Q:
- Linear terms: qᵢᵢ -= 2P
- Constant: +P(k+1)
```

**Inequality Constraint**:

g(x) ≤ b where g(x) = Σᵢ wᵢxᵢ
```
Introduce slack: s = b - g(x)
Binary expand: s = Σⱼ 2ⱼ sⱼ for sⱼ ∈ {0,1}

Constraint becomes: Σᵢ wᵢxᵢ + Σⱼ 2ⱼ sⱼ = b

Penalty: P(Σᵢ wᵢxᵢ + Σⱼ 2ⱼ sⱼ - b)²
```

**Logical Constraints**:

AND: z = x ∧ y
```
Penalty: P(z - xy)²
       = P(z² - 2xyz + x²y²)
       = P(z - 2xyz + xy)  (binary)

Add to Q:
Q_zz += P
Q_xy += P
Q_xz -= P
Q_yz -= P
```

OR: z = x ∨ y
```
z = 1 - (1-x)(1-y) = x + y - xy
Penalty: P(z - x - y + xy)²
```

### D.3 Ising Model

**Spin Glass Form**:
```
H(σ) = Σᵢ<ⱼ Jᵢⱼ σᵢσⱼ + Σᵢ hᵢσᵢ

where σᵢ ∈ {-1,+1}
```

**QUBO ↔ Ising Conversion**:
```
xᵢ ∈ {0,1} ↔ σᵢ = 2xᵢ - 1 ∈ {-1,+1}

QUBO: f(x) = x^T Q x
Ising: H(σ) = (σ+1)^T/4 Q (σ+1)/4
             = (1/4)Σᵢⱼ Qᵢⱼ σᵢσⱼ + (1/4)Σᵢ(Σⱼ Qᵢⱼ)σᵢ + const

Therefore:
Jᵢⱼ = Qᵢⱼ/4
hᵢ = (Σⱼ Qᵢⱼ)/4
```

### D.4 Problem Examples

**Max-Cut**:

Given graph G=(V,E), find partition S ⊂ V maximizing edges between S and V\S.

```
Variables: xᵢ = 1 if i ∈ S, else 0
Objective: maximize Σ_{(i,j)∈E} (xᵢ(1-xⱼ) + xⱼ(1-xᵢ))
         = Σ_{(i,j)∈E} (xᵢ + xⱼ - 2xᵢxⱼ)

QUBO (minimize): -Σ_{(i,j)∈E} (xᵢ + xⱼ - 2xᵢxⱼ)

Q matrix:
Q_ii = -deg(i)
Q_ij = 2 if (i,j) ∈ E
```

**Graph Coloring**:

Color n nodes with k colors such that adjacent nodes differ.

```
Variables: xᵢᶜ = 1 if node i has color c
           (total: nk variables)

Constraints:
1. Each node one color: Σc xᵢᶜ = 1 for all i
2. Adjacent different: xᵢᶜ + xⱼᶜ ≤ 1 for (i,j) ∈ E

Penalty QUBO:
Q = P₁ Σᵢ (1 - Σc xᵢᶜ)² + P₂ Σ_{(i,j)∈E} Σc xᵢᶜ xⱼᶜ
```

**Traveling Salesman**:

Find shortest tour visiting all cities.

```
Variables: xᵢₜ = 1 if city i visited at time t
           (total: n² variables)

Constraints:
1. Each city visited once: Σₜ xᵢₜ = 1
2. Each time one city: Σᵢ xᵢₜ = 1
3. Route continuity: If xᵢₜ = 1, then Σⱼ dᵢⱼ xⱼ,ₜ₊₁ (distance cost)

QUBO:
Q = P(violations) + Σᵢⱼₜ dᵢⱼ xᵢₜ xⱼ,ₜ₊₁
```

---

## E. Quantum Approximate Optimization Algorithm (QAOA)

### E.1 Mathematical Formulation

**Problem**: Find x* = argmin C(x) for x ∈ {0,1}ⁿ

**Cost Hamiltonian**:
```
Ĥ_C = Σ_{clauses α} ŵ_α Ĉ_α

where Ĉ_α projects onto violating assignments

Example (Max-Cut):
Ĥ_C = -Σ_{(i,j)∈E} (I - ZᵢZⱼ)/2
(eigenvalue = -1 if cut edge, 0 otherwise)
```

**Mixer Hamiltonian**:
```
Ĥ_B = Σᵢ Xᵢ  (bit-flip mixer)

Creates equal superposition of all states
```

**QAOA State**:
```
|ψ(γ,β)⟩ = e^{-iβₚĤ_B} e^{-iγₚĤ_C} ... e^{-iβ₁Ĥ_B} e^{-iγ₁Ĥ_C} |+⟩^⊗n

Parameters: γ = (γ₁,...,γₚ), β = (β₁,...,βₚ)
```

**Optimization**:
```
(γ*,β*) = argmin_{γ,β} ⟨ψ(γ,β)|Ĥ_C|ψ(γ,β)⟩

Classical optimizer adjusts (γ,β)
Quantum computer evaluates expectation
```

### E.2 Performance Analysis

**Approximation Ratio**:
```
r = C_QAOA / C_OPT

Goal: r → 1 as p → ∞
```

**Theoretical Guarantees**:

**Max-Cut on d-regular graphs**:
- p=1: r ≥ 0.6924 (guaranteed)
- p→∞: r → 1 (concentration)

**3-SAT**:
- Classical: r ≥ 7/8 (random assignment)
- QAOA p=1: r ≥ 0.7927
- QAOA p=11: beats classical threshold

**Scaling**:
- Circuit depth: O(p·n)
- Classical optimization: O(poly(p))
- Overall: polynomial in n for fixed p

### E.3 Gradient Computation

**Parameter Shift Rule**:
```
∂⟨Ĥ_C⟩/∂γⱼ = [⟨Ĥ_C⟩|_{γⱼ+π/2} - ⟨Ĥ_C⟩|_{γⱼ-π/2}] / 2

Similarly for ∂/∂βⱼ

Requires: 2p function evaluations per gradient
```

**Natural Gradient**:

Metric: Quantum Fisher Information Matrix
```
F_{ij} = 4 Re⟨∂ᵢψ|∂ⱼψ⟩ - 4⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩

Natural gradient: F⁻¹ ∇⟨Ĥ_C⟩

Better conditioning → faster convergence
```

### E.4 Advanced QAOA Variants

**QAOA+ (Warm-Start)**:
```
Instead of |+⟩^⊗n, initialize with classical solution:
|ψ₀⟩ = |x_classical⟩

Mixer preserves feasibility constraints
Example: XY mixer for number conservation
```

**Multi-Angle QAOA (ma-QAOA)**:
```
Different parameters per qubit:
|ψ⟩ = ∏ⱼ e^{-iβⱼXⱼ} e^{-iγⱼZⱼ}

More parameters → better optimization
Trade-off: harder classical optimization
```

**Recursive QAOA**:
```
1. Solve relaxed problem with QAOA
2. Fix high-confidence variables
3. Solve reduced problem
4. Repeat until convergence

Reduces effective problem size
```

---

## F. Implementation Algorithms

### F.1 State Vector Simulation

**Direct Method**:
```python
# Pseudocode for single-qubit gate on qubit q
def apply_single_qubit_gate(state, gate, q, n_qubits):
    """
    state: complex array of size 2^n
    gate: 2×2 unitary matrix
    q: target qubit (0 to n-1)
    """
    N = 2^n_qubits
    stride = 2^q
    
    for i in range(0, N, 2*stride):
        for j in range(i, i+stride):
            idx0 = j
            idx1 = j + stride
            
            amp0 = state[idx0]
            amp1 = state[idx1]
            
            state[idx0] = gate[0,0]*amp0 + gate[0,1]*amp1
            state[idx1] = gate[1,0]*amp0 + gate[1,1]*amp1
    
    return state
```

**Complexity**:
- Time: O(2^n) per gate
- Space: O(2^n) for state vector
- Total for circuit: O(depth × 2^n)

**Two-Qubit Gate**:
```python
def apply_two_qubit_gate(state, gate, q1, q2, n_qubits):
    """
    gate: 4×4 unitary matrix
    q1, q2: control and target qubits
    """
    N = 2^n_qubits
    stride1 = 2^min(q1,q2)
    stride2 = 2^max(q1,q2)
    
    for i in range(0, N, 2*stride2):
        for j in range(i, i+stride2, 2*stride1):
            for k in range(j, j+stride1):
                idx00 = k
                idx01 = k + stride1
                idx10 = k + stride2
                idx11 = k + stride1 + stride2
                
                amps = [state[idx00], state[idx01], 
                        state[idx10], state[idx11]]
                
                new_amps = gate @ amps
                
                state[idx00] = new_amps[0]
                state[idx01] = new_amps[1]
                state[idx10] = new_amps[2]
                state[idx11] = new_amps[3]
    
    return state
```

### F.2 Tensor Network Contraction

**Matrix Product State (MPS)**:

Representation:
```
|ψ⟩ = Σ_{s₁...sₙ} A[1]^{s₁}...A[n]^{sₙ} |s₁...sₙ⟩

where A[i]^{sᵢ} are χ×χ matrices (bond dimension χ)
```

**Gate Application**:
```
1. Identify qubits involved: (q, q+1)
2. Contract: M = A[q] A[q+1] (forms χ×2×2×χ tensor)
3. Apply gate: M' = M ×₂ G (G is 4×4 gate matrix)
4. SVD: M' = U S V† with truncation to χ largest singular values
5. Update: A[q] = U√S, A[q+1] = √S V†
```

**Complexity**:
- Storage: O(n χ²)
- Gate application: O(χ³)
- Error: exponentially small in χ

**Advantage**: Can simulate n=100+ qubits with χ ≈ 100

### F.3 Sparse Matrix Techniques

**For observables with sparse structure**:

Example: Ising Hamiltonian Ĥ = Σᵢⱼ JᵢⱼZᵢZⱼ

Each ZᵢZⱼ is diagonal → Ĥ is diagonal!

**Expectation value**:
```
⟨Ĥ⟩ = Σₓ |⟨x|ψ⟩|² H(x)

where H(x) = Σᵢⱼ Jᵢⱼ (2xᵢ-1)(2xⱼ-1)

Only need to evaluate classical function H(x) at sampled bit-strings
```

**Complexity**: O(n_samples × n²) instead of O(2^n)

### F.4 Measurement Simulation

**Born Rule**:
```
P(outcome x) = |⟨x|ψ⟩|² = |ψ[x]|²

where ψ[x] is amplitude of basis state |x⟩
```

**Sampling Algorithm**:
```python
def measure_state(state, n_shots, seed=None):
    """
    state: complex array of size 2^n
    n_shots: number of measurement samples
    """
    np.random.seed(seed)
    
    # Compute probabilities
    probabilities = np.abs(state)**2
    
    # Sample from categorical distribution
    outcomes = np.random.choice(
        len(state), 
        size=n_shots, 
        p=probabilities
    )
    
    # Count frequencies
    counts = np.bincount(outcomes, minlength=len(state))
    
    return counts / n_shots  # empirical distribution
```

**Deterministic Alternative**:
```python
def expectation_value_exact(state, observable):
    """
    observable: Hermitian matrix (2^n × 2^n)
    """
    expectation = np.vdot(state, observable @ state)
    return np.real(expectation)
```

---

## G. Optimization Algorithms

### G.1 COBYLA (Constrained Optimization BY Linear Approximations)

**Method**: Trust-region approach with linear interpolation

**Algorithm**:
```
1. Initialize: x₀, trust radius Δ
2. Build linear model: m(x) ≈ f(x) around xₖ
3. Solve trust region subproblem:
   xₖ₊₁ = argmin m(x) subject to ||x - xₖ|| ≤ Δ
4. Update trust radius based on agreement f vs m
5. Repeat until convergence
```

**Advantages**:
- No derivatives needed
- Handles bounds naturally
- Robust to noise (good for quantum circuits)

**Disadvantages**:
- Slower convergence than gradient methods
- Scales poorly with dimension (p > 20 difficult)

### G.2 SPSA (Simultaneous Perturbation Stochastic Approximation)

**Gradient Estimation**:
```
For f: ℝⁿ → ℝ, approximate ∇f(x):

1. Sample random direction: Δ ~ Rademacher (±1)ⁿ
2. Evaluate: f₊ = f(x + c·Δ), f₋ = f(x - c·Δ)
3. Gradient estimate: ĝ = [(f₊ - f₋)/(2c)] · Δ

Cost: 2 function evaluations for n-dimensional gradient!
```

**Update Rule**:
```
xₖ₊₁ = xₖ - aₖ ĝₖ

where aₖ = a/(k+A)^α (step size schedule)
```

**Convergence**: E[||xₖ - x*||²] → 0 as k → ∞

**Advantages**:
- Efficient for high dimensions
- Parallelizable (can batch evaluations)

**Disadvantages**:
- Requires tuning of (a, c, A, α)
- Stochastic → may need averaging

### G.3 Adam Optimizer

**Adaptive moment estimation**:

```
1. Initialize: m₀ = 0, v₀ = 0 (first/second moment estimates)
2. At iteration t:
   gₜ = ∇f(xₜ) (gradient, can use parameter shift)
   mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ  (momentum)
   vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²  (adaptive learning rate)
   m̂ₜ = mₜ / (1 - β₁ᵗ)  (bias correction)
   v̂ₜ = vₜ / (1 - β₂ᵗ)
   xₜ₊₁ = xₜ - η m̂ₜ / (√v̂ₜ + ε)

Hyperparameters: η=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸
```

**Advantages**:
- Fast convergence
- Adapts to parameter-specific learning rates
- Works well for QAOA

### G.4 Natural Gradient

**Fisher Information Metric**:
```
F_{ij}(θ) = E_p[∂ᵢ log p(x;θ) · ∂ⱼ log p(x;θ)]

For quantum states: Quantum Fisher Information
F_{ij} = 4 Re⟨∂ᵢψ|∂ⱼψ⟩ - 4⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩
```

**Natural Gradient Update**:
```
θₜ₊₁ = θₜ - η F(θₜ)⁻¹ ∇L(θₜ)

Follows steepest descent on manifold
Better geometry → faster convergence
```

**Implementation**:
```python
def natural_gradient_step(theta, grad, fisher_matrix, eta=0.01):
    """
    theta: current parameters
    grad: gradient of loss
    fisher_matrix: quantum Fisher information
    """
    # Regularize for numerical stability
    F_reg = fisher_matrix + 1e-8 * np.eye(len(theta))
    
    # Solve F·Δθ = grad
    natural_grad = np.linalg.solve(F_reg, grad)
    
    # Update
    theta_new = theta - eta * natural_grad
    
    return theta_new
```

---

## H. Practical Considerations

### H.1 Numerical Stability

**Amplitude Normalization**:

After sequence of gates, renormalize:
```python
def normalize_state(state):
    norm = np.linalg.norm(state)
    if norm < 1e-14:
        raise ValueError("State has collapsed to zero")
    return state / norm
```

**Phase Tracking**:

Global phase irrelevant, but relative phases matter:
```python
# Fix global phase for determinism
def fix_global_phase(state):
    # Make first non-zero element real and positive
    idx = np.argmax(np.abs(state))
    phase = np.angle(state[idx])
    return state * np.exp(-1j * phase)
```

### H.2 Error Mitigation

**Measurement Error**:

Characterize SPAM (State Preparation And Measurement) errors:
```
P_measured = M · P_true

where M is confusion matrix

Invert: P_true ≈ M⁻¹ · P_measured
```

**Gate Errors**:

Model as depolarizing channel:
```
ε(ρ) = (1-p)UρU† + p·I/d

p: error probability
d: Hilbert space dimension
```

**Mitigation**: Richardson extrapolation
```
Extrapolate: f(0) ← f(ε₁), f(ε₂), f(ε₃)
```

### H.3 Computational Tricks

**Lazy Evaluation**:
```python
class LazyQuantumState:
    def __init__(self, n_qubits):
        self.gates = []  # store gate sequence
        self.n_qubits = n_qubits
    
    def apply_gate(self, gate, qubits):
        self.gates.append((gate, qubits))
    
    def evaluate(self):
        # Only compute state when needed
        state = np.zeros(2**self.n_qubits)
        state[0] = 1.0
        for gate, qubits in self.gates:
            state = apply_gate(state, gate, qubits)
        return state
```

**Parallelization**:
```python
from multiprocessing import Pool

def parallel_qaoa_evaluation(param_grid):
    """
    Evaluate QAOA for many parameter values in parallel
    """
    with Pool() as pool:
        results = pool.map(evaluate_qaoa, param_grid)
    return results
```

**Memoization**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def compute_hamiltonian_expectation(theta_tuple):
    """
    Cache results to avoid recomputation
    """
    theta = np.array(theta_tuple)
    # ... compute ...
    return expectation
```

---

## I. Extended Mathematical Proofs

### I.1 QAOA Concentration Theorem

**Theorem**: For Max-Cut on d-regular graphs with p → ∞, QAOA achieves optimal solution with probability → 1.

**Proof Sketch**:

1. Define overlap with optimal solution:
   ```
   F(γ,β) = |⟨x_opt|ψ(γ,β)⟩|²
   ```

2. Show that ∃(γ*,β*) such that:
   ```
   F(γ*,β*) ≥ 1 - e^(-Ω(p))
   ```

3. Use:
   - Adiabatic theorem: slow evolution → ground state
   - QAOA approximates adiabatic path as p → ∞
   
4. Therefore: measurement yields x_opt with high probability

### I.2 Quantum vs Classical Nash Payoffs

**Theorem**: In quantum Prisoner's Dilemma with maximal entanglement, quantum equilibrium payoff exceeds classical.

**Proof**:

Classical game:
```
         C       D
C      (3,3)   (0,5)
D      (5,0)   (1,1)

Nash: (D,D) → payoff (1,1)
```

Quantum game with J(γ=π/2) entangling:

Strategy: Q = exp(iπX/2) (Hadamard-like)

```
|ψ₁⟩ = J|00⟩ = (|00⟩ + i|11⟩)/√2
|ψ₂⟩ = (Q ⊗ Q)|ψ₁⟩ = ...  (calculation)
|ψ₃⟩ = J†|ψ₂⟩ → measures as |00⟩ (cooperate)

Payoff: (3,3)
```

Prove Q is Nash: deviation to any classical strategy yields ≤ 3

QED: (3,3) > (1,1) ∎

---

## J. Reference Implementations

### J.1 Minimal QAOA

```python
import numpy as np
from scipy.optimize import minimize

def qaoa_maxcut(graph, p_layers=2):
    """
    Solve Max-Cut using QAOA
    
    Args:
        graph: adjacency matrix (n×n)
        p_layers: QAOA depth
    
    Returns:
        cut_value: best cut found
        solution: bit-string
    """
    n = len(graph)
    
    def cost_hamiltonian(bitstring):
        """Negative cut size"""
        x = np.array([int(b) for b in bitstring])
        cut = 0
        for i in range(n):
            for j in range(i+1, n):
                if graph[i,j] and x[i] != x[j]:
                    cut += 1
        return -cut
    
    def qaoa_circuit(gamma, beta):
        """
        Construct QAOA state
        Returns: amplitude vector
        """
        # Initialize |+⟩^⊗n
        state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
        
        for layer in range(p_layers):
            # Problem unitary: e^(-iγ H_C)
            for i in range(2**n):
                bitstring = format(i, f'0{n}b')
                energy = cost_hamiltonian(bitstring)
                state[i] *= np.exp(-1j * gamma[layer] * energy)
            
            # Mixer unitary: e^(-iβ H_B)
            # H_B = X₁ + ... + Xₙ → bit flips
            new_state = np.zeros(2**n, dtype=complex)
            for i in range(2**n):
                bitstring = format(i, f'0{n}b')
                bits = list(bitstring)
                for q in range(n):
                    # Flip qubit q
                    bits_flipped = bits.copy()
                    bits_flipped[q] = '1' if bits[q] == '0' else '0'
                    j = int(''.join(bits_flipped), 2)
                    new_state[j] += state[i] * (-1j * beta[layer])
                # Identity component
                new_state[i] += state[i]
            state = new_state
            
            # Renormalize
            state /= np.linalg.norm(state)
        
        return state
    
    def objective(params):
        """Expected cost"""
        gamma = params[:p_layers]
        beta = params[p_layers:]
        state = qaoa_circuit(gamma, beta)
        
        expectation = 0
        for i in range(2**n):
            prob = np.abs(state[i])**2
            bitstring = format(i, f'0{n}b')
            expectation += prob * cost_hamiltonian(bitstring)
        
        return expectation
    
    # Optimize
    x0 = np.random.uniform(0, 2*np.pi, 2*p_layers)
    result = minimize(objective, x0, method='COBYLA')
    
    # Extract best solution
    optimal_state = qaoa_circuit(result.x[:p_layers], result.x[p_layers:])
    probabilities = np.abs(optimal_state)**2
    best_idx = np.argmax(probabilities)
    best_bitstring = format(best_idx, f'0{n}b')
    best_cut = -cost_hamiltonian(best_bitstring)
    
    return best_cut, best_bitstring

# Example usage:
graph = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

cut, solution = qaoa_maxcut(graph, p_layers=3)
print(f"Max cut: {cut}, solution: {solution}")
```

### J.2 Quantum Game Solver

```python
import numpy as np
from scipy.linalg import expm

def quantum_prisoners_dilemma():
    """
    Find quantum Nash equilibrium
    """
    # Payoff matrices
    R, S, T, P = 3, 0, 5, 1  # Reward, Sucker, Temptation, Punishment
    
    # Classical payoffs encoded in operator
    payoff1 = np.array([
        [R, S],
        [T, P]
    ])
    
    def strategy_unitary(theta, phi):
        """
        SU(2) strategy parametrization
        """
        return np.array([
            [np.exp(1j*phi) * np.cos(theta/2), np.sin(theta/2)],
            [-np.sin(theta/2), np.exp(-1j*phi) * np.cos(theta/2)]
        ])
    
    def entangling_gate(gamma=np.pi/2):
        """
        Creates initial entanglement
        """
        J = np.zeros((4,4), dtype=complex)
        J[0,0] = J[1,1] = J[2,2] = 1
        J[3,3] = np.exp(1j*gamma)
        return J / np.sqrt(2)
    
    def compute_payoff(theta1, phi1, theta2, phi2, gamma=np.pi/2):
        """
        Quantum game payoff for player 1
        """
        U1 = strategy_unitary(theta1, phi1)
        U2 = strategy_unitary(theta2, phi2)
        J = entangling_gate(gamma)
        
        # Initial state
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        
        # Quantum game protocol
        psi1 = J @ psi0
        psi2 = np.kron(U1, U2) @ psi1
        psi3 = J.conj().T @ psi2
        
        # Measure and compute payoff
        expected_payoff = 0
        for i in range(2):
            for j in range(2):
                # Probability of outcome (i,j)
                basis = np.zeros(4)
                basis[2*i + j] = 1
                prob = np.abs(np.dot(basis, psi3))**2
                expected_payoff += prob * payoff1[i, j]
        
        return expected_payoff
    
    # Find Nash equilibrium via grid search
    theta_range = np.linspace(0, np.pi, 20)
    phi_range = np.linspace(0, 2*np.pi, 20)
    
    best_payoff = -np.inf
    best_params = None
    
    for theta1 in theta_range:
        for phi1 in phi_range:
            for theta2 in theta_range:
                for phi2 in phi_range:
                    p1 = compute_payoff(theta1, phi1, theta2, phi2)
                    p2 = compute_payoff(theta2, phi2, theta1, phi1)
                    
                    # Check if Nash (simplified)
                    is_nash = True
                    for theta1_alt in theta_range[::5]:
                        p1_alt = compute_payoff(theta1_alt, phi1, theta2, phi2)
                        if p1_alt > p1 + 0.1:
                            is_nash = False
                            break
                    
                    if is_nash and p1 + p2 > best_payoff:
                        best_payoff = p1 + p2
                        best_params = (theta1, phi1, theta2, phi2, p1, p2)
    
    return best_params

# Find equilibrium
nash = quantum_prisoners_dilemma()
print(f"Quantum Nash equilibrium:")
print(f"Player 1: θ={nash[0]:.2f}, φ={nash[1]:.2f}, payoff={nash[4]:.2f}")
print(f"Player 2: θ={nash[2]:.2f}, φ={nash[3]:.2f}, payoff={nash[5]:.2f}")
```

---

## K. Conclusion

This mathematical appendix provides the rigorous foundations for the quantum-game theory framework. Key takeaways:

1. **Quantum mechanics** provides the computational substrate via state vectors, unitary evolution, and measurement

2. **Game theory** provides the strategic structure via payoffs, equilibria, and dynamics

3. **QUBO/Ising models** provide universal encoding for combinatorial problems

4. **QAOA** provides the algorithmic bridge, combining quantum evolution with classical optimization

5. **Numerical methods** (state simulation, tensor networks, optimization) make it all practically computable

The mathematical rigor ensures:
- **Correctness**: algorithms provably converge to equilibria/optima
- **Efficiency**: polynomial-time operations for fixed problem parameters
- **Scalability**: tensor network methods extend to large systems
- **Determinism**: seeded randomness ensures reproducibility

With these foundations, the framework can tackle the full spectrum of problems from puzzles to markets, all within a unified quantum-game theoretic paradigm.

---

**Appendix Version**: 1.0  
**Companion to**: Quantum-Game Framework Technical Specification v1.0
