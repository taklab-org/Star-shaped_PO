# Computer-assisted proofs of star-shaped periodic solutions of a delay differential equation

This repository contains the Julia code for the computer-assisted proofs in

> S. Motonaga and A. Takayasu, *Star-shaped periodic solutions of a system of delay differential
> equations: Computer-assisted proof.*

We consider the delay differential equation (DDE), written in complex form as

$$
\dot z(t) = -\alpha e^{i\theta}\bigl(1-|z(t)|^2\bigr) z(t-1),
$$

and prove the existence of star-shaped periodic solutions for seven parameter pairs
$(\alpha,\theta)$ with $\theta=\pi/n$. The solutions are sought in the form
$z(t)=w(t)e^{i\mu t}$ with a 2-periodic profile $w$ and the exact rotation frequency
$\mu=\theta-\pi/2$. The profile $w$ has real Fourier coefficients, and the proof validates a zero
of the resulting real system with the Radii Polynomial Theorem, using interval arithmetic.

## Contents

| File | Role |
|---|---|
| `DDE_w_general.jl` | Proof code. The routine `solve_and_prove_real` computes an approximate profile and the bounds $Y$, $Z_0$, $Z_1$ (called `Z2` in the code) and checks the radii polynomial. |
| `audit/2026-09-08/validate_B.jl` | Driver that runs the proof for the seven cases of the paper and checks the bounds in its table. |
| `wproof/Project.toml`, `wproof/Manifest.toml` | Julia environment with the exact package versions. |
| `wproof/runtests.jl` | Consistency tests of the operators and bounds (finite-difference and support checks). These tests are not part of the proof. |

## Requirements

- Julia 1.12 (the results were obtained with Julia 1.12.7). With juliaup, use `julia +1.12` in the
  commands below if the default channel points to another version.
- RadiiPolynomial.jl 0.10.0 and IntervalArithmetic.jl 1.0.9, installed from `wproof/Manifest.toml`

## Reproducing the proofs

Run the following commands from the root of this repository.

1. Install the packages:
   ```bash
   julia --project=wproof -e 'using Pkg; Pkg.instantiate()'
   ```
2. Run the proofs for the seven cases:
   ```bash
   julia audit/2026-09-08/validate_B.jl
   ```
   `DDE_w_general.jl` activates the `wproof` environment by itself. The driver first checks the
   SHA-256 hash of `DDE_w_general.jl`, so the proof code must not be modified. It writes the
   proof result and the coefficients of the approximate profile (decimal and bit patterns) of each
   case to `audit/2026-09-08/certificates/`. Every case ends with the line
   `GUARANTEE_ASSERTIONS n=... all=true`; any failed check stops the script with an error.
   The run takes about 10 seconds on a recent laptop after the packages are compiled.
3. (Optional) Run the consistency tests:
   ```bash
   julia --project=wproof wproof/runtests.jl
   ```

## Results

All computations use $N=30$ Fourier modes, the weight $\nu=1$ and the radius bound $R=10^{-2}$.
The bounds are upper endpoints of the computed intervals, rounded upward, and $r$ is an admissible
radius (Table 2 of the paper).

| $n$ | $\alpha$ | $\mu$ | $Y$ | $Z_0$ | $Z_1$ | $r$ |
|---|---|---|---|---|---|---|
| 3  | 6    | $-\pi/6$    | 4.6e-14 | 0.237 | 2.1e2 | 6.1e-14 |
| 4  | 4    | $-\pi/4$    | 3.1e-14 | 0.158 | 1.1e2 | 3.8e-14 |
| 5  | 4    | $-3\pi/10$  | 3.1e-14 | 0.179 | 7.0e1 | 3.9e-14 |
| 6  | 13/5 | $-\pi/3$    | 2.2e-14 | 0.077 | 8.6e1 | 2.5e-14 |
| 7  | 5/2  | $-5\pi/14$  | 2.2e-14 | 0.075 | 6.4e1 | 2.5e-14 |
| 10 | 5/2  | $-2\pi/5$   | 1.7e-14 | 0.082 | 3.9e1 | 2.0e-14 |
| 35 | 5/2  | $-33\pi/70$ | 1.5e-14 | 0.085 | 2.5e1 | 1.7e-14 |

## Notes

- The parameters $\alpha$ and $\theta/\pi$ are given as rational numbers, and $\pi$ is enclosed
  rigorously. The rotation frequency $\mu=\theta-\pi/2$ is fixed exactly, so that $\beta=i\alpha$.
- `validate_B.jl` replaces one helper (`ΛR`) so that the integer $k$ in $\omega k$ is converted to an
  interval explicitly. This keeps the guarantee flags of IntervalArithmetic.jl valid and does not
  change the computed bounds.

## License

This code is released under the MIT License; see `LICENSE`.
