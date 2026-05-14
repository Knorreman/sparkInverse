# sparkInverse

Distributed matrix inversion for Apache Spark, packaged as a reusable Scala library.

## Structure

- `core`: publishable library code and tests
- `bench`: benchmark app for local experiments and performance comparisons

The library API now lives under the `sparkinverse` package. Benchmark code is no longer mixed into the main library artifact.

## Installation

As a local multi-project dependency:

```scala
lazy val sparkInverse = RootProject(file("path/to/sparkInverse"))

lazy val yourProject = project
  .in(file("."))
  .dependsOn(sparkInverse / "core")
```

If you want source dependencies directly, the reusable module is the `core` subproject.

## Quick Start

### Syntax API

```scala
import sparkinverse.api.{IterativeInverseConfig, PseudoInverseSide, RecursiveInverseConfig}
import sparkinverse.syntax.block._
import sparkinverse.syntax.coordinate._

val inverse = blockMatrix.inverse()
val cubicInverse = blockMatrix.iterativeInverse(IterativeInverseConfig(order = 3))
val pseudoInverse = coordinateMatrix.pseudoInverse(PseudoInverseSide.Left)
```

### Configured Inversion

```scala
import sparkinverse.api.RecursiveInverseConfig

val inverse = blockMatrix.inverse(
  RecursiveInverseConfig(
    limit = 4096,
    midSplits = 4,
    useCheckpoints = true,
    targetOutputPartitions = Some(64)
  )
)
```

### Iterative Inversion

```scala
import sparkinverse.api.{IterativeInverseConfig, AlphaStrategy, PolynomialStyle}

// Default: Frobenius alpha + binomial hyperpower, order 2
val inverse = blockMatrix.iterativeInverse()

// Third-order Newton-Schulz with adaptive alpha
val cubicInverse = blockMatrix.iterativeInverse(
  IterativeInverseConfig(
    order = 3,
    maxIter = 20,
    tolerance = 1e-10,
    checkpointEvery = 5,
    alphaStrategy = AlphaStrategy.Adaptive,
    polynomialStyle = PolynomialStyle.CANS
  )
)
```

### Alpha Scaling Strategies

The initial approximation X₀ = α·Aᵀ needs α ≤ 1/σ₁² to converge. The choice affects both convergence rate and Spark cost:

```scala
// Original: α = 1/(‖A‖₁ · ‖A‖_∞)
// Cost: 1 shuffle + 1 action (via fused normOneAndNormInf)
AlphaStrategy.NormProduct

// New default: α = 1/‖A‖²_F — tighter bound, no shuffle
AlphaStrategy.Frobenius

// Best for ill-conditioned matrices: α = 1/σ₁² via power iteration
// Cost: 2·N distributed matrix multiplies
AlphaStrategy.PowerIteration(powerIterations = 3)

// Start with Frobenius, refine from first-iteration residual
AlphaStrategy.Adaptive
```

For well-conditioned matrices (κ < 100), `Frobenius` is the recommended default — it produces an equally tight initial α at zero shuffle cost. On ill-conditioned matrices (κ > 1000), `PowerIteration` can be the difference between convergence and divergence. `Adaptive` gives you the best of both at essentially Frobenius cost.

### Polynomial Styles

```scala
// Classical hyperpower: C = I + R + R² + ... (default)
PolynomialStyle.Binomial

// CANS-style: same coefficients now, infrastructure for future
// Chebyshev-optimal minimax coefficients on order 3
PolynomialStyle.CANS
```

### Migration (Before -> After)

```scala
// Before
MatrixInversion.block(blockMatrix).iterativeInverse(3, IterativeInverseConfig(maxIter = 20))

// After
blockMatrix.iterativeInverse(IterativeInverseConfig(order = 3, maxIter = 20))
```

## Supported Operations

For `BlockMatrix` and `CoordinateMatrix`:

- `inverse`
- `iterativeInverse` (set order in `IterativeInverseConfig.order`)
- `localInverse`
- `svdInverse`
- `pseudoInverse(PseudoInverseSide.Left|Right)`
- `normOne` — ‖A‖₁ via column sums
- `normInf` — ‖A‖\_∞ via row sums
- `normOneAndNormInf` — fused: both norms in 1 shuffle + 1 action
- `frobeniusNormSquared` — ‖A‖²_F, zero shuffle
- `scalarMultiply`
- `negate`

Additional distributed arithmetic helpers for `CoordinateMatrix`:

- `multiply`
- `add`
- `subtract`
- `partitionBy`
- `transpose`

## Configuration

### RecursiveInverseConfig

- `limit`: local inversion threshold
- `midSplits`: Spark matrix-multiply parallelism hint
- `useCheckpoints`: requires `SparkContext#setCheckpointDir`
- `targetOutputPartitions`: optional output coalesce target
- `unionCoalesceThreshold`: no-shuffle coalesce trigger threshold
- `minBlockSizeForPersistence`: skip persistence for tiny intermediates

### IterativeInverseConfig

| Field             | Type              | Default               | Description                               |
| ----------------- | ----------------- | --------------------- | ----------------------------------------- |
| `order`           | `Int`             | `2`                   | Newton-Schulz hyperpower order (2–10)     |
| `maxIter`         | `Int`             | `30`                  | Maximum iterations                        |
| `tolerance`       | `Double`          | `1e-15`               | Convergence threshold on `‖I - AX‖_F / n` |
| `useCheckpoints`  | `Boolean`         | `true`                | Enable RDD checkpointing                  |
| `checkpointEvery` | `Int`             | `5`                   | Checkpoint interval                       |
| `midSplits`       | `Int`             | `1`                   | Multiplication parallelism hint           |
| `persistLevel`    | `StorageLevel`    | `MEMORY_AND_DISK_SER` | Storage level for intermediates           |
| `alphaStrategy`   | `AlphaStrategy`   | `Frobenius`           | Initial scaling α computation             |
| `polynomialStyle` | `PolynomialStyle` | `Binomial`            | Correction polynomial coefficients        |

## Choosing An Algorithm

- Use recursive inversion as the default general-purpose algorithm.
- Use iterative inversion when the matrix is well-conditioned enough for Newton-Schulz to converge quickly.
- Use `iterativeInverse(IterativeInverseConfig(order = 3, ...))` for cubic hyperpower when you want fewer iterations at the cost of more multiplies per step.
- **Use `AlphaStrategy.Frobenius`** (default) for the cheapest α computation — no shuffle, tighter bound than the legacy NormProduct.
- **Use `AlphaStrategy.PowerIteration`** on ill-conditioned matrices (κ > 1000) where a precise α estimate is needed for convergence.
- **Use `AlphaStrategy.Adaptive`** to get near-optimal α at Frobenius cost — refines α after the first iteration using the observed residual.
- Use `localInverse` or `svdInverse` only for matrices small enough to collect to the driver.

## Benchmarks

The benchmark app now lives in `bench`:

```bash
sbt bench/run
```

The benchmark is intentionally simple: it runs a small fixed set of matrix sizes with a hand-picked config for Schur complement, third-order hyperpower iteration, fourth-order hyperpower iteration, and Newton-Schulz, and each timed section forces Spark execution before recording the result.

## Tests

```bash
sbt core/test
```

## Requirements

- Scala 2.13+
- Apache Spark 4.1.1+
- Java 17+

## License

MIT
