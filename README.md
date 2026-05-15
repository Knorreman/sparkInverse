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
import sparkinverse.api.{IterativeInverseConfig, AlphaStrategy}

// Default: Frobenius alpha + order-2 Newton-Schulz
val inverse = blockMatrix.iterativeInverse()

// Third-order Newton-Schulz with conservative adaptive alpha
val cubicInverse = blockMatrix.iterativeInverse(
  IterativeInverseConfig(
    order = 3,
    maxIter = 20,
    tolerance = 1e-10,
    checkpointEvery = 5,
    alphaStrategy = AlphaStrategy.Adaptive
  )
)
```

### Alpha Scaling Strategies

The initial approximation X₀ = α·Aᵀ needs α ≤ 1/σ₁² to converge. The choice affects both convergence rate and Spark cost:

```scala
// Original: α = 1/(‖A‖₁ · ‖A‖_∞)
// Cost: 2 shuffles + 2 actions (normOne + normInf)
AlphaStrategy.NormProduct

// New default: α = 1/‖A‖²_F — safe, simple, no shuffle
AlphaStrategy.Frobenius

// Best for ill-conditioned matrices: α = 1/σ₁² via power iteration
// Cost: 2·N distributed matrix multiplies
AlphaStrategy.PowerIteration(powerIterations = 3)

// Experimental: start with Frobenius, conservatively shrink α from first residual
AlphaStrategy.Adaptive
```

`Frobenius` is the recommended default for distributed execution because it is safe and avoids the row/column norm shuffles used by `NormProduct`. It is not universally tighter than `NormProduct`, but is often competitive at lower Spark cost. On ill-conditioned matrices (κ > 1000), `PowerIteration` can be worth its extra distributed multiplies because it directly estimates σ₁².

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
- `normOne` — ‖A‖₁ via column sums (1 shuffle)
- `normInf` — ‖A‖\_∞ via row sums (1 shuffle)
- `frobeniusNormSquared` — ‖A‖²_F, zero shuffle
- `scalarMultiply`
- `negate`

Additional distributed arithmetic helpers for `CoordinateMatrix`:

> Note: Coordinate inversion paths still route through `BlockMatrix`, but conversion now uses an adaptive block-size heuristic based on coordinate density to reduce dense-block memory blowups on sparse inputs.

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
- `regularizationLambda`: Tikhonov regularization parameter for pseudo-inverse (default `0.0`)

### IterativeInverseConfig

| Field                      | Type            | Default               | Description                                       |
| -------------------------- | --------------- | --------------------- | ------------------------------------------------- |
| `order`                    | `Int`           | `2`                   | Newton-Schulz hyperpower order (2–10)             |
| `maxIter`                  | `Int`           | `30`                  | Maximum iterations                                |
| `tolerance`                | `Double`        | `1e-15`               | Convergence threshold on `‖I - AX‖_F / n`         |
| `useCheckpoints`           | `Boolean`       | `true`                | Enable RDD checkpointing                          |
| `checkpointEvery`          | `Int`           | `5`                   | Checkpoint interval                               |
| `midSplits`                | `Int`           | `1`                   | Multiplication parallelism hint                   |
| `persistLevel`             | `StorageLevel`  | `MEMORY_AND_DISK_SER` | Storage level for intermediates                   |
| `alphaStrategy`            | `AlphaStrategy` | `Frobenius`           | Initial scaling α computation                     |
| `convergenceCheckInterval` | `Int`           | `1`                   | Check `‖I - AX‖_F / n` every N iterations         |
| `adaptiveOrder`            | `Boolean`       | `false`               | Reduce hyperpower order as residual shrinks       |
| `adaptiveOrderFallback`    | `Double`        | `0.3`                 | Switch to order 2 when `‖R‖_F/√n` below threshold |

## Choosing An Algorithm

- Use recursive inversion as the default general-purpose algorithm.
- Use iterative inversion when the matrix is well-conditioned enough for Newton-Schulz to converge quickly.
- Use `iterativeInverse(IterativeInverseConfig(order = 3, ...))` for cubic hyperpower when you want fewer iterations at the cost of more multiplies per step.
- **Use `AlphaStrategy.Frobenius`** (default) for a safe zero-shuffle α computation.
- **Use `AlphaStrategy.PowerIteration`** on ill-conditioned matrices (κ > 1000) where a precise σ₁² estimate is worth extra distributed multiplies.
- **Use `AlphaStrategy.Adaptive`** only experimentally; it conservatively shrinks the Frobenius α after the first residual and is intended for stability, not guaranteed acceleration.
- Use `localInverse` or `svdInverse` only for matrices small enough to collect to the driver.

## Benchmarks

Benchmark apps live in the `bench` module so normal `core/test` stays focused on correctness.

General inversion benchmark:

```bash
sbt bench/run
```

Alpha strategy benchmark:

```bash
sbt "bench/runMain sparkinverse.benchmark.AlphaStrategyBenchmark"
sbt "bench/runMain sparkinverse.benchmark.AlphaStrategyBenchmark --sizes 100,500,1000 --orders 2,3"
```

The benchmark apps force Spark execution before recording timings. Large matrix cases (n ≥ 1000) can take minutes under `local[*]`; use a real Spark cluster for meaningful shuffle/I/O measurements.

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
