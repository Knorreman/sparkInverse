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
import sparkinverse.api.IterativeInverseConfig

val inverse = blockMatrix.iterativeInverse(
  IterativeInverseConfig(
    order = 2,
    maxIter = 30,
    tolerance = 1e-10,
    checkpointEvery = 5
  )
)

val cubicInverse = blockMatrix.iterativeInverse(
  IterativeInverseConfig(
    order = 3,
    maxIter = 20,
    tolerance = 1e-10,
    checkpointEvery = 5
  )
)
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
- `normOne`
- `normInf`
- `frobeniusNormSquared`
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

- `order`
- `maxIter`
- `tolerance`
- `useCheckpoints`
- `checkpointEvery`
- `midSplits`
- `persistLevel`

## Choosing An Algorithm

- Use recursive inversion as the default general-purpose algorithm.
- Use iterative inversion when the matrix is well-conditioned enough for Newton-Schulz to converge quickly.
- Use `iterativeInverse(IterativeInverseConfig(order = 3, ...))` for cubic hyperpower when you want fewer iterations at the cost of more multiplies per step.
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
