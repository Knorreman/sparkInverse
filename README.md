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

### Facade API

```scala
import sparkinverse.api.MatrixInversion

val inverse = MatrixInversion.block(blockMatrix).inverse()
val pseudoInverse = MatrixInversion.coordinate(coordinateMatrix).leftPseudoInverse()
```

### Configured Inversion

```scala
import sparkinverse.api.{MatrixInversion, RecursiveInverseConfig}

val inverse = MatrixInversion.block(blockMatrix).inverse(
  RecursiveInverseConfig(
    limit = 4096,
    numMidDimSplits = 4,
    useCheckpoints = true
  )
)
```

### Iterative Inversion

```scala
import sparkinverse.api.{IterativeInverseConfig, MatrixInversion}

val inverse = MatrixInversion.block(blockMatrix).iterativeInverse(
  IterativeInverseConfig(
    maxIter = 30,
    tolerance = 1e-10,
    checkpointInterval = 5
  )
)
```

### Optional Syntax Imports

If you prefer extension methods:

```scala
import sparkinverse.syntax.block._
import sparkinverse.syntax.coordinate._

val blockInverse = blockMatrix.inverse()
val coordinateInverse = coordinateMatrix.iterativeInverse()
```

## Supported Operations

For `BlockMatrix` and `CoordinateMatrix`:

- `inverse`
- `iterativeInverse`
- `localInverse`
- `svdInverse`
- `leftPseudoInverse`
- `rightPseudoInverse`
- `normOne`
- `normInf`
- `frobeniusNormSquared`
- `scalarMultiply`
- `negate`

Additional distributed arithmetic helpers for `CoordinateMatrix` are available through the facade and syntax layer:

- `multiply`
- `add`
- `subtract`
- `partitionBy`
- `transpose`

## Configuration

### RecursiveInverseConfig

- `limit`: local inversion threshold
- `numMidDimSplits`: Spark matrix-multiply parallelism hint
- `useCheckpoints`: requires `SparkContext#setCheckpointDir`
- `tuning`: advanced execution tuning, tracing, and coalescing controls

### IterativeInverseConfig

- `maxIter`
- `tolerance`
- `useCheckpoints`
- `checkpointInterval`
- `numMidDimSplits`
- `tuning`: advanced persistence, checkpoint, and tracing controls

## Choosing An Algorithm

- Use recursive inversion as the default general-purpose algorithm.
- Use iterative inversion when the matrix is well-conditioned enough for Newton-Schulz to converge quickly.
- Use `localInverse` or `svdInverse` only for matrices small enough to collect to the driver.

## Benchmarks

The benchmark app now lives in `bench`:

```bash
sbt bench/run
```

The benchmark is intentionally simple: it runs a small fixed set of matrix sizes with a hand-picked config for Schur complement and Newton-Schulz, and each timed section forces Spark execution before recording the result.

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
