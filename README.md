# sparkInverse

A distributed matrix inversion library for Apache Spark, implementing block-wise matrix inversion using the Schur complement method.

## Overview

`sparkInverse` provides efficient algorithms for inverting large matrices stored as Apache Spark's `BlockMatrix` or `CoordinateMatrix`. It uses recursive block-wise inversion based on the [block matrix inversion formula](https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion):

```
| E  F |⁻¹   | E⁻¹ + E⁻¹FS⁻¹GE⁻¹    -E⁻¹FS⁻¹ |
| G  H |   = |    -S⁻¹GE⁻¹              S⁻¹  |

where S = H - GE⁻¹F (Schur complement)
```

The algorithm recursively inverts submatrices until they are small enough to handle locally using Breeze's dense inverse.

## Installation

Clone this repository and add it as a dependency in your `build.sbt`:

```scala
lazy val sparkInverse = project
  .in(file("path/to/sparkInverse"))
  .settings(
    name := "sparkInverse"
  )

lazy val yourProject = project
  .in(file("."))
  .dependsOn(sparkInverse)
```

## Usage

### BlockMatrix Inverse

```scala
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.linalg.DenseMatrix
import Inverse.BlockMatrixInverse

// Create a 4x4 BlockMatrix from 2x2 blocks
val blocks = Seq(
  ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
  ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
  ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
  ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
)
val matrix = new BlockMatrix(sc.parallelize(blocks, 4), 2, 2)

// Invert using recursive Schur complement (default)
val inverted = matrix.inverse()

// Or with custom parameters
val inverted2 = matrix.inverse(limit = 4096, numMidDimSplits = 4, useCheckpoints = true)

// SVD-based inverse (for smaller matrices)
val svdInverted = matrix.svdInv()

// Local inverse (collects to driver)
val localInverted = matrix.localInv()
```

### Iterative Inverse (Newton-Schulz)

For diagonally dominant or well-conditioned matrices, the iterative Newton-Schulz method can be faster:

```scala
// Iterative inverse with convergence checking
val iterativeInv = matrix.iterativeInverse(
  maxIter = 30,
  tolerance = 1e-10,
  useCheckpoints = true,
  checkpointInterval = 5
)

// Uses X_{k+1} = X_k * (2I - A * X_k)
// Converges quadratically for diagonally dominant matrices
```

### Pseudo-Inverse for Non-Square Matrices

```scala
import Inverse.BlockMatrixInverse

// Left pseudo-inverse: (A^T * A)⁻¹ * A^T
// For matrices with more rows than columns (tall matrices)
val tallMatrix: BlockMatrix = ...
val leftPinv = tallMatrix.leftPseudoInverse()

// Right pseudo-inverse: A^T * (A * A^T)⁻¹
// For matrices with more columns than rows (wide matrices)
val wideMatrix: BlockMatrix = ...
val rightPinv = wideMatrix.rightPseudoInverse()
```

### CoordinateMatrix Support

All inverse methods are also available for `CoordinateMatrix`:

```scala
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.MatrixEntry

val entries = sc.parallelize(Seq(
  MatrixEntry(0, 0, 1.0), MatrixEntry(0, 1, 2.0),
  MatrixEntry(1, 0, 3.0), MatrixEntry(1, 1, 4.0)
))
val coordMatrix = new CoordinateMatrix(entries, 2, 2)

val inverted = coordMatrix.inverse()
val iterativeInv = coordMatrix.iterativeInverse()
```

## Configuration

### Recursive Inverse Parameters

- `limit`: Maximum dimension for submatrices to invert locally (default: 4096)
- `numMidDimSplits`: Parallelism factor for matrix multiplication (default: 1)
- `useCheckpoints`: Whether to use Spark checkpointing to manage lineage (default: true)

### Iterative Inverse Parameters

- `maxIter`: Maximum iterations (default: 30)
- `tolerance`: Convergence tolerance on ||I - A*X||_F / n (default: 1e-10)
- `checkpointInterval`: How often to checkpoint (default: 5)

### Performance Notes

1. **Checkpoints**: Enable checkpointing for large matrices to avoid lineage explosion
2. **Block size**: Choose block sizes that fit in memory (typically 1024-4096)
3. **Parallelism**: Increase `numMidDimSplits` for better parallelism on large clusters
4. **Iterative vs Recursive**: Use iterative method for diagonally dominant matrices; recursive for general matrices

## Testing

```bash
sbt test
```

## Requirements

- Scala 2.13+
- Apache Spark 4.1.1+
- Breeze (included via Spark)

## License

MIT License - see LICENSE file
