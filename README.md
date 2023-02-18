# sparkInverse - Apache Spark implementation of matrix inversion 
## Implementation through block inversion https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
The implementation is recursive for the inverse submatrices 
# Usage

```scala
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import Inverse.BlockMatrixInverse

// Invert a square matrix
val matrix: BlockMatrix =
...
val matrix_inverted = matrix.inverse()

// Invert a non squared matrix
val not_square_matrix: BlockMatrix =
...
val matrix_pseudo_inverse = not_square_matrix.leftPseudoInverse()
```

# Notes
Make sure that each submatrix is smaller than your 'limit' argument. It will otherwise cause an empty collection error. 