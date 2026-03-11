package sparkinverse.api

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix}
import sparkinverse.block.BlockMatrixOps
import sparkinverse.coordinate.CoordinateMatrixOps

object MatrixInversion {
  def block(matrix: BlockMatrix): BlockMatrixOps = new BlockMatrixOps(matrix)

  def coordinate(matrix: CoordinateMatrix): CoordinateMatrixOps = new CoordinateMatrixOps(matrix)
}
