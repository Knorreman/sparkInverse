package sparkinverse.syntax

import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import sparkinverse.api.{IterativeInverseConfig, PseudoInverseSide, RecursiveInverseConfig}
import sparkinverse.coordinate.CoordinateMatrixOps

object coordinate {
  implicit class CoordinateMatrixSyntax(private val matrix: CoordinateMatrix) extends AnyVal {
    private def ops: CoordinateMatrixOps = new CoordinateMatrixOps(matrix)

    def inverse(): CoordinateMatrix = ops.inverse()
    def inverse(config: RecursiveInverseConfig): CoordinateMatrix = ops.inverse(config)
    def iterativeInverse(config: IterativeInverseConfig = IterativeInverseConfig()): CoordinateMatrix =
      ops.iterativeInverse(config)
    def localInverse(): CoordinateMatrix = ops.localInverse()
    def svdInverse(): CoordinateMatrix = ops.svdInverse()
    def pseudoInverse(side: PseudoInverseSide): CoordinateMatrix = ops.pseudoInverse(side)
    def pseudoInverse(side: PseudoInverseSide, config: RecursiveInverseConfig): CoordinateMatrix = ops.pseudoInverse(side, config)
    def normOne(): Double = ops.normOne()
    def normInf(): Double = ops.normInf()
    def frobeniusNormSquared(): Double = ops.frobeniusNormSquared()
    def scalarMultiply(scalar: Double): CoordinateMatrix = ops.scalarMultiply(scalar)
    def negate(): CoordinateMatrix = ops.negate()
    def multiply(other: CoordinateMatrix): CoordinateMatrix = ops.multiply(other)
    def add(other: CoordinateMatrix): CoordinateMatrix = ops.add(other)
    def subtract(other: CoordinateMatrix): CoordinateMatrix = ops.subtract(other)
    def partitionBy(partitioner: Partitioner): CoordinateMatrix = ops.partitionBy(partitioner)
    def transposeDistributed(): CoordinateMatrix = ops.transpose()
  }
}
