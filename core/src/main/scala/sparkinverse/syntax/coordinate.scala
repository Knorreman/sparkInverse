package sparkinverse.syntax

import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import sparkinverse.api.{IterativeInverseConfig, MatrixInversion, RecursiveInverseConfig}

object coordinate {
  implicit class CoordinateMatrixSyntax(private val matrix: CoordinateMatrix) extends AnyVal {
    def inverse(): CoordinateMatrix = MatrixInversion.coordinate(matrix).inverse()
    def inverse(config: RecursiveInverseConfig): CoordinateMatrix = MatrixInversion.coordinate(matrix).inverse(config)
    def iterativeInverse(): CoordinateMatrix = MatrixInversion.coordinate(matrix).iterativeInverse()
    def iterativeInverse(config: IterativeInverseConfig): CoordinateMatrix = MatrixInversion.coordinate(matrix).iterativeInverse(config)
    def localInverse(): CoordinateMatrix = MatrixInversion.coordinate(matrix).localInverse()
    def svdInverse(): CoordinateMatrix = MatrixInversion.coordinate(matrix).svdInverse()
    def leftPseudoInverse(): CoordinateMatrix = MatrixInversion.coordinate(matrix).leftPseudoInverse()
    def leftPseudoInverse(config: RecursiveInverseConfig): CoordinateMatrix = MatrixInversion.coordinate(matrix).leftPseudoInverse(config)
    def rightPseudoInverse(): CoordinateMatrix = MatrixInversion.coordinate(matrix).rightPseudoInverse()
    def rightPseudoInverse(config: RecursiveInverseConfig): CoordinateMatrix = MatrixInversion.coordinate(matrix).rightPseudoInverse(config)
    def normOne(): Double = MatrixInversion.coordinate(matrix).normOne()
    def normInf(): Double = MatrixInversion.coordinate(matrix).normInf()
    def frobeniusNormSquared(): Double = MatrixInversion.coordinate(matrix).frobeniusNormSquared()
    def scalarMultiply(scalar: Double): CoordinateMatrix = MatrixInversion.coordinate(matrix).scalarMultiply(scalar)
    def negate(): CoordinateMatrix = MatrixInversion.coordinate(matrix).negate()
    def multiply(other: CoordinateMatrix): CoordinateMatrix = MatrixInversion.coordinate(matrix).multiply(other)
    def add(other: CoordinateMatrix): CoordinateMatrix = MatrixInversion.coordinate(matrix).add(other)
    def subtract(other: CoordinateMatrix): CoordinateMatrix = MatrixInversion.coordinate(matrix).subtract(other)
    def partitionBy(partitioner: Partitioner): CoordinateMatrix = MatrixInversion.coordinate(matrix).partitionBy(partitioner)
    def transposeDistributed(): CoordinateMatrix = MatrixInversion.coordinate(matrix).transpose()
  }
}
