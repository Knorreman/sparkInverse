package sparkinverse.syntax

import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import sparkinverse.api.{IterativeInverseConfig, MatrixInversion, RecursiveInverseConfig}

object block {
  implicit class BlockMatrixSyntax(private val matrix: BlockMatrix) extends AnyVal {
    def inverse(): BlockMatrix = MatrixInversion.block(matrix).inverse()
    def inverse(config: RecursiveInverseConfig): BlockMatrix = MatrixInversion.block(matrix).inverse(config)
    def iterativeInverse(): BlockMatrix = MatrixInversion.block(matrix).iterativeInverse()
    def iterativeInverse(config: IterativeInverseConfig): BlockMatrix = MatrixInversion.block(matrix).iterativeInverse(config)
    def localInverse(): BlockMatrix = MatrixInversion.block(matrix).localInverse()
    def svdInverse(): BlockMatrix = MatrixInversion.block(matrix).svdInverse()
    def leftPseudoInverse(): BlockMatrix = MatrixInversion.block(matrix).leftPseudoInverse()
    def leftPseudoInverse(config: RecursiveInverseConfig): BlockMatrix = MatrixInversion.block(matrix).leftPseudoInverse(config)
    def rightPseudoInverse(): BlockMatrix = MatrixInversion.block(matrix).rightPseudoInverse()
    def rightPseudoInverse(config: RecursiveInverseConfig): BlockMatrix = MatrixInversion.block(matrix).rightPseudoInverse(config)
    def normOne(): Double = MatrixInversion.block(matrix).normOne()
    def normInf(): Double = MatrixInversion.block(matrix).normInf()
    def frobeniusNormSquared(): Double = MatrixInversion.block(matrix).frobeniusNormSquared()
    def scalarMultiply(scalar: Double): BlockMatrix = MatrixInversion.block(matrix).scalarMultiply(scalar)
    def negate(): BlockMatrix = MatrixInversion.block(matrix).negate()
  }
}
