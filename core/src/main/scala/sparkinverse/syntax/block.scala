package sparkinverse.syntax

import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import sparkinverse.api.{IterativeInverseConfig, PseudoInverseSide, RecursiveInverseConfig}
import sparkinverse.block.BlockMatrixOps

object block {
  implicit class BlockMatrixSyntax(private val matrix: BlockMatrix) extends AnyVal {
    private def ops: BlockMatrixOps = new BlockMatrixOps(matrix)

    def inverse(): BlockMatrix = ops.inverse()
    def inverse(config: RecursiveInverseConfig): BlockMatrix = ops.inverse(config)
    def iterativeInverse(config: IterativeInverseConfig = IterativeInverseConfig()): BlockMatrix =
      ops.iterativeInverse(config)
    def localInverse(): BlockMatrix = ops.localInverse()
    def svdInverse(): BlockMatrix = ops.svdInverse()
    def luInverse(): BlockMatrix = ops.luInverse()
    def pseudoInverse(side: PseudoInverseSide): BlockMatrix = ops.pseudoInverse(side)
    def pseudoInverse(side: PseudoInverseSide, config: RecursiveInverseConfig): BlockMatrix = ops.pseudoInverse(side, config)
    def normOne(): Double = ops.normOne()
    def normInf(): Double = ops.normInf()
    def frobeniusNormSquared(): Double = ops.frobeniusNormSquared()
    def scalarMultiply(scalar: Double): BlockMatrix = ops.scalarMultiply(scalar)
    def negate(): BlockMatrix = ops.negate()
  }
}
