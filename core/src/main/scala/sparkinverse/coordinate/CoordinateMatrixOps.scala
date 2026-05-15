package sparkinverse.coordinate

import org.apache.spark.HashPartitioner
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import sparkinverse.api.{IterativeInverseConfig, PseudoInverseSide, RecursiveInverseConfig}
import sparkinverse.block.BlockMatrixOps

object CoordinateMatrixOps {
  private[sparkinverse] def adaptiveBlockSize(minDim: Int, density: Double): Int = {
    val rawSize =
      if (density < 1e-4) 32
      else if (density < 1e-3) 64
      else if (density < 1e-2) 128
      else if (density < 5e-2) 256
      else if (density < 2e-1) 512
      else 1024

    math.max(1, math.min(rawSize, minDim))
  }
}

final class CoordinateMatrixOps private[sparkinverse] (val matrix: CoordinateMatrix) {
  private lazy val defaultBlockSize: Int = {
    val minDim = math.max(1L, math.min(matrix.numRows(), matrix.numCols())).toInt
    val total = matrix.numRows().toDouble * matrix.numCols().toDouble
    val density = if (total <= 0.0) 1.0 else matrix.entries.count().toDouble / total
    CoordinateMatrixOps.adaptiveBlockSize(minDim, density)
  }

  private def toBlock = matrix.toBlockMatrix(defaultBlockSize, defaultBlockSize)

  private[sparkinverse] def selectedBlockSizeForTesting: Int = defaultBlockSize

  private def fromBlock(blockOps: BlockMatrixOps => org.apache.spark.mllib.linalg.distributed.BlockMatrix): CoordinateMatrix =
    blockOps(new BlockMatrixOps(toBlock)).toCoordinateMatrix()

  private def partitionCountFor(otherPartitions: Int): Int =
    math.max(1, math.max(matrix.entries.getNumPartitions, otherPartitions))

  private def estimatedPartitionCountFor(other: CoordinateMatrix): Int = {
    val base = partitionCountFor(other.entries.getNumPartitions)
    val dimFactor = math.max(1, ((matrix.numRows() + matrix.numCols() + other.numCols()) / 4000L).toInt)
    val scaled = base * dimFactor
    math.max(base, math.min(base * 8, scaled))
  }

  private def keyedEntries(entries: org.apache.spark.rdd.RDD[MatrixEntry], scale: Double = 1.0) =
    entries.map { case MatrixEntry(i, j, v) => ((i, j), v * scale) }

  def svdInverse(): CoordinateMatrix = fromBlock(_.svdInverse())

  def localInverse(): CoordinateMatrix = fromBlock(_.localInverse())

  def negate(): CoordinateMatrix = {
    val newEntries = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, -v) }
    new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
  }

  def inverse(): CoordinateMatrix = inverse(RecursiveInverseConfig())

  def inverse(config: RecursiveInverseConfig): CoordinateMatrix = fromBlock(_.inverse(config))

  def normOne(): Double =
    matrix.entries.map { case MatrixEntry(_, j, v) => (j, math.abs(v)) }.reduceByKey(_ + _).values.max()

  def normInf(): Double =
    matrix.entries.map { case MatrixEntry(i, _, v) => (i, math.abs(v)) }.reduceByKey(_ + _).values.max()

  def frobeniusNormSquared(): Double = matrix.entries.map { case MatrixEntry(_, _, v) => v * v }.sum()

  def scalarMultiply(scalar: Double): CoordinateMatrix = {
    val newEntries = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, v * scalar) }
    new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
  }

  def iterativeInverse(config: IterativeInverseConfig = IterativeInverseConfig()): CoordinateMatrix =
    fromBlock(_.iterativeInverse(config))

  def multiply(other: CoordinateMatrix): CoordinateMatrix = multiply(matrix, other)

  private def multiply(left: CoordinateMatrix, right: CoordinateMatrix): CoordinateMatrix = {
    val partitioner = new HashPartitioner(estimatedPartitionCountFor(right))
    val leftByMid = left.entries.map { case MatrixEntry(i, j, v) => (j, (i, v)) }.partitionBy(partitioner)
    val rightByMid = right.entries.map { case MatrixEntry(j, k, w) => (j, (k, w)) }.partitionBy(partitioner)
    val productEntries = leftByMid.join(rightByMid)
      .map { case (_, ((i, v), (k, w))) => ((i, k), v * w) }
      .reduceByKey(partitioner, _ + _)
      .filter { case (_, sum) => sum != 0.0 }
      .map { case ((i, k), sum) => MatrixEntry(i, k, sum) }
    new CoordinateMatrix(productEntries, left.numRows(), right.numCols())
  }

  def add(other: CoordinateMatrix): CoordinateMatrix = add(matrix, other)

  private def add(left: CoordinateMatrix, right: CoordinateMatrix): CoordinateMatrix = {
    val partitioner = new HashPartitioner(estimatedPartitionCountFor(right))
    val entries = keyedEntries(left.entries)
      .partitionBy(partitioner)
      .union(keyedEntries(right.entries).partitionBy(partitioner))
      .reduceByKey(partitioner, _ + _)
      .filter { case (_, sum) => sum != 0.0 }
      .map { case ((i, j), v) => MatrixEntry(i, j, v) }
    new CoordinateMatrix(entries, left.numRows(), left.numCols())
  }

  def subtract(other: CoordinateMatrix): CoordinateMatrix = subtract(matrix, other)

  private def subtract(left: CoordinateMatrix, right: CoordinateMatrix): CoordinateMatrix = {
    val partitioner = new HashPartitioner(estimatedPartitionCountFor(right))
    val entries = keyedEntries(left.entries)
      .partitionBy(partitioner)
      .union(keyedEntries(right.entries, scale = -1.0).partitionBy(partitioner))
      .reduceByKey(partitioner, _ + _)
      .filter { case (_, sum) => sum != 0.0 }
      .map { case ((i, j), v) => MatrixEntry(i, j, v) }
    new CoordinateMatrix(entries, left.numRows(), left.numCols())
  }

  def partitionBy(partitioner: Partitioner): CoordinateMatrix = {
    val newEntries = matrix.entries.keyBy(x => (x.i, x.j)).partitionBy(partitioner).values
    new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
  }

  def transpose(): CoordinateMatrix = {
    val t = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(j, i, v) }
    new CoordinateMatrix(t, matrix.numCols(), matrix.numRows())
  }

  def pseudoInverse(side: PseudoInverseSide): CoordinateMatrix = pseudoInverse(side, RecursiveInverseConfig())

  def pseudoInverse(side: PseudoInverseSide, config: RecursiveInverseConfig): CoordinateMatrix =
    fromBlock(_.pseudoInverse(side, config))
}
