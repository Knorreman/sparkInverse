package sparkinverse.core

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

private[sparkinverse] object MatrixInternals {
  def maybeCoalesceNoShuffle[T](rdd: RDD[T], targetPartitions: Int, threshold: Int): RDD[T] = {
    val current = rdd.getNumPartitions
    if (targetPartitions > 0 && current - targetPartitions > threshold) {
      rdd.coalesce(targetPartitions, shuffle = false)
    } else {
      rdd
    }
  }

  def scaleDenseCopy(mat: Matrix, scale: Double): Matrix = {
    val arr = mat.toArray
    var i = 0
    while (i < arr.length) {
      arr(i) *= scale
      i += 1
    }
    new DenseMatrix(mat.numRows, mat.numCols, arr).asInstanceOf[Matrix]
  }

  def maybeCheckpoint[T](rdd: RDD[T], useCheckpoints: Boolean): Unit = {
    if (useCheckpoints) {
      rdd.checkpoint()
    }
  }

  def eyeBlockMatrix(n: Long, value: Double, rowsPerBlock: Int, colsPerBlock: Int, storageLevel: StorageLevel,
                     template: BlockMatrix): BlockMatrix = {
    val sc = template.blocks.sparkContext
    val diagonal = sc.range(start = 0, end = n).map(i => MatrixEntry(i, i, value))
    val cm = new CoordinateMatrix(diagonal, n, n)
    val bm = cm.toBlockMatrix(rowsPerBlock, colsPerBlock)
    bm.blocks.persist(storageLevel)
    bm
  }

  def eyeCoordinateMatrix(n: Long, value: Double, storageLevel: StorageLevel,
                          template: CoordinateMatrix): CoordinateMatrix = {
    val sc = template.entries.sparkContext
    val diagonal = sc.range(start = 0, end = n).map(i => MatrixEntry(i, i, value))
    val cm = new CoordinateMatrix(diagonal, n, n)
    cm.entries.persist(storageLevel)
    cm
  }

  /** Scale a BlockMatrix by a scalar, producing a new BlockMatrix.
    * Unlike scalarMultiply on BlockMatrixOps which uses mapValues,
    * this creates dense copies of each block to ensure mutability safety.
    */
  def scaleBlockMatrix(mat: BlockMatrix, scale: Double): BlockMatrix = {
    val newBlocks = mat.blocks.map { case ((i, j), block) =>
      ((i, j), scaleDenseCopy(block, scale))
    }
    new BlockMatrix(newBlocks, mat.rowsPerBlock, mat.colsPerBlock, mat.numRows(), mat.numCols())
  }

  /** Create a random n×1 column vector as a BlockMatrix with entries
    * drawn from a standard normal distribution, then normalized to unit
    * Frobenius norm. Used for power iteration to estimate σ₁².
    */
  def randomBlockVector(n: Long, colsPerBlock: Int, rowsPerBlock: Int,
                        sc: org.apache.spark.SparkContext, seed: Long): BlockMatrix = {
    import org.apache.spark.mllib.linalg.DenseMatrix
    import org.apache.spark.mllib.linalg.distributed.BlockMatrix

    // Create random blocks for a n×1 matrix
    val numBlockRows = ((n + rowsPerBlock - 1) / rowsPerBlock).toInt
    val rng = new java.util.Random(seed)

    val blocks = (0 until numBlockRows).map { blockIdx =>
      val startRow = blockIdx * rowsPerBlock
      val endRow = math.min(startRow + rowsPerBlock, n.toInt)
      val numRows = endRow - startRow
      val data = new Array[Double](numRows)  // numRows × 1
      var i = 0
      var sumSq = 0.0
      while (i < numRows) {
        val v = rng.nextGaussian()
        data(i) = v
        sumSq += v * v
        i += 1
      }
      ((blockIdx, 0), new DenseMatrix(numRows, 1, data))
    }

    // Normalize to unit Frobenius norm
    var totalNormSq = 0.0
    blocks.foreach { case (_, mat) =>
      val arr = mat.toArray
      var i = 0
      while (i < arr.length) {
        totalNormSq += arr(i) * arr(i)
        i += 1
      }
    }
    val scale = 1.0 / math.sqrt(totalNormSq)
    val normalizedBlocks = blocks.map { case (idx, mat) =>
      (idx, scaleDenseCopy(mat.asInstanceOf[Matrix], scale))
    }.toSeq

    new BlockMatrix(sc.parallelize(normalizedBlocks), rowsPerBlock, colsPerBlock, n, 1)
  }
}
