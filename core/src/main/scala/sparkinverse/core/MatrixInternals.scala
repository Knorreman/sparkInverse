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
}
