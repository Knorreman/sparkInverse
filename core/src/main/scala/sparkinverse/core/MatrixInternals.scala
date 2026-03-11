package sparkinverse.core

import breeze.linalg.{DenseMatrix => BDM, inv => BINV}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

private[sparkinverse] object MatrixInternals {
  private val eyeBlockMatrixMap = mutable.Map.empty[(Long, Double, Int, Int), BlockMatrix]
  private val eyeCoordinateMatrixMap = mutable.Map.empty[(Long, Double), CoordinateMatrix]

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

  def luInverse(data: Array[Double], n: Int): Array[Double] = {
    require(data.length == n * n, s"luInverse: expected ${n * n} elements, got ${data.length}")
    require(n > 0, "luInverse: matrix dimension must be positive")
    val a = new BDM[Double](n, n, data.clone())
    val inv = BINV(a)
    inv.toArray
  }

  def maybeCheckpoint[T](rdd: RDD[T], useCheckpoints: Boolean, useLocalCheckpoint: Boolean): Unit = {
    if (useCheckpoints) {
      rdd.checkpoint()
    } else if (useLocalCheckpoint) {
      rdd.localCheckpoint()
    }
  }

  def eyeBlockMatrix(n: Long, value: Double, rowsPerBlock: Int, colsPerBlock: Int, storageLevel: StorageLevel,
                     template: BlockMatrix): BlockMatrix = {
    val key = (n, value, rowsPerBlock, colsPerBlock)
    eyeBlockMatrixMap.getOrElseUpdate(key, {
      val sc = template.blocks.sparkContext
      val diagonal = sc.range(start = 0, end = n).map(i => MatrixEntry(i, i, value))
      val cm = new CoordinateMatrix(diagonal, n, n)
      val bm = cm.toBlockMatrix(rowsPerBlock, colsPerBlock)
      bm.blocks.setName("eye_" + key)
      bm.blocks.persist(storageLevel)
      bm
    })
  }

  def eyeCoordinateMatrix(n: Long, value: Double, storageLevel: StorageLevel,
                          template: CoordinateMatrix): CoordinateMatrix = {
    eyeCoordinateMatrixMap.getOrElseUpdate((n, value), {
      val sc = template.entries.sparkContext
      val diagonal = sc.range(start = 0, end = n).map(i => MatrixEntry(i, i, value))
      val cm = new CoordinateMatrix(diagonal, n, n)
      cm.entries.setName("eye_" + (n, value))
      cm.entries.persist(storageLevel)
      cm
    })
  }
}
