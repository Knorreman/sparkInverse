package sparkinverse.coordinate

import org.apache.spark.HashPartitioner
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import sparkinverse.api.{IterativeInverseConfig, RecursiveInverseConfig}
import sparkinverse.core.MatrixInternals

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

final class CoordinateMatrixOps private[sparkinverse] (val matrix: CoordinateMatrix) {
  private val iterativeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
  private val cachedMatrices: ListBuffer[CoordinateMatrix] = mutable.ListBuffer.empty

  private case class CoordinateQuadrants(e: CoordinateMatrix, f: CoordinateMatrix, g: CoordinateMatrix, h: CoordinateMatrix)

  private def withName(mat: CoordinateMatrix, name: String): CoordinateMatrix = {
    mat.entries.setName(name)
    mat
  }

  private def persistAndTrack(mat: CoordinateMatrix, useCheckpoints: Boolean): CoordinateMatrix = {
    if (useCheckpoints) {
      mat.entries.persist(iterativeStorageLevel)
      mat.entries.checkpoint()
    } else {
      mat.entries.persist(iterativeStorageLevel)
    }
    cachedMatrices.addOne(mat)
    mat
  }

  private def validateInverseInputs(useCheckpoints: Boolean): Unit = {
    require(!useCheckpoints || matrix.entries.sparkContext.getCheckpointDir.isDefined,
      "Checkpointing dir has to be set when useCheckpoints=true!")
    require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
  }

  private def persistIfNeeded(rdd: RDD[MatrixEntry], storageLevel: StorageLevel): Boolean = {
    val shouldPersist = rdd.getStorageLevel == StorageLevel.NONE
    if (shouldPersist) {
      rdd.persist(storageLevel)
    }
    shouldPersist
  }

  private def partitionCountFor(otherPartitions: Int): Int = {
    math.max(1, math.max(matrix.entries.getNumPartitions, otherPartitions))
  }

  private def estimatedPartitionCountFor(other: CoordinateMatrix): Int = {
    val base = partitionCountFor(other.entries.getNumPartitions)
    val dimFactor = math.max(1, ((matrix.numRows() + matrix.numCols() + other.numCols()) / 4000L).toInt)
    val scaled = base * dimFactor
    math.max(base, math.min(base * 8, scaled))
  }

  private def effectiveIterativeCheckpointEvery(config: IterativeInverseConfig): Int = {
    val base = math.max(1, config.tuning.checkpointEvery)
    if (matrix.numRows() >= config.tuning.largeMatrixThreshold) {
      math.max(1, math.min(base, config.tuning.largeMatrixCheckpointEvery))
    } else {
      base
    }
  }

  private def keyedEntries(entries: RDD[MatrixEntry], scale: Double = 1.0): RDD[((Long, Long), Double)] = {
    entries.map { case MatrixEntry(i, j, v) => ((i, j), v * scale) }
  }

  private def splitQuadrants(entries: RDD[MatrixEntry], m: Int): CoordinateQuadrants = {
    val e = withName(new CoordinateMatrix(entries.filter(x => x.i < m && x.j < m)), "E")
    val f = withName(new CoordinateMatrix(entries.filter(x => x.i < m && x.j >= m)
      .map { case MatrixEntry(i, j, v) => MatrixEntry(i, j - m, v) }), "F")
    val g = withName(new CoordinateMatrix(entries.filter(x => x.i >= m && x.j < m)
      .map { case MatrixEntry(i, j, v) => MatrixEntry(i - m, j, v) }), "G")
    val h = withName(new CoordinateMatrix(entries.filter(x => x.i >= m && x.j >= m)
      .map { case MatrixEntry(i, j, v) => MatrixEntry(i - m, j - m, v) }), "H")
    CoordinateQuadrants(e, f, g, h)
  }

  private def shiftAndScaleEntries(entries: RDD[MatrixEntry], rowOffset: Int = 0, colOffset: Int = 0,
                                   scale: Double = 1.0): RDD[MatrixEntry] = {
    if (rowOffset == 0 && colOffset == 0 && scale == 1.0) {
      entries
    } else {
      entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i + rowOffset, j + colOffset, v * scale) }
    }
  }

  private def invertPartition(partition: CoordinateMatrix, partitionSize: Int,
                              config: RecursiveInverseConfig, depth: Int, name: String): CoordinateMatrix = {
    val inv =
      if (partitionSize > config.limit) {
        new CoordinateMatrixOps(partition).inverseInternal(config, depth + 1)
      } else {
        new CoordinateMatrixOps(partition).localInverse()
      }
    persistAndTrack(withName(inv, name), config.useCheckpoints)
  }

  def svdInverse(): CoordinateMatrix = {
    val indexed = matrix.toIndexedRowMatrix()
    val n = indexed.numCols().toInt
    val svd = indexed.computeSVD(n, computeU = true, rCond = 0)
    require(svd.s.size >= n,
      "svdInverse called on singular matrix." + indexed.rows.collect().mkString("Array(", ", ", ")") +
        svd.s.toArray.mkString("Array(", ", ", ")"))
    val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
    transpose(svd.U.multiply(invS).multiply(svd.V.transpose).toCoordinateMatrix())
  }

  def localInverse(): CoordinateMatrix = {
    val localMat = matrix.toBlockMatrix().toLocalMatrix()
    val n = localMat.numRows
    val invData = MatrixInternals.luInverse(localMat.toArray, n)
    val sc = matrix.entries.sparkContext
    val entries = sc.parallelize(
      for (i <- 0 until n; j <- 0 until n) yield MatrixEntry(i.toLong, j.toLong, invData(i + j * n))
    )
    new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
  }

  def negate(): CoordinateMatrix = {
    val newEntries = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, -v) }
    new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
  }

  def inverse(): CoordinateMatrix = inverse(RecursiveInverseConfig())

  def inverse(config: RecursiveInverseConfig): CoordinateMatrix = inverseInternal(config, depth = 0)

  private[coordinate] def inverseInternal(config: RecursiveInverseConfig, depth: Int): CoordinateMatrix = {
    validateInverseInputs(config.useCheckpoints)
    val m = ((matrix.numCols() + 1) / 2).toInt
    val entries = matrix.entries
    val numParts = matrix.entries.getNumPartitions

    println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

    val CoordinateQuadrants(e, f, g, h) = splitQuadrants(entries, m)
    persistAndTrack(e, config.useCheckpoints)
    persistAndTrack(f, config.useCheckpoints)
    persistAndTrack(g, config.useCheckpoints)
    persistAndTrack(h, config.useCheckpoints)

    val eInv = invertPartition(e, m, config, depth, "E_inv")
    val geInv = persistAndTrack(withName(multiply(g, eInv), "GE_inv"), config.useCheckpoints)
    val eInvF = persistAndTrack(withName(multiply(eInv, f), "E_invF"), config.useCheckpoints)
    val schur = persistAndTrack(withName(subtract(h, multiply(g, eInvF)), "S"), config.useCheckpoints)
    val sInv = invertPartition(schur, m, config, depth, "S_inv")
    val sInvGeInv = persistAndTrack(withName(multiply(sInv, geInv), "S_invGE_inv"), config.useCheckpoints)
    val eInvFSInv = persistAndTrack(withName(multiply(eInvF, sInv), "E_invFS_inv"), config.useCheckpoints)
    val topLeft = add(eInv, multiply(eInvFSInv, geInv))

    val sc = topLeft.entries.sparkContext
    val unionedEntries = sc.union(
      topLeft.entries,
      shiftAndScaleEntries(eInvFSInv.entries, colOffset = m, scale = -1.0),
      shiftAndScaleEntries(sInvGeInv.entries, rowOffset = m, scale = -1.0),
      shiftAndScaleEntries(sInv.entries, rowOffset = m, colOffset = m)
    )
    val defaultOutputParts = math.max(numParts, math.min(unionedEntries.getNumPartitions, numParts * 2))
    val outputParts = config.tuning.targetOutputPartitions.getOrElse(defaultOutputParts)
    val allEntries = MatrixInternals.maybeCoalesceNoShuffle(unionedEntries, outputParts, config.tuning.unionCoalesceThreshold)
    val cm = new CoordinateMatrix(allEntries, matrix.numRows(), matrix.numCols())
    if (config.useCheckpoints) {
      cm.entries.persist(iterativeStorageLevel)
      cm.entries.checkpoint()
    }
    cachedMatrices.foreach(cached => cached.entries.unpersist(true))
    cm
  }

  def normOne(): Double = {
    matrix.entries.map { case MatrixEntry(_, j, v) => (j, math.abs(v)) }.reduceByKey(_ + _).values.max()
  }

  def normInf(): Double = {
    matrix.entries.map { case MatrixEntry(i, _, v) => (i, math.abs(v)) }.reduceByKey(_ + _).values.max()
  }

  def frobeniusNormSquared(): Double = matrix.entries.map { case MatrixEntry(_, _, v) => v * v }.sum()

  def scalarMultiply(scalar: Double): CoordinateMatrix = {
    val newEntries = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, v * scalar) }
    new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
  }

  def iterativeInverse(): CoordinateMatrix = iterativeInverse(IterativeInverseConfig())

  def iterativeInverse(config: IterativeInverseConfig): CoordinateMatrix = {
    require(!config.useCheckpoints || matrix.entries.sparkContext.getCheckpointDir.isDefined,
      "Checkpointing dir has to be set when useCheckpoints=true!")
    require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")

    val storageLevel = config.tuning.persistLevel
    val effectiveCheckpointEvery = effectiveIterativeCheckpointEvery(config)
    val shouldPersistInput = matrix.entries.getStorageLevel == StorageLevel.NONE
    if (shouldPersistInput) {
      matrix.entries.persist(storageLevel)
    }

    val n = matrix.numRows()
    val norm1 = normOne()
    val normInfValue = normInf()
    val alpha = 1.0 / (norm1 * normInfValue)
    println(s"iterativeInverse: n=$n, ||A||_1=$norm1, ||A||_inf=$normInfValue, alpha=$alpha")

    var x = scalarMultiply(alpha, transpose(matrix))
    x.entries.persist(storageLevel)
    MatrixInternals.maybeCheckpoint(x.entries, config.useCheckpoints, config.tuning.useLocalCheckpoint)

    val eye = MatrixInternals.eyeCoordinateMatrix(n, 1.0, iterativeStorageLevel, matrix)
    val twoEye = MatrixInternals.eyeCoordinateMatrix(n, 2.0, iterativeStorageLevel, matrix)

    var converged = false
    var iter = 0
    while (iter < config.maxIter && !converged) {
      iter += 1
      val ax = multiply(matrix, x)
      ax.entries.persist(storageLevel)

      val residual = subtract(eye, ax)
      val metric = math.sqrt(new CoordinateMatrixOps(residual).frobeniusNormSquared()) / n
      println(s"iterativeInverse iter=$iter: ||I - A*X||_F / n = $metric")
      if (metric < config.tolerance) {
        converged = true
      }

      if (!converged) {
        val twoIMinusAx = subtract(twoEye, ax)
        val xNew = multiply(x, twoIMinusAx)
        val oldX = x
        x = xNew
        x.entries.persist(storageLevel)
        if (iter % effectiveCheckpointEvery == 0) {
          MatrixInternals.maybeCheckpoint(x.entries, config.useCheckpoints, config.tuning.useLocalCheckpoint)
          x.entries.count()
        }
        oldX.entries.unpersist(true)
      }

      ax.entries.unpersist(true)
    }

    if (!converged) {
      println(s"Warning: iterativeInverse did not converge after ${config.maxIter} iterations")
    }
    if (shouldPersistInput) {
      matrix.entries.unpersist(false)
    }
    x
  }

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

  def transpose(): CoordinateMatrix = transpose(matrix)

  private def transpose(source: CoordinateMatrix): CoordinateMatrix = {
    val t = source.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(j, i, v) }
    new CoordinateMatrix(t, source.numCols(), source.numRows())
  }

  private def scalarMultiply(scalar: Double, source: CoordinateMatrix): CoordinateMatrix = {
    val entries = source.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, v * scalar) }
    new CoordinateMatrix(entries, source.numRows(), source.numCols())
  }

  def leftPseudoInverse(): CoordinateMatrix = leftPseudoInverse(RecursiveInverseConfig())

  def leftPseudoInverse(config: RecursiveInverseConfig): CoordinateMatrix = {
    val at = transpose()
    val persistedAt = persistIfNeeded(at.entries, iterativeStorageLevel)
    try {
      val gram = multiply(at, matrix)
      val persistedGram = persistIfNeeded(gram.entries, iterativeStorageLevel)
      try {
        multiply(new CoordinateMatrixOps(gram).inverse(config), at)
      } finally {
        if (persistedGram) gram.entries.unpersist(false)
      }
    } finally {
      if (persistedAt) at.entries.unpersist(false)
    }
  }

  def rightPseudoInverse(): CoordinateMatrix = rightPseudoInverse(RecursiveInverseConfig())

  def rightPseudoInverse(config: RecursiveInverseConfig): CoordinateMatrix = {
    val at = transpose()
    val persistedAt = persistIfNeeded(at.entries, iterativeStorageLevel)
    try {
      val gram = multiply(matrix, at)
      val persistedGram = persistIfNeeded(gram.entries, iterativeStorageLevel)
      try {
        multiply(at, new CoordinateMatrixOps(gram).inverse(config))
      } finally {
        if (persistedGram) gram.entries.unpersist(false)
      }
    } finally {
      if (persistedAt) at.entries.unpersist(false)
    }
  }
}
