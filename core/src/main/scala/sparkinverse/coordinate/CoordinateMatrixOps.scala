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
import org.slf4j.LoggerFactory

final class CoordinateMatrixOps private[sparkinverse] (val matrix: CoordinateMatrix) {
  private val logger = LoggerFactory.getLogger(classOf[CoordinateMatrixOps])
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
      "Checkpoint directory must be configured when useCheckpoints=true. " +
      "Use sc.setCheckpointDir() to set the checkpoint directory.")
    require(matrix.numRows() == matrix.numCols(), 
      s"Matrix must be square for inversion. Found ${matrix.numRows()} rows and ${matrix.numCols()} columns.")
    
    // Additional validation
    if (matrix.numRows() > 100000) {
      logger.warn("Large matrix detected ({}x{}). Consider using iterative methods for better performance.", 
        matrix.numRows(), matrix.numCols())
    }
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
    
    require(matrix.numRows() == matrix.numCols(), 
      s"Matrix must be square for SVD inversion. Found ${matrix.numRows()} rows and ${matrix.numCols()} columns.")
    
    val svd = indexed.computeSVD(n, computeU = true, rCond = 0)
    
    // Check for singular matrix
    if (svd.s.size < n) {
      val smallestSingularValue = if (svd.s.size > 0) svd.s.toArray.min else 0.0
      throw new IllegalArgumentException(
        s"Matrix is singular (rank deficiency detected). " +
        s"Matrix size: ${n}x${n}, Non-zero singular values: ${svd.s.size}, " +
        s"Smallest singular value: $smallestSingularValue. " +
        s"Consider using a pseudo-inverse or adding regularization.")
    }
    
    // Check for near-singular matrix
    val smallestSingularValue = svd.s.toArray.min
    val largestSingularValue = svd.s.toArray.max
    val conditionNumber = largestSingularValue / smallestSingularValue
    if (conditionNumber > 1e12) {
      logger.warn("Matrix is ill-conditioned (condition number ~{}). " +
        "SVD inverse may be numerically unstable. Consider using iterative methods with regularization.", 
        conditionNumber)
    }
    
    val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
    transpose(svd.U.multiply(invS).multiply(svd.V.transpose).toCoordinateMatrix())
  }

  def localInverse(): CoordinateMatrix = {
    svdInverse()
  }

  def negate(): CoordinateMatrix = {
    val newEntries = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, -v) }
    new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
  }

  def inverse(): CoordinateMatrix = inverse(RecursiveInverseConfig())

  def inverse(config: RecursiveInverseConfig): CoordinateMatrix = inverseInternal(config, depth = 0)

  private[coordinate] def inverseInternal(config: RecursiveInverseConfig, depth: Int): CoordinateMatrix = {
    validateInverseInputs(config.useCheckpoints)
    if (matrix.numRows() <= math.max(1L, config.limit.toLong)) {
      return localInverse()
    }
    val m = ((matrix.numCols() + 1) / 2).toInt
    val entries = matrix.entries
    val numParts = matrix.entries.getNumPartitions

    logger.info("Input matrix shape: {}, {} At depth={}", matrix.numRows(), matrix.numCols(), depth)

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

  private def buildHyperpowerCorrection(eye: CoordinateMatrix, residual: CoordinateMatrix, order: Int,
                                        storageLevel: StorageLevel): (CoordinateMatrix, Seq[CoordinateMatrix]) = {
    require(order >= 2, "hyperpower order must be at least 2")
    var correction = add(eye, residual)
    val extraPowers = ListBuffer.empty[CoordinateMatrix]
    var currentPower = residual
    var exponent = 2
    while (exponent < order) {
      val nextPower = multiply(currentPower, residual)
      nextPower.entries.persist(storageLevel)
      extraPowers += nextPower
      correction = add(correction, nextPower)
      currentPower = nextPower
      exponent += 1
    }
    (correction, extraPowers.toList)
  }

  private def hyperpowerInverseInternal(config: IterativeInverseConfig, order: Int, algorithmName: String): CoordinateMatrix = {
    require(order >= 2, "hyperpower order must be at least 2")
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
    logger.info("{}: n={}, ||A||_1={}, ||A||_inf={}, alpha={}, order={}", algorithmName, n, norm1, normInfValue, alpha, order)

    var x = scalarMultiply(alpha, transpose(matrix))
    x.entries.persist(storageLevel)
    MatrixInternals.maybeCheckpoint(x.entries, config.useCheckpoints, config.tuning.useLocalCheckpoint)

    val eye = MatrixInternals.eyeCoordinateMatrix(n, 1.0, iterativeStorageLevel, matrix)

    var converged = false
    var iter = 0
    while (iter < config.maxIter && !converged) {
      iter += 1
      val ax = multiply(matrix, x)
      ax.entries.persist(storageLevel)

      val residual = subtract(eye, ax)
      residual.entries.persist(storageLevel)
      val metric = math.sqrt(new CoordinateMatrixOps(residual).frobeniusNormSquared()) / n
      logger.debug("{} iter={}: ||I - A*X||_F / n = {}", algorithmName, iter, metric)
      if (metric < config.tolerance) {
        converged = true
      }

      if (!converged) {
        val (correction, extraPowers) = buildHyperpowerCorrection(eye, residual, order, storageLevel)
        val xNew = multiply(x, correction)
        val oldX = x
        x = xNew
        x.entries.persist(storageLevel)
        if (iter % effectiveCheckpointEvery == 0) {
          MatrixInternals.maybeCheckpoint(x.entries, config.useCheckpoints, config.tuning.useLocalCheckpoint)
          x.entries.count()
        }
        oldX.entries.unpersist(true)
        extraPowers.foreach(_.entries.unpersist(true))
      }

      residual.entries.unpersist(true)
      ax.entries.unpersist(true)
    }

    if (!converged) {
      logger.warn("{} did not converge after {} iterations", algorithmName, config.maxIter)
    }
    if (shouldPersistInput) {
      matrix.entries.unpersist(false)
    }
    x
  }

  def iterativeInverse(): CoordinateMatrix = iterativeInverse(IterativeInverseConfig())

  def iterativeInverse(config: IterativeInverseConfig): CoordinateMatrix =
    hyperpowerInverseInternal(config, order = 2, algorithmName = "iterativeInverse")

  def hyperpowerInverse(): CoordinateMatrix = hyperpowerInverse(IterativeInverseConfig())

  def hyperpowerInverse(config: IterativeInverseConfig): CoordinateMatrix =
    hyperpowerInverse(order = 3, config)

  def hyperpowerInverse(order: Int): CoordinateMatrix = hyperpowerInverse(order, IterativeInverseConfig())

  def hyperpowerInverse(order: Int, config: IterativeInverseConfig): CoordinateMatrix =
    hyperpowerInverseInternal(config, order = order, algorithmName = "hyperpowerInverse")

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
