package sparkinverse.block

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import sparkinverse.api.{IterativeInverseConfig, PseudoInverseSide, RecursiveInverseConfig}
import sparkinverse.core.MatrixInternals

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import org.slf4j.LoggerFactory

private[block] object BlockMatrixOps {
  final case class MidTaggedBlock(blockIndex: Int, block: Matrix, isLeftRole: Boolean)
  final case class MidBlockBuffers(left: ArrayBuffer[(Int, Matrix)], right: ArrayBuffer[(Int, Matrix)])

  def multiplyBlocks(left: Matrix, right: Matrix): Matrix = right match {
    case dense: DenseMatrix => left.multiply(dense)
    case sparse: SparseMatrix => left.multiply(sparse.toDense)
    case _ => throw new IllegalArgumentException(s"Unrecognized matrix type ${right.getClass}.")
  }

  def addBlocks(left: Matrix, right: Matrix): Matrix = {
    require(left.numRows == right.numRows && left.numCols == right.numCols,
      s"Cannot add blocks of different shapes: ${left.numRows}x${left.numCols} vs ${right.numRows}x${right.numCols}")
    val summed = left.toArray
    val rightValues = right.toArray
    var idx = 0
    while (idx < summed.length) {
      summed(idx) += rightValues(idx)
      idx += 1
    }
    new DenseMatrix(left.numRows, left.numCols, summed).asInstanceOf[Matrix]
  }
}

final class BlockMatrixOps private[sparkinverse] (val matrix: BlockMatrix) {
  private val logger = LoggerFactory.getLogger(classOf[BlockMatrixOps])
  private val iterativeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
  private val cachedMatrices: ListBuffer[BlockMatrix] = mutable.ListBuffer.empty

  private case class BlockQuadrants(e: BlockMatrix, f: BlockMatrix, g: BlockMatrix, h: BlockMatrix)

  private def withName(mat: BlockMatrix, name: String): BlockMatrix = {
    mat.blocks.setName(name)
    mat
  }

  private def shouldPersist(mat: BlockMatrix, threshold: Int = 1000000): Boolean = {
    val estimatedElements = mat.numRows() * mat.numCols()
    estimatedElements > threshold
  }

  private def persistAndTrack(mat: BlockMatrix, useCheckpoints: Boolean): BlockMatrix = {
    persistAndTrack(mat, useCheckpoints, shouldPersist(mat))
  }

  private def persistAndTrack(mat: BlockMatrix, useCheckpoints: Boolean, forcePersist: Boolean): BlockMatrix = {
    if (forcePersist) {
      if (useCheckpoints) {
        mat.blocks.persist(iterativeStorageLevel)
        mat.blocks.checkpoint()
      } else {
        mat.blocks.persist(iterativeStorageLevel)
      }
      cachedMatrices.addOne(mat)
    }
    mat
  }

  private def validateInverseInputs(useCheckpoints: Boolean): Unit = {
    require(!useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
      "Checkpoint directory must be configured when useCheckpoints=true. " +
      "Use sc.setCheckpointDir() to set the checkpoint directory.")
    require(matrix.numRows() == matrix.numCols(), 
      s"Matrix must be square for inversion. Found ${matrix.numRows()} rows and ${matrix.numCols()} columns.")
    require(matrix.colsPerBlock == matrix.rowsPerBlock, 
      s"Block dimensions must be square. Found rowsPerBlock=${matrix.rowsPerBlock} and colsPerBlock=${matrix.colsPerBlock}.")
    
    // Additional validation for iterative methods
    if (matrix.numRows() > 100000) {
      logger.warn("Large matrix detected ({}x{}). Consider using iterative methods for better performance.", 
        matrix.numRows(), matrix.numCols())
    }
  }

  private def effectiveMidDimSplits(requestedSplits: Int): Int = math.max(1, requestedSplits)

  private def persistIfNeeded(rdd: RDD[((Int, Int), Matrix)], storageLevel: StorageLevel): Boolean = {
    val shouldPersist = rdd.getStorageLevel == StorageLevel.NONE
    if (shouldPersist) {
      rdd.persist(storageLevel)
    }
    shouldPersist
  }

  private def splitQuadrants(m: Int, splitSize: Long, res: Long): BlockQuadrants = {
    val rowsPerBlock = matrix.rowsPerBlock
    val colsPerBlock = matrix.colsPerBlock
    val blocks = matrix.blocks

    val e = withName(new BlockMatrix(
      blocks.filter { case ((i, j), _) => i < m && j < m },
      rowsPerBlock, colsPerBlock, splitSize, splitSize), "E")
    val f = withName(new BlockMatrix(
      blocks.filter { case ((i, j), _) => i < m && j >= m }.map { case ((i, j), block) => ((i, j - m), block) },
      rowsPerBlock, colsPerBlock, splitSize, res), "F")
    val g = withName(new BlockMatrix(
      blocks.filter { case ((i, j), _) => i >= m && j < m }.map { case ((i, j), block) => ((i - m, j), block) },
      rowsPerBlock, colsPerBlock, res, splitSize), "G")
    val h = withName(new BlockMatrix(
      blocks.filter { case ((i, j), _) => i >= m && j >= m }.map { case ((i, j), block) => ((i - m, j - m), block) },
      rowsPerBlock, colsPerBlock, res, res), "H")

    BlockQuadrants(e, f, g, h)
  }

  private def shiftAndScaleBlocks(blocks: RDD[((Int, Int), Matrix)], rowOffset: Int = 0, colOffset: Int = 0,
                                  scale: Double = 1.0): RDD[((Int, Int), Matrix)] = {
    if (rowOffset == 0 && colOffset == 0 && scale == 1.0) {
      blocks
    } else {
      blocks.map { case ((i, j), mat) =>
        val shiftedIndex = (i + rowOffset, j + colOffset)
        val adjusted = if (scale == 1.0) mat else MatrixInternals.scaleDenseCopy(mat, scale)
        (shiftedIndex, adjusted)
      }
    }
  }

  private def invertPartition(partition: BlockMatrix, partitionBlockSize: Int, recurseThresholdInBlocks: Int,
                              config: RecursiveInverseConfig, depth: Int, name: String): BlockMatrix = {
    val inv =
      if (partitionBlockSize > recurseThresholdInBlocks) {
        new BlockMatrixOps(partition).inverseInternal(config, depth + 1)
      } else {
        new BlockMatrixOps(partition).localInverse()
      }
    persistAndTrack(withName(inv, name), config.useCheckpoints)
  }

  def svdInverse(): BlockMatrix = {
    val colsPerBlock = matrix.colsPerBlock
    val rowsPerBlock = matrix.rowsPerBlock
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
    svd.U.multiply(invS).multiply(svd.V.transpose).toBlockMatrix(colsPerBlock, rowsPerBlock).transpose
  }

  def localInverse(): BlockMatrix = {
    svdInverse()
  }

  def negate(): BlockMatrix = {
    val newBlocks = matrix.blocks.map { case ((i, j), mat) =>
      ((i, j), MatrixInternals.scaleDenseCopy(mat, -1.0))
    }
    new BlockMatrix(newBlocks, matrix.rowsPerBlock, matrix.colsPerBlock, matrix.numRows(), matrix.numCols())
  }

  def inverse(): BlockMatrix = inverse(RecursiveInverseConfig())

  def inverse(config: RecursiveInverseConfig): BlockMatrix = inverseInternal(config, depth = 0)

  private[block] def inverseInternal(config: RecursiveInverseConfig, depth: Int): BlockMatrix = {
    validateInverseInputs(config.useCheckpoints)
    val colsPerBlock = matrix.colsPerBlock
    val rowsPerBlock = matrix.rowsPerBlock
    val midSplits = effectiveMidDimSplits(config.midSplits)
    val numBlockCols = ((matrix.numCols() + colsPerBlock - 1) / colsPerBlock).toInt
    if (matrix.numRows() <= config.limit || numBlockCols <= 1) {
      return localInverse()
    }
    val m = (numBlockCols + 1) / 2
    val splitSize = math.min(matrix.numRows(), m.toLong * rowsPerBlock)
    val numParts = matrix.blocks.getNumPartitions

    logger.info("Input matrix shape: {}, {} At depth={}", matrix.numRows(), matrix.numCols(), depth)

    val res = matrix.numRows() - splitSize
    val BlockQuadrants(e, f, g, h) = splitQuadrants(m, splitSize, res)
    
    val persistThreshold = config.minBlockSizeForPersistence

    persistAndTrack(e, config.useCheckpoints, shouldPersist(e, persistThreshold))
    persistAndTrack(f, config.useCheckpoints, shouldPersist(f, persistThreshold))
    persistAndTrack(g, config.useCheckpoints, shouldPersist(g, persistThreshold))
    persistAndTrack(h, config.useCheckpoints, shouldPersist(h, persistThreshold))

    val recurseThresholdInBlocks = math.max(1, config.limit / colsPerBlock)
    val eInv = invertPartition(e, m, recurseThresholdInBlocks, config, depth, "E_inv")
    val geInv = withName(g.multiply(eInv, midSplits), "GE_inv")
    val eInvF = withName(eInv.multiply(f, midSplits), "E_invF")
    persistAndTrack(geInv, config.useCheckpoints, shouldPersist(geInv, persistThreshold))
    persistAndTrack(eInvF, config.useCheckpoints, shouldPersist(eInvF, persistThreshold))

    val schur = withName(h.subtract(g.multiply(eInvF, midSplits)), "S")
    persistAndTrack(schur, config.useCheckpoints, shouldPersist(schur, persistThreshold))

    val sInv = invertPartition(schur, m, recurseThresholdInBlocks, config, depth, "S_inv")
    val sInvGeInv = withName(sInv.multiply(geInv, midSplits), "S_invGE_inv")
    val eInvFSInv = withName(eInvF.multiply(sInv, midSplits), "E_invFS_inv")
    persistAndTrack(sInvGeInv, config.useCheckpoints, shouldPersist(sInvGeInv, persistThreshold))
    persistAndTrack(eInvFSInv, config.useCheckpoints, shouldPersist(eInvFSInv, persistThreshold))

    val topLeft = eInv.add(eInvFSInv.multiply(geInv, midSplits))
    val sc = topLeft.blocks.sparkContext
    val unionedBlocks = sc.union(
      topLeft.blocks,
      shiftAndScaleBlocks(eInvFSInv.blocks, colOffset = m, scale = -1.0),
      shiftAndScaleBlocks(sInvGeInv.blocks, rowOffset = m, scale = -1.0),
      shiftAndScaleBlocks(sInv.blocks, rowOffset = m, colOffset = m)
    )
    val defaultOutputParts = math.max(numParts, math.min(unionedBlocks.getNumPartitions, numParts * 2))
    val outputParts = config.targetOutputPartitions.getOrElse(defaultOutputParts)
    val allBlocks = MatrixInternals.maybeCoalesceNoShuffle(unionedBlocks, outputParts, config.unionCoalesceThreshold)
    val bm = new BlockMatrix(allBlocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
    if (config.useCheckpoints) {
      bm.blocks.persist(iterativeStorageLevel)
      bm.blocks.checkpoint()
    }
    cachedMatrices.foreach(cached => cached.blocks.unpersist(true))
    bm
  }

  def normOne(): Double = {
    val cpb = matrix.colsPerBlock
    matrix.blocks.flatMap { case ((_, j), mat) =>
      val arr = mat.toArray
      val nRows = mat.numRows
      val nCols = mat.numCols
      val results = new Array[(Int, Double)](nCols)
      var c = 0
      while (c < nCols) {
        var colSum = 0.0
        var r = 0
        while (r < nRows) {
          colSum += math.abs(arr(r + c * nRows))
          r += 1
        }
        results(c) = (j * cpb + c, colSum)
        c += 1
      }
      results.iterator
    }.reduceByKey(_ + _).values.max()
  }

  def normInf(): Double = {
    val rpb = matrix.rowsPerBlock
    matrix.blocks.flatMap { case ((i, _), mat) =>
      val arr = mat.toArray
      val nRows = mat.numRows
      val nCols = mat.numCols
      val results = new Array[(Int, Double)](nRows)
      var r = 0
      while (r < nRows) {
        var rowSum = 0.0
        var c = 0
        while (c < nCols) {
          rowSum += math.abs(arr(r + c * nRows))
          c += 1
        }
        results(r) = (i * rpb + r, rowSum)
        r += 1
      }
      results.iterator
    }.reduceByKey(_ + _).values.max()
  }

  def frobeniusNormSquared(): Double = {
    matrix.blocks.map { case (_, mat) =>
      val arr = mat.toArray
      var i = 0
      var sum = 0.0
      while (i < arr.length) {
        val v = arr(i)
        sum += v * v
        i += 1
      }
      sum
    }.sum()
  }

  def scalarMultiply(scalar: Double): BlockMatrix = {
    val newBlocks = matrix.blocks.map { case ((i, j), mat) =>
      ((i, j), MatrixInternals.scaleDenseCopy(mat, scalar))
    }
    new BlockMatrix(newBlocks, matrix.rowsPerBlock, matrix.colsPerBlock, matrix.numRows(), matrix.numCols())
  }

  private def squareMultiply(source: BlockMatrix, midSplits: Int): BlockMatrix = {
    require(source.numRows() == source.numCols(), "Matrix has to be square!")
    require(source.colsPerBlock == source.rowsPerBlock, "Sub-matrices has to be square!")
    require(midSplits >= 1, s"midSplits must be >= 1, got $midSplits")
    val numBlockRows = ((source.numRows() + source.rowsPerBlock - 1) / source.rowsPerBlock).toInt
    val numBlockCols = ((source.numCols() + source.colsPerBlock - 1) / source.colsPerBlock).toInt
    val basePartitions = math.max(1, source.blocks.getNumPartitions)
    val midPartitions = math.max(basePartitions, math.min(numBlockCols * midSplits, basePartitions * midSplits))
    val outputPartitions = math.max(basePartitions, math.min(numBlockRows * numBlockCols, basePartitions * midSplits))
    val midPartitioner = new HashPartitioner(midPartitions)
    val outputPartitioner = new HashPartitioner(outputPartitions)

    val partialProducts = source.blocks
      .flatMap { case ((rowBlockIndex, colBlockIndex), block) =>
        Iterator(
          (colBlockIndex, BlockMatrixOps.MidTaggedBlock(rowBlockIndex, block, isLeftRole = true)),
          (rowBlockIndex, BlockMatrixOps.MidTaggedBlock(colBlockIndex, block, isLeftRole = false))
        )
      }
      .partitionBy(midPartitioner)
      .mapPartitions { iter =>
        val groupedByMid = mutable.HashMap.empty[Int, BlockMatrixOps.MidBlockBuffers]
        iter.foreach { case (midIndex, taggedBlock) =>
          val buffers = groupedByMid.getOrElseUpdate(
            midIndex,
            BlockMatrixOps.MidBlockBuffers(ArrayBuffer.empty[(Int, Matrix)], ArrayBuffer.empty[(Int, Matrix)])
          )
          if (taggedBlock.isLeftRole) {
            buffers.left += ((taggedBlock.blockIndex, taggedBlock.block))
          } else {
            buffers.right += ((taggedBlock.blockIndex, taggedBlock.block))
          }
        }
        groupedByMid.iterator.flatMap { case (_, buffers) =>
          for {
            (rowBlockIndex, leftBlock) <- buffers.left.iterator
            (colBlockIndex, rightBlock) <- buffers.right.iterator
          } yield ((rowBlockIndex, colBlockIndex), BlockMatrixOps.multiplyBlocks(leftBlock, rightBlock))
        }
      }

    val newBlocks = partialProducts.reduceByKey(outputPartitioner,
      (left, right) => BlockMatrixOps.addBlocks(left, right))
    new BlockMatrix(newBlocks, source.rowsPerBlock, source.colsPerBlock, source.numRows(), source.numCols())
  }

  private[sparkinverse] def squareBlocks(midSplits: Int): BlockMatrix =
    squareMultiply(matrix, midSplits)

  // Builds residual powers R^2 .. R^(maxExponent) using repeated squaring to
  // minimize multiply calls (e.g. R^4 = (R^2)^2 needs 2 multiplies instead of 3).
  // CoordinateMatrixOps uses sequential multiplication instead, which may
  // accumulate floating-point errors differently but is simpler for sparse data.
  private def buildHyperpowerPowers(residual: BlockMatrix, maxExponent: Int,
                                    midSplits: Int, storageLevel: StorageLevel): Seq[BlockMatrix] = {
    if (maxExponent < 2) {
      return Seq.empty
    }

    val powers = mutable.HashMap(1 -> residual)
    val builtPowers = ListBuffer.empty[BlockMatrix]
    var exponent = 2
    while (exponent <= maxExponent) {
      val nextPower =
        if (exponent % 2 == 0) {
          squareMultiply(powers(exponent / 2), midSplits)
        } else {
          val split = Integer.highestOneBit(exponent - 1)
          powers(split).multiply(powers(exponent - split), midSplits)
        }
      nextPower.blocks.persist(storageLevel)
      powers(exponent) = nextPower
      builtPowers += nextPower
      exponent += 1
    }
    builtPowers.toList
  }

  private def buildHyperpowerCorrection(eye: BlockMatrix, residual: BlockMatrix, order: Int,
                                        midSplits: Int, storageLevel: StorageLevel): (BlockMatrix, Seq[BlockMatrix]) = {
    require(order >= 2, "hyperpower order must be at least 2")
    var correction = eye.add(residual)
    val extraPowers = buildHyperpowerPowers(residual, order - 1, midSplits, storageLevel)
    extraPowers.foreach { power =>
      correction = correction.add(power)
    }
    (correction, extraPowers)
  }

  private[sparkinverse] def hyperpowerCorrection(order: Int, midSplits: Int): BlockMatrix = {
    require(order >= 2, "hyperpower order must be at least 2")
    require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
    require(matrix.colsPerBlock == matrix.rowsPerBlock, "Sub-matrices has to be square!")

    val eye = MatrixInternals.eyeBlockMatrix(
      matrix.numRows(),
      1.0,
      matrix.rowsPerBlock,
      matrix.colsPerBlock,
      iterativeStorageLevel,
      matrix
    )
    val (correction, extraPowers) = buildHyperpowerCorrection(
      eye,
      matrix,
      order,
      midSplits,
      iterativeStorageLevel
    )
    extraPowers.foreach(_.blocks.unpersist(false))
    correction
  }

  private def iterativeInverseInternal(config: IterativeInverseConfig): BlockMatrix = {
    val order = config.order
    val algorithmName = s"iterativeInverse(order=$order)"
    require(order >= 2 && order <= 10,
      s"Iterative inverse order should be between 2 and 10 for numerical stability. Got order=$order.")
    require(!config.useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
      "Checkpoint directory must be configured when useCheckpoints=true. " +
      "Use sc.setCheckpointDir() to set the checkpoint directory.")
    require(matrix.numRows() == matrix.numCols(), 
      s"Matrix must be square for inversion. Found ${matrix.numRows()} rows and ${matrix.numCols()} columns.")
    require(config.maxIter > 0, s"Maximum iterations must be positive. Got maxIter=${config.maxIter}.")
    require(config.tolerance > 0, s"Tolerance must be positive. Got tolerance=${config.tolerance}.")

    val storageLevel = config.persistLevel
    val checkpointEvery = math.max(1, config.checkpointEvery)
    val midSplits = math.max(1, config.midSplits)
    val shouldPersistInput = matrix.blocks.getStorageLevel == StorageLevel.NONE
    if (shouldPersistInput) {
      matrix.blocks.persist(storageLevel)
    }

    val n = matrix.numRows()
    val norm1 = normOne()
    val normInfValue = normInf()
    val alpha = 1.0 / (norm1 * normInfValue)

    logger.info("{}: n={}, ||A||_1={}, ||A||_inf={}, alpha={}, order={}",
      algorithmName, n, norm1, normInfValue, alpha, order)

    var x = new BlockMatrixOps(matrix.transpose).scalarMultiply(alpha)
    x.blocks.persist(storageLevel)
    MatrixInternals.maybeCheckpoint(x.blocks, config.useCheckpoints)

    val eye = MatrixInternals.eyeBlockMatrix(n, 1.0, matrix.rowsPerBlock, matrix.colsPerBlock, iterativeStorageLevel, matrix)

    var converged = false
    var iter = 0
    var previousMetric = Double.MaxValue

    while (iter < config.maxIter && !converged) {
      iter += 1
      val ax = matrix.multiply(x, midSplits)
      ax.blocks.persist(storageLevel)

      val residual = eye.subtract(ax)
      residual.blocks.persist(storageLevel)
      val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
      logger.debug("{} iter={}: ||I - A*X||_F / n = {}", algorithmName, iter, metric)
      previousMetric = metric
      
      if (metric < config.tolerance) {
        converged = true
      }

      if (!converged) {
        val (correction, extraPowers) = buildHyperpowerCorrection(eye, residual, order, midSplits, storageLevel)
        val xNew = x.multiply(correction, midSplits)
        val oldX = x
        x = xNew
        x.blocks.persist(storageLevel)
        if (iter % checkpointEvery == 0) {
          MatrixInternals.maybeCheckpoint(x.blocks, config.useCheckpoints)
          x.blocks.count()
        }
        oldX.blocks.unpersist(true)
        extraPowers.foreach(_.blocks.unpersist(true))
      }

      residual.blocks.unpersist(true)
      ax.blocks.unpersist(true)
    }

    if (!converged) {
      logger.warn("{} did not converge after {} iterations. Last metric: {}",
        algorithmName, config.maxIter, previousMetric)
    }

    if (shouldPersistInput) {
      matrix.blocks.unpersist(false)
    }

    x
  }

  def iterativeInverse(config: IterativeInverseConfig = IterativeInverseConfig()): BlockMatrix =
    iterativeInverseInternal(config)

  def pseudoInverse(side: PseudoInverseSide): BlockMatrix = pseudoInverse(side, RecursiveInverseConfig())

  def pseudoInverse(side: PseudoInverseSide, config: RecursiveInverseConfig): BlockMatrix = {
    val at = matrix.transpose
    val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
    try {
      val midSplits = math.max(1, config.midSplits)
      val gram = side match {
        case PseudoInverseSide.Left => at.multiply(matrix, midSplits)
        case PseudoInverseSide.Right => matrix.multiply(at, midSplits)
      }
      val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
      try {
        side match {
          case PseudoInverseSide.Left =>
            new BlockMatrixOps(gram).inverse(config).multiply(at, midSplits)
          case PseudoInverseSide.Right =>
            at.multiply(new BlockMatrixOps(gram).inverse(config), midSplits)
        }
      } finally {
        if (persistedGram) gram.blocks.unpersist(false)
      }
    } finally {
      if (persistedAt) at.blocks.unpersist(false)
    }
  }
}
