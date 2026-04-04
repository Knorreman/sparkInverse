package sparkinverse.block

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import sparkinverse.api.{IterativeInverseConfig, IterativeTuning, RecursiveInverseConfig, RecursiveTuning}
import sparkinverse.core.MatrixInternals

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import org.slf4j.LoggerFactory

private[block] object BlockMatrixOps {
  private val logger = LoggerFactory.getLogger(classOf[BlockMatrixOps])
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

  private def persistAndTrack(mat: BlockMatrix, useCheckpoints: Boolean): BlockMatrix = {
    persistAndTrack(mat, useCheckpoints, shouldPersist(mat))
  }

  private def shouldPersist(mat: BlockMatrix): Boolean = {
    // Estimate memory usage: each element is 8 bytes (double)
    val estimatedElements = mat.numRows() * mat.numCols()
    // Use configuration threshold if available, otherwise use default
    val threshold = 1000000 // Default: 1M elements
    estimatedElements > threshold
  }

  private def estimateConditionNumber(): Double = {
    val n1 = normOne()
    val nInf = normInf()
    // Simple condition number estimate using norm products
    // This is not exact but gives an indication of conditioning
    n1 * nInf
  }

  private def computeImprovedInitialAlpha(conditionThreshold: Double = 1e6): Double = {
    val n1 = normOne()
    val nInf = normInf()
    val alpha = 1.0 / (n1 * nInf)
    
    // For ill-conditioned matrices, use a smaller initial step size
    val conditionEstimate = n1 * nInf
    if (conditionEstimate > conditionThreshold * 10) {
      // Matrix appears ill-conditioned, use more conservative initial step
      alpha * 0.5
    } else if (conditionEstimate > conditionThreshold) {
      // Moderately ill-conditioned
      alpha * 0.75
    } else {
      alpha
    }
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

  private def effectiveMidDimSplits(numMidDimSplits: Int, tuning: RecursiveTuning): Int = {
    val requested = math.max(1, numMidDimSplits)
    if (!tuning.adaptiveMidDimSplits || requested > 1) {
      requested
    } else {
      computeAdaptiveSplits()
    }
  }

  private def computeAdaptiveSplits(): Int = {
    val blockRows = ((matrix.numRows() + matrix.rowsPerBlock - 1) / matrix.rowsPerBlock).toInt
    val blockCols = ((matrix.numCols() + matrix.colsPerBlock - 1) / matrix.colsPerBlock).toInt
    val partitionBound = math.max(1, matrix.blocks.getNumPartitions / 2)
    
    // Base splits from geometry
    val baseSplits = math.min(math.min(blockRows, blockCols), partitionBound)
    
    // Adjust based on matrix size
    val n = matrix.numRows()
    val sizeFactor = if (n > 10000) {
      // Large matrix - use more splits for better parallelism
      math.min(16, baseSplits * 2)
    } else if (n > 1000) {
      // Medium matrix
      math.min(8, baseSplits)
    } else {
      // Small matrix - fewer splits to avoid overhead
      math.max(1, baseSplits / 2)
    }
    
    // Ensure we don't create too many small blocks
    val minBlockSize = 100 // Minimum block size in elements
    val maxSplits = math.min(sizeFactor, (matrix.numRows() / minBlockSize).toInt)
    
    math.max(1, math.min(16, maxSplits))
  }

  private def effectiveIterativeMidDimSplits(numMidDimSplits: Int, tuning: IterativeTuning): Int = {
    val requested = math.max(1, numMidDimSplits)
    if (!tuning.adaptiveMidDimSplits || requested > 1) {
      requested
    } else {
      val baseSplits = computeAdaptiveSplits()
      val adaptiveMax = math.max(1, tuning.maxAdaptiveMidDimSplits)
      math.max(1, math.min(adaptiveMax, baseSplits))
    }
  }

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

  private def localDenseToBlockMatrix(data: Array[Double], n: Int, rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    val numBlockRows = (n + rowsPerBlock - 1) / rowsPerBlock
    val numBlockCols = (n + colsPerBlock - 1) / colsPerBlock
    val blockCount = numBlockRows * numBlockCols
    val blocks = new Array[((Int, Int), Matrix)](blockCount)

    var idx = 0
    var bi = 0
    while (bi < numBlockRows) {
      val rowStart = bi * rowsPerBlock
      val blockNumRows = math.min(rowsPerBlock, n - rowStart)
      var bj = 0
      while (bj < numBlockCols) {
        val colStart = bj * colsPerBlock
        val blockNumCols = math.min(colsPerBlock, n - colStart)
        val blockData = Array.ofDim[Double](blockNumRows * blockNumCols)
        var c = 0
        while (c < blockNumCols) {
          val globalCol = colStart + c
          val srcOffset = rowStart + globalCol * n
          val dstOffset = c * blockNumRows
          System.arraycopy(data, srcOffset, blockData, dstOffset, blockNumRows)
          c += 1
        }
        blocks(idx) = ((bi, bj), new DenseMatrix(blockNumRows, blockNumCols, blockData).asInstanceOf[Matrix])
        idx += 1
        bj += 1
      }
      bi += 1
    }

    val sc = matrix.blocks.sparkContext
    val targetParts = math.max(1, math.min(blockCount, matrix.blocks.getNumPartitions))
    val blockRdd = sc.parallelize(blocks.toSeq, targetParts)
    new BlockMatrix(blockRdd, rowsPerBlock, colsPerBlock, n.toLong, n.toLong)
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
    val midSplits = effectiveMidDimSplits(config.numMidDimSplits, config.tuning)
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
    
    // Smart persistence: only persist large matrices
    val shouldPersistE = shouldPersist(e)
    val shouldPersistF = shouldPersist(f)
    val shouldPersistG = shouldPersist(g)
    val shouldPersistH = shouldPersist(h)
    
    persistAndTrack(e, config.useCheckpoints, shouldPersistE)
    persistAndTrack(f, config.useCheckpoints, shouldPersistF)
    persistAndTrack(g, config.useCheckpoints, shouldPersistG)
    persistAndTrack(h, config.useCheckpoints, shouldPersistH)

    val recurseThresholdInBlocks = math.max(1, config.limit / colsPerBlock)
    val eInv = invertPartition(e, m, recurseThresholdInBlocks, config, depth, "E_inv")
    val geInv = withName(g.multiply(eInv, midSplits), "GE_inv")
    val eInvF = withName(eInv.multiply(f, midSplits), "E_invF")
    persistAndTrack(geInv, config.useCheckpoints, shouldPersist(geInv))
    persistAndTrack(eInvF, config.useCheckpoints, shouldPersist(eInvF))

    val schur = withName(h.subtract(g.multiply(eInvF, midSplits)), "S")
    persistAndTrack(schur, config.useCheckpoints, shouldPersist(schur))

    val sInv = invertPartition(schur, m, recurseThresholdInBlocks, config, depth, "S_inv")
    val sInvGeInv = withName(sInv.multiply(geInv, midSplits), "S_invGE_inv")
    val eInvFSInv = withName(eInvF.multiply(sInv, midSplits), "E_invFS_inv")
    persistAndTrack(sInvGeInv, config.useCheckpoints, shouldPersist(sInvGeInv))
    persistAndTrack(eInvFSInv, config.useCheckpoints, shouldPersist(eInvFSInv))

    val topLeft = eInv.add(eInvFSInv.multiply(geInv, midSplits))
    val sc = topLeft.blocks.sparkContext
    val unionedBlocks = sc.union(
      topLeft.blocks,
      shiftAndScaleBlocks(eInvFSInv.blocks, colOffset = m, scale = -1.0),
      shiftAndScaleBlocks(sInvGeInv.blocks, rowOffset = m, scale = -1.0),
      shiftAndScaleBlocks(sInv.blocks, rowOffset = m, colOffset = m)
    )
    val defaultOutputParts = math.max(numParts, math.min(unionedBlocks.getNumPartitions, numParts * 2))
    val outputParts = config.tuning.targetOutputPartitions.getOrElse(defaultOutputParts)
    val allBlocks = MatrixInternals.maybeCoalesceNoShuffle(unionedBlocks, outputParts, config.tuning.unionCoalesceThreshold)
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

  private def squareMultiply(source: BlockMatrix, numMidDimSplits: Int): BlockMatrix = {
    require(source.numRows() == source.numCols(), "Matrix has to be square!")
    require(source.colsPerBlock == source.rowsPerBlock, "Sub-matrices has to be square!")

    val midSplits = math.max(1, numMidDimSplits)
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

  private[sparkinverse] def squareBlocks(numMidDimSplits: Int): BlockMatrix =
    squareMultiply(matrix, numMidDimSplits)

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

  private[sparkinverse] def hyperpowerCorrection(order: Int, numMidDimSplits: Int): BlockMatrix = {
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
      math.max(1, numMidDimSplits),
      iterativeStorageLevel
    )
    extraPowers.foreach(_.blocks.unpersist(false))
    correction
  }

  private def iterativeInverseInternal(config: IterativeInverseConfig, order: Int): BlockMatrix = {
    val algorithmName = s"iterativeInverse(order=$order)"
    require(order >= 2, s"Iterative inverse order must be at least 2. Got order=$order.")
    require(!config.useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
      "Checkpoint directory must be configured when useCheckpoints=true. " +
      "Use sc.setCheckpointDir()to set the checkpoint directory.")
    require(matrix.numRows() == matrix.numCols(), 
      s"Matrix must be square for inversion. Found ${matrix.numRows()} rows and ${matrix.numCols()} columns.")
    
    // Validate configuration
    require(config.maxIter > 0, s"Maximum iterations must be positive. Got maxIter=${config.maxIter}.")
    require(config.tolerance > 0, s"Tolerance must be positive. Got tolerance=${config.tolerance}.")
    require(order >= 2 && order <= 10, 
      s"Iterative inverse order should be between 2 and 10 for numerical stability. Got order=$order.")

    val tuning = config.tuning
    val storageLevel = tuning.persistLevel
    val baseCheckpointEvery = math.max(1, tuning.checkpointEvery)
    val effectiveCheckpointEvery =
      if (matrix.numRows() >= tuning.largeMatrixThreshold) {
        math.max(1, math.min(baseCheckpointEvery, tuning.largeMatrixCheckpointEvery))
      } else {
        baseCheckpointEvery
      }
    val midSplits = effectiveIterativeMidDimSplits(config.numMidDimSplits, tuning)
    val shouldPersistInput = matrix.blocks.getStorageLevel == StorageLevel.NONE
    if (shouldPersistInput) {
      matrix.blocks.persist(storageLevel)
    }

    val n = matrix.numRows()
    val norm1 = normOne()
    val normInfValue = normInf()
    val conditionEstimate = norm1 * normInfValue
    
    val alpha = if (tuning.adaptiveStepSize) {
      computeImprovedInitialAlpha(tuning.conditionNumberThreshold)
    } else {
      1.0 / (norm1 * normInfValue)
    }
    
    logger.info("{}: n={}, ||A||_1={}, ||A||_inf={}, alpha={}, order={}, conditionEstimate={}", 
      algorithmName, n, norm1, normInfValue, alpha, order, conditionEstimate)

    var x = matrix.transpose
    x = scalarMultiply(alpha)
    x.blocks.persist(storageLevel)
    MatrixInternals.maybeCheckpoint(x.blocks, config.useCheckpoints, tuning.useLocalCheckpoint)

    val eye = MatrixInternals.eyeBlockMatrix(n, 1.0, matrix.rowsPerBlock, matrix.colsPerBlock, iterativeStorageLevel, matrix)

    var converged = false
    var iter = 0
    var previousMetric = Double.MaxValue
    var divergenceCount = 0
    val maxDivergenceCount = if (tuning.divergenceDetection) tuning.maxDivergenceCount else Int.MaxValue
    
    while (iter < config.maxIter && !converged && divergenceCount < maxDivergenceCount) {
      iter += 1
      val ax = matrix.multiply(x, midSplits)
      ax.blocks.persist(storageLevel)

      val residual = eye.subtract(ax)
      residual.blocks.persist(storageLevel)
      val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
      logger.debug("{} iter={}: ||I - A*X||_F / n = {}", algorithmName, iter, metric)
      
      // Check for divergence
      if (tuning.divergenceDetection && metric > previousMetric * 1.5) { // Metric increased by more than 50%
        divergenceCount += 1
        logger.warn("{} detected potential divergence at iter {}: metric increased from {} to {}", 
          algorithmName, iter, previousMetric, metric)
      } else {
        divergenceCount = 0 // Reset counter if not diverging
      }
      
      previousMetric = metric
      
      if (metric < config.tolerance) {
        converged = true
      }

      if (!converged && divergenceCount < maxDivergenceCount) {
        val (correction, extraPowers) = buildHyperpowerCorrection(eye, residual, order, midSplits, storageLevel)
        val xNew = x.multiply(correction, midSplits)
        val oldX = x
        x = xNew
        x.blocks.persist(storageLevel)
        if (iter % effectiveCheckpointEvery == 0) {
          MatrixInternals.maybeCheckpoint(x.blocks, config.useCheckpoints, tuning.useLocalCheckpoint)
          x.blocks.count()
        }
        oldX.blocks.unpersist(true)
        extraPowers.foreach(_.blocks.unpersist(true))
      }

      residual.blocks.unpersist(true)
      ax.blocks.unpersist(true)
    }

    if (!converged) {
      if (divergenceCount >= maxDivergenceCount) {
        logger.error("{} failed due to detected divergence after {} iterations. " +
          "Consider using a better initial approximation or smaller step size.", algorithmName, iter)
        throw new IllegalArgumentException(s"$algorithmName failed due to divergence after $iter iterations")
      } else {
        logger.warn("{} did not converge after {} iterations. Last metric: {}", 
          algorithmName, config.maxIter, previousMetric)
      }
    }

    if (shouldPersistInput) {
      matrix.blocks.unpersist(false)
    }

    x
  }

  def iterativeInverse(order: Int = 2, config: IterativeInverseConfig = IterativeInverseConfig()): BlockMatrix =
    iterativeInverseInternal(config, order)

  def leftPseudoInverse(): BlockMatrix = leftPseudoInverse(RecursiveInverseConfig())

  def leftPseudoInverse(config: RecursiveInverseConfig): BlockMatrix = {
    val at = matrix.transpose
    val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
    try {
      val gram = at.multiply(matrix, config.numMidDimSplits)
      val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
      try {
        new BlockMatrixOps(gram).inverse(config).multiply(at, config.numMidDimSplits)
      } finally {
        if (persistedGram) gram.blocks.unpersist(false)
      }
    } finally {
      if (persistedAt) at.blocks.unpersist(false)
    }
  }

  def rightPseudoInverse(): BlockMatrix = rightPseudoInverse(RecursiveInverseConfig())

  def rightPseudoInverse(config: RecursiveInverseConfig): BlockMatrix = {
    val at = matrix.transpose
    val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
    try {
      val gram = matrix.multiply(at, config.numMidDimSplits)
      val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
      try {
        at.multiply(new BlockMatrixOps(gram).inverse(config), config.numMidDimSplits)
      } finally {
        if (persistedGram) gram.blocks.unpersist(false)
      }
    } finally {
      if (persistedAt) at.blocks.unpersist(false)
    }
  }
}
