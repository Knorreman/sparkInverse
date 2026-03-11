package sparkinverse.block

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import sparkinverse.api.{IterativeInverseConfig, IterativeTuning, RecursiveInverseConfig, RecursiveTuning}
import sparkinverse.core.MatrixInternals

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

final class BlockMatrixOps private[sparkinverse] (val matrix: BlockMatrix) {
  private val iterativeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
  private val cachedMatrices: ListBuffer[BlockMatrix] = mutable.ListBuffer.empty

  private case class BlockQuadrants(e: BlockMatrix, f: BlockMatrix, g: BlockMatrix, h: BlockMatrix)

  private def withName(mat: BlockMatrix, name: String): BlockMatrix = {
    mat.blocks.setName(name)
    mat
  }

  private def persistAndTrack(mat: BlockMatrix, useCheckpoints: Boolean): BlockMatrix = {
    if (useCheckpoints) {
      mat.blocks.persist(iterativeStorageLevel)
      mat.blocks.checkpoint()
    } else {
      mat.blocks.persist(iterativeStorageLevel)
    }
    cachedMatrices.addOne(mat)
    mat
  }

  private def validateInverseInputs(useCheckpoints: Boolean): Unit = {
    require(!useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
      "Checkpointing dir has to be set when useCheckpoints=true!")
    require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
    require(matrix.colsPerBlock == matrix.rowsPerBlock, "Sub-matrices has to be square!")
  }

  private def effectiveMidDimSplits(numMidDimSplits: Int, tuning: RecursiveTuning): Int = {
    val requested = math.max(1, numMidDimSplits)
    if (!tuning.adaptiveMidDimSplits || requested > 1) {
      requested
    } else {
      val blockRows = ((matrix.numRows() + matrix.rowsPerBlock - 1) / matrix.rowsPerBlock).toInt
      val blockCols = ((matrix.numCols() + matrix.colsPerBlock - 1) / matrix.colsPerBlock).toInt
      val partitionBound = math.max(1, matrix.blocks.getNumPartitions / 2)
      math.max(1, math.min(16, math.min(math.min(blockRows, blockCols), partitionBound)))
    }
  }

  private def effectiveIterativeMidDimSplits(numMidDimSplits: Int, tuning: IterativeTuning): Int = {
    val requested = math.max(1, numMidDimSplits)
    if (!tuning.adaptiveMidDimSplits || requested > 1) {
      requested
    } else {
      val blockRows = ((matrix.numRows() + matrix.rowsPerBlock - 1) / matrix.rowsPerBlock).toInt
      val blockCols = ((matrix.numCols() + matrix.colsPerBlock - 1) / matrix.colsPerBlock).toInt
      val partitionBound = math.max(1, matrix.blocks.getNumPartitions / 2)
      val adaptiveMax = math.max(1, tuning.maxAdaptiveMidDimSplits)
      math.max(1, math.min(adaptiveMax, math.min(math.min(blockRows, blockCols), partitionBound)))
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
    val svd = indexed.computeSVD(n, computeU = true, rCond = 0)
    require(svd.s.size >= n,
      "svdInverse called on singular matrix." + indexed.rows.collect().mkString("Array(", ", ", ")") +
        svd.s.toArray.mkString("Array(", ", ", ")"))

    val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
    svd.U.multiply(invS).multiply(svd.V.transpose).toBlockMatrix(colsPerBlock, rowsPerBlock).transpose
  }

  def localInverse(): BlockMatrix = {
    val localMat = matrix.toLocalMatrix()
    val n = localMat.numRows
    val invData = MatrixInternals.luInverse(localMat.toArray, n)
    localDenseToBlockMatrix(invData, n, matrix.rowsPerBlock, matrix.colsPerBlock)
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
    val m = (numBlockCols + 1) / 2
    val splitSize = math.min(matrix.numRows(), m.toLong * rowsPerBlock)
    val numParts = matrix.blocks.getNumPartitions

    println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

    val res = matrix.numRows() - splitSize
    val BlockQuadrants(e, f, g, h) = splitQuadrants(m, splitSize, res)
    persistAndTrack(e, config.useCheckpoints)
    persistAndTrack(f, config.useCheckpoints)
    persistAndTrack(g, config.useCheckpoints)
    persistAndTrack(h, config.useCheckpoints)

    val recurseThresholdInBlocks = math.max(1, config.limit / colsPerBlock)
    val eInv = invertPartition(e, m, recurseThresholdInBlocks, config, depth, "E_inv")
    val geInv = withName(g.multiply(eInv, midSplits), "GE_inv")
    val eInvF = withName(eInv.multiply(f, midSplits), "E_invF")
    persistAndTrack(geInv, config.useCheckpoints)
    persistAndTrack(eInvF, config.useCheckpoints)

    val schur = withName(h.subtract(g.multiply(eInvF, midSplits)), "S")
    persistAndTrack(schur, config.useCheckpoints)

    val sInv = invertPartition(schur, m, recurseThresholdInBlocks, config, depth, "S_inv")
    val sInvGeInv = withName(sInv.multiply(geInv, midSplits), "S_invGE_inv")
    val eInvFSInv = withName(eInvF.multiply(sInv, midSplits), "E_invFS_inv")
    persistAndTrack(sInvGeInv, config.useCheckpoints)
    persistAndTrack(eInvFSInv, config.useCheckpoints)

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
      (0 until nCols).map { c =>
        val colSum = (0 until nRows).map(r => math.abs(arr(r + c * nRows))).sum
        (j * cpb + c, colSum)
      }
    }.reduceByKey(_ + _).values.max()
  }

  def normInf(): Double = {
    val rpb = matrix.rowsPerBlock
    matrix.blocks.flatMap { case ((i, _), mat) =>
      val arr = mat.toArray
      val nRows = mat.numRows
      val nCols = mat.numCols
      (0 until nRows).map { r =>
        val rowSum = (0 until nCols).map(c => math.abs(arr(r + c * nRows))).sum
        (i * rpb + r, rowSum)
      }
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

  def iterativeInverse(): BlockMatrix = iterativeInverse(IterativeInverseConfig())

  def iterativeInverse(config: IterativeInverseConfig): BlockMatrix = {
    require(!config.useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
      "Checkpointing dir has to be set when useCheckpoints=true!")
    require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")

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
    val alpha = 1.0 / (norm1 * normInfValue)
    println(s"iterativeInverse: n=$n, ||A||_1=$norm1, ||A||_inf=$normInfValue, alpha=$alpha")

    var x = matrix.transpose
    x = scalarMultiply(alpha)
    x.blocks.persist(storageLevel)
    MatrixInternals.maybeCheckpoint(x.blocks, config.useCheckpoints, tuning.useLocalCheckpoint)

    val eye = MatrixInternals.eyeBlockMatrix(n, 1.0, matrix.rowsPerBlock, matrix.colsPerBlock, iterativeStorageLevel, matrix)
    val twoEye = MatrixInternals.eyeBlockMatrix(n, 2.0, matrix.rowsPerBlock, matrix.colsPerBlock, iterativeStorageLevel, matrix)

    var converged = false
    var iter = 0
    while (iter < config.maxIter && !converged) {
      iter += 1
      val ax = matrix.multiply(x, midSplits)
      ax.blocks.persist(storageLevel)

      val residual = eye.subtract(ax)
      val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
      println(s"iterativeInverse iter=$iter: ||I - A*X||_F / n = $metric")
      if (metric < config.tolerance) {
        converged = true
      }

      if (!converged) {
        val twoIMinusAx = twoEye.subtract(ax)
        val xNew = x.multiply(twoIMinusAx, midSplits)
        val oldX = x
        x = xNew
        x.blocks.persist(storageLevel)
        if (iter % effectiveCheckpointEvery == 0) {
          MatrixInternals.maybeCheckpoint(x.blocks, config.useCheckpoints, tuning.useLocalCheckpoint)
          x.blocks.count()
        }
        oldX.blocks.unpersist(true)
      }

      ax.blocks.unpersist(true)
    }

    if (!converged) {
      println(s"Warning: iterativeInverse did not converge after ${config.maxIter} iterations")
    }

    if (shouldPersistInput) {
      matrix.blocks.unpersist(false)
    }

    x
  }

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
