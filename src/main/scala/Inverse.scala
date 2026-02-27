import org.apache.spark.HashPartitioner
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import breeze.linalg.{DenseMatrix => BDM, inv => BINV}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object Inverse {
  final case class RecursivePerfConfig(
    trace: Boolean = false,
    targetOutputPartitions: Option[Int] = None,
    unionCoalesceThreshold: Int = 8,
    adaptiveMidDimSplits: Boolean = true)

  final case class IterativePerfConfig(
    persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
    checkpointEvery: Int = 5,
    useLocalCheckpoint: Boolean = false,
    adaptiveMidDimSplits: Boolean = true,
    maxAdaptiveMidDimSplits: Int = 16,
    largeMatrixCheckpointEvery: Int = 2,
    largeMatrixThreshold: Int = 4000,
    trace: Boolean = false)

  private val eyeBlockMatrixMap: mutable.Map[(Long, Double, Int, Int), BlockMatrix] = mutable.Map[(Long, Double, Int, Int), BlockMatrix]()
  private val eyeCoordinateMatrixMap: mutable.Map[(Long, Double), CoordinateMatrix] = mutable.Map[(Long, Double), CoordinateMatrix]()

  private def trace(enabled: Boolean, message: => String): Unit = {
    if (enabled) {
      println(message)
    }
  }

  private def timed[A](enabled: Boolean, label: String)(f: => A): A = {
    if (!enabled) {
      f
    } else {
      val t0 = System.nanoTime()
      val out = f
      val elapsedMs = (System.nanoTime() - t0) / 1e6
      println(f"[perf] $label took $elapsedMs%.2f ms")
      out
    }
  }

  private def maybeCoalesceNoShuffle[T](rdd: RDD[T], targetPartitions: Int, threshold: Int): RDD[T] = {
    val current = rdd.getNumPartitions
    if (targetPartitions > 0 && current - targetPartitions > threshold) {
      rdd.coalesce(targetPartitions, shuffle = false)
    } else {
      rdd
    }
  }

  private def scaleDenseCopy(mat: Matrix, scale: Double): Matrix = {
    val arr = mat.toArray
    var i = 0
    while (i < arr.length) {
      arr(i) *= scale
      i += 1
    }
    new DenseMatrix(mat.numRows, mat.numCols, arr).asInstanceOf[Matrix]
  }

  /**
   * Invert an n×n matrix using Breeze dense inverse.
   *
   * @param data Column-major array of length n*n (not modified)
   * @param n    Matrix dimension
   * @return Column-major array of length n*n containing the inverse
   */
  private[Inverse] def luInverse(data: Array[Double], n: Int): Array[Double] = {
    require(data.length == n * n, s"luInverse: expected ${n * n} elements, got ${data.length}")
    require(n > 0, "luInverse: matrix dimension must be positive")
    val a = new BDM[Double](n, n, data.clone())
    val inv = BINV(a)
    inv.toArray
  }

  implicit class BlockMatrixInverse(val matrix: BlockMatrix) {

    private val iterativeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
    private val cachedMatrices: ListBuffer[BlockMatrix] = mutable.ListBuffer.empty
    private case class BlockQuadrants(E: BlockMatrix, F: BlockMatrix, G: BlockMatrix, H: BlockMatrix)

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

    private def effectiveMidDimSplits(numMidDimSplits: Int, perf: RecursivePerfConfig): Int = {
      val requested = math.max(1, numMidDimSplits)
      if (!perf.adaptiveMidDimSplits || requested > 1) {
        requested
      } else {
        val blockRows = ((matrix.numRows() + matrix.rowsPerBlock - 1) / matrix.rowsPerBlock).toInt
        val blockCols = ((matrix.numCols() + matrix.colsPerBlock - 1) / matrix.colsPerBlock).toInt
        val partitionBound = math.max(1, matrix.blocks.getNumPartitions / 2)
        val adaptive = math.max(1, math.min(16, math.min(math.min(blockRows, blockCols), partitionBound)))
        adaptive
      }
    }

    private def effectiveIterativeMidDimSplits(numMidDimSplits: Int, perf: IterativePerfConfig): Int = {
      val requested = math.max(1, numMidDimSplits)
      if (!perf.adaptiveMidDimSplits || requested > 1) {
        requested
      } else {
        val blockRows = ((matrix.numRows() + matrix.rowsPerBlock - 1) / matrix.rowsPerBlock).toInt
        val blockCols = ((matrix.numCols() + matrix.colsPerBlock - 1) / matrix.colsPerBlock).toInt
        val partitionBound = math.max(1, matrix.blocks.getNumPartitions / 2)
        val adaptiveMax = math.max(1, perf.maxAdaptiveMidDimSplits)
        math.max(1, math.min(adaptiveMax, math.min(math.min(blockRows, blockCols), partitionBound)))
      }
    }

    private def maybeCheckpoint(
      rdd: RDD[((Int, Int), Matrix)],
      useCheckpoints: Boolean,
      useLocalCheckpoint: Boolean): Unit = {
      if (useCheckpoints) {
        rdd.checkpoint()
      } else if (useLocalCheckpoint) {
        rdd.localCheckpoint()
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

      val E = new BlockMatrix(blocks.filter { case ((i, j), _) => i < m && j < m },
        rowsPerBlock, colsPerBlock, splitSize, splitSize).setName("E")
      val F = new BlockMatrix(blocks.filter { case ((i, j), _) => i < m && j >= m }
        .map { case ((i, j), block) => ((i, j - m), block) },
        rowsPerBlock, colsPerBlock, splitSize, res).setName("F")
      val G = new BlockMatrix(blocks.filter { case ((i, j), _) => i >= m && j < m }
        .map { case ((i, j), block) => ((i - m, j), block) },
        rowsPerBlock, colsPerBlock, res, splitSize).setName("G")
      val H = new BlockMatrix(blocks.filter { case ((i, j), _) => i >= m && j >= m }
        .map { case ((i, j), block) => ((i - m, j - m), block) },
        rowsPerBlock, colsPerBlock, res, res).setName("H")

      BlockQuadrants(E, F, G, H)
    }

    private def shiftAndScaleBlocks(
      blocks: RDD[((Int, Int), Matrix)],
      rowOffset: Int = 0,
      colOffset: Int = 0,
      scale: Double = 1.0): RDD[((Int, Int), Matrix)] = {
      if (rowOffset == 0 && colOffset == 0 && scale == 1.0) {
        blocks
      } else {
        blocks.map { case ((i, j), mat) =>
          val shiftedIndex = (i + rowOffset, j + colOffset)
          val adjusted = if (scale == 1.0) mat else scaleDenseCopy(mat, scale)
          (shiftedIndex, adjusted)
        }
      }
    }

    private def invertPartition(
      partition: BlockMatrix,
      partitionBlockSize: Int,
      recurseThresholdInBlocks: Int,
      limit: Int,
      numMidDimSplits: Int,
      useCheckpoints: Boolean,
      depth: Int,
      name: String,
      perf: RecursivePerfConfig): BlockMatrix = {
      val inv = if (partitionBlockSize > recurseThresholdInBlocks) {
        partition.inverse(limit, numMidDimSplits, useCheckpoints, depth = depth + 1, perf)
      } else {
        partition.localInv()
      }
      val named = inv.setName(name)
      persistAndTrack(named, useCheckpoints)
      named
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

    /**
     * Computes the matrix inverse through the SVD method.
     * Should be used for relatively small BlockMatrices.
     *
     * @return Inverted BlockMatrix
     */
    def svdInv(): BlockMatrix = {
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      val X = matrix.toIndexedRowMatrix()
      val n = X.numCols().toInt

      val svd = X.computeSVD(n, computeU = true, rCond = 0)
      require(svd.s.size >= n, "svdInv called on singular matrix." + X.rows.collect().mkString("Array(", ", ", ")") + svd.s.toArray.mkString("Array(", ", ", ")"))

      // Create the inv diagonal matrix from S
      val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
      val U = svd.U
      val V = svd.V

      U.multiply(invS)
        .multiply(V.transpose)
        .toBlockMatrix(colsPerBlock, rowsPerBlock)
        .transpose
    }

    /**
     * Computes the matrix inverse by collecting to the driver and using LU factorization.
     * Uses Breeze dense inverse on the collected local matrix.
     * Should be used for relatively small BlockMatrices (the base case of recursive inversion).
     *
     * @return Inverted BlockMatrix
     */
    def localInv(): BlockMatrix = {
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      val localMat = matrix.toLocalMatrix()
      val n = localMat.numRows
      val invData = luInverse(localMat.toArray, n)
      localDenseToBlockMatrix(invData, n, rowsPerBlock, colsPerBlock)
    }

    /**
     * Return the negative of this [[BlockMatrix]].
     * A.negative() = -A
     *
     * @return BlockMatrix
     */
    def negative(): BlockMatrix = {
      val newblocks = matrix.blocks.map { case ((i, j), mat) =>
        ((i, j), scaleDenseCopy(mat, -1.0))
      }
      new BlockMatrix(newblocks, matrix.rowsPerBlock, matrix.colsPerBlock, matrix.numRows(), matrix.numCols())
    }

    /**
     * Computes the inverse of this square [[BlockMatrix]].
     *
     * @param limit           Size limit of the block partitions that will end the recursion.
     *                        When the block partition is smaller than `limit`,
     *                        the inverse will be computed using the SVD method.
     * @param numMidDimSplits Number of splits to cut on the middle dimension when doing
     *                        multiplication. For example, when multiplying a Matrix `A` of
     *                        size `m x n` with Matrix `B` of size `n x k`, this parameter
     *                        configures the parallelism to use when grouping the matrices. The
     *                        parallelism will increase from `m x k` to `m x k x numMidDimSplits`,
     *                        which in some cases also reduces total shuffled data.
     * @param useCheckpoints  Whether to use checkpointing when applying the algorithm. This has
     *                        performance benefits since the lineage can get very large.
     * @return BlockMatrix
     */
    def inverse(limit: Int, numMidDimSplits: Int, useCheckpoints: Boolean = true, depth: Int = 0): BlockMatrix = {
      inverse(limit, numMidDimSplits, useCheckpoints, depth, RecursivePerfConfig())
    }

    def inverse(limit: Int, numMidDimSplits: Int, useCheckpoints: Boolean, depth: Int, perf: RecursivePerfConfig): BlockMatrix = {
      validateInverseInputs(useCheckpoints)
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      val midSplits = effectiveMidDimSplits(numMidDimSplits, perf)
      val numBlockCols = ((matrix.numCols() + colsPerBlock - 1) / colsPerBlock).toInt
      val m = (numBlockCols + 1) / 2
      val splitSize = math.min(matrix.numRows(), m.toLong * rowsPerBlock)
      val numParts = (matrix.blocks.getNumPartitions).toInt

      trace(perf.trace, s"[perf][recursive] depth=$depth inputParts=${matrix.blocks.getNumPartitions} midSplits=$midSplits")
      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

      val res = matrix.numRows() - splitSize

      val BlockQuadrants(e, f, g, h) = splitQuadrants(m, splitSize, res)

      persistAndTrack(e, useCheckpoints)
      persistAndTrack(f, useCheckpoints)
      persistAndTrack(g, useCheckpoints)
      persistAndTrack(h, useCheckpoints)

      val recurseThresholdInBlocks = math.max(1, limit / colsPerBlock)
      val E_inv = timed(perf.trace, s"[perf][recursive] depth=$depth E inverse") {
        invertPartition(e, m, recurseThresholdInBlocks, limit, midSplits, useCheckpoints, depth, "E_inv", perf)
      }

      val GE_inv = timed(perf.trace, s"[perf][recursive] depth=$depth G*E_inv") {
        g.multiply(E_inv, midSplits).setName("GE_inv")
      }
      val E_invF = timed(perf.trace, s"[perf][recursive] depth=$depth E_inv*F") {
        E_inv.multiply(f, midSplits).setName("E_invF")
      }

      persistAndTrack(GE_inv, useCheckpoints)
      persistAndTrack(E_invF, useCheckpoints)

      val S = timed(perf.trace, s"[perf][recursive] depth=$depth Schur S") {
        h.subtract(g.multiply(E_invF, midSplits)).setName("S")
      }
      persistAndTrack(S, useCheckpoints)

      val S_inv = timed(perf.trace, s"[perf][recursive] depth=$depth S inverse") {
        invertPartition(S, m, recurseThresholdInBlocks, limit, midSplits, useCheckpoints, depth, "S_inv", perf)
      }

      val S_invGE_inv = S_inv.multiply(GE_inv, midSplits).setName("S_invGE_inv")
      val E_invFS_inv = E_invF.multiply(S_inv, midSplits).setName("E_invFS_inv")
      persistAndTrack(S_invGE_inv, useCheckpoints)
      persistAndTrack(E_invFS_inv, useCheckpoints)

      val topLeft = E_inv.add(E_invFS_inv.multiply(GE_inv, midSplits))
      val sc = topLeft.blocks.sparkContext
      val unionedBlocks = sc.union(
        topLeft.blocks,
        shiftAndScaleBlocks(E_invFS_inv.blocks, colOffset = m, scale = -1.0),
        shiftAndScaleBlocks(S_invGE_inv.blocks, rowOffset = m, scale = -1.0),
        shiftAndScaleBlocks(S_inv.blocks, rowOffset = m, colOffset = m))
      val defaultOutputParts = math.max(numParts, math.min(unionedBlocks.getNumPartitions, numParts * 2))
      val outputParts = perf.targetOutputPartitions.getOrElse(defaultOutputParts)
      val all_blocks = maybeCoalesceNoShuffle(unionedBlocks, outputParts, perf.unionCoalesceThreshold)

      val bm = new BlockMatrix(all_blocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
      if (useCheckpoints) {
        bm.blocks.persist(iterativeStorageLevel)
        bm.blocks.checkpoint()
      }
      cachedMatrices.foreach(bm => bm.blocks.unpersist(true))
      bm
    }

    def inverse(): BlockMatrix = {
      inverse(4096, 1)
    }

    def inverse(limit: Int): BlockMatrix = {
      inverse(limit, 1)
    }

    /**
     * Compute the 1-norm (max column sum of absolute values).
     */
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

    /**
     * Compute the infinity-norm (max row sum of absolute values).
     */
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

    /**
     * Compute the squared Frobenius norm (sum of squared elements).
     */
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

    /**
     * Multiply every element by a scalar.
     */
    def scalarMultiply(scalar: Double): BlockMatrix = {
      val newblocks = matrix.blocks.map { case ((i, j), mat) =>
        ((i, j), scaleDenseCopy(mat, scalar))
      }
      new BlockMatrix(newblocks, matrix.rowsPerBlock, matrix.colsPerBlock, matrix.numRows(), matrix.numCols())
    }

    /**
     * Newton-Schulz iterative matrix inversion.
     *
     * Uses the iteration X_{k+1} = X_k * (2I - A * X_k) starting from
     * X_0 = alpha * A^T where alpha = 1 / (||A||_1 * ||A||_inf).
     * Converges quadratically for well-conditioned (e.g. diagonally dominant) matrices.
     *
     * @param maxIter             Maximum number of iterations
     * @param tolerance           Convergence tolerance on ||I - A*X||_F / n
     * @param useCheckpoints      Whether to use checkpointing
     * @param checkpointInterval  How often to checkpoint (every N iterations)
     * @return Inverted BlockMatrix
     */
    def iterativeInverse(maxIter: Int = 30, tolerance: Double = 1e-10,
                         useCheckpoints: Boolean = true, checkpointInterval: Int = 5,
                         numMidDimSplits: Int = 1): BlockMatrix = {
      iterativeInverse(
        maxIter,
        tolerance,
        useCheckpoints,
        checkpointInterval,
        numMidDimSplits,
        IterativePerfConfig(
          persistLevel = iterativeStorageLevel,
          checkpointEvery = checkpointInterval,
          useLocalCheckpoint = false,
          adaptiveMidDimSplits = true,
          maxAdaptiveMidDimSplits = 16,
          largeMatrixCheckpointEvery = 2,
          largeMatrixThreshold = 4000,
          trace = false))
    }

    def iterativeInverse(
      maxIter: Int,
      tolerance: Double,
      useCheckpoints: Boolean,
      checkpointInterval: Int,
      numMidDimSplits: Int,
      perf: IterativePerfConfig): BlockMatrix = {
      require(!useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
        "Checkpointing dir has to be set when useCheckpoints=true!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val storageLevel = perf.persistLevel
      val baseCheckpointEvery = math.max(1, perf.checkpointEvery)
      val effectiveCheckpointEvery =
        if (matrix.numRows() >= perf.largeMatrixThreshold) {
          math.max(1, math.min(baseCheckpointEvery, perf.largeMatrixCheckpointEvery))
        } else {
          baseCheckpointEvery
        }
      val midSplits = effectiveIterativeMidDimSplits(numMidDimSplits, perf)
      if (perf.checkpointEvery != checkpointInterval) {
        trace(perf.trace, s"[perf][iterative] using checkpointEvery=${perf.checkpointEvery} (requested=$checkpointInterval)")
      }

      val shouldPersistInput = matrix.blocks.getStorageLevel == StorageLevel.NONE
      if (shouldPersistInput) {
        matrix.blocks.persist(storageLevel)
      }

      val n = matrix.numRows()
      val norm1 = timed(perf.trace, "[perf][iterative] ||A||_1") { matrix.normOne() }
      val normInfVal = timed(perf.trace, "[perf][iterative] ||A||_inf") { matrix.normInf() }
      val alpha = 1.0 / (norm1 * normInfVal)

      println(s"iterativeInverse: n=$n, ||A||_1=$norm1, ||A||_inf=$normInfVal, alpha=$alpha")

      // X_0 = alpha * A^T
      // Use MEMORY_AND_DISK_SER so Spark can spill to disk under memory pressure
      // instead of OOMing when n is large.
      var X = matrix.transpose.scalarMultiply(alpha)
      X.blocks.persist(storageLevel)
      maybeCheckpoint(X.blocks, useCheckpoints, perf.useLocalCheckpoint)

      val eye = createEye(n, 1.0)
      val twoEye = createEye(n, 2.0)

      var converged = false
      var iter = 0
      while (iter < maxIter && !converged) {
        iter += 1

        // AX = A * X_k — computed once and reused for both the update step and
        // the convergence check, avoiding a redundant second multiply.
        val AX = timed(perf.trace, s"[perf][iterative] iter=$iter AX") { matrix.multiply(X, midSplits) }
        AX.blocks.persist(storageLevel)

        // Check convergence using AX (= A * X_k) that we already have.
        // This checks the current iterate X_k one step before committing to X_{k+1}.
        val residual = eye.subtract(AX)
        val frobSq = residual.frobeniusNormSquared()
        val metric = math.sqrt(frobSq) / n
        println(s"iterativeInverse iter=$iter: ||I - A*X||_F / n = $metric")
        if (metric < tolerance) {
          converged = true
        }

        if (!converged) {
          // X_{k+1} = X_k * (2I - A * X_k)
          val twoI_minus_AX = twoEye.subtract(AX)
          val X_new = timed(perf.trace, s"[perf][iterative] iter=$iter X update") { X.multiply(twoI_minus_AX, midSplits) }

          val oldX = X
          X = X_new
          X.blocks.persist(storageLevel)
          if (iter % effectiveCheckpointEvery == 0) {
            maybeCheckpoint(X.blocks, useCheckpoints, perf.useLocalCheckpoint)
            // Only force materialization when checkpointing to avoid an extra job every iteration.
            X.blocks.count()
          }
          oldX.blocks.unpersist(true)
        }

        AX.blocks.unpersist(true)
      }

      if (!converged) {
        println(s"Warning: iterativeInverse did not converge after $maxIter iterations")
      }

      if (shouldPersistInput) {
        matrix.blocks.unpersist(false)
      }

      X
    }

    /**
     * Creates identity matrix of size n and value in the diagonal
     *
     * @param n     Matrix size
     * @param value Value of the diagonal
     * @return Diagonal BlockMatrix
     */
    private def createEye(n: Long, value: Double = 1.0): BlockMatrix = {
      val key = (n, value, matrix.rowsPerBlock, matrix.colsPerBlock)
      eyeBlockMatrixMap.getOrElseUpdate(key, {
        val sc = matrix.blocks.sparkContext
        val diagonal = sc.range(start = 0, end = n)
          .map {
            i => {
              MatrixEntry(i, i, value)
            }
          }
        val cm = new CoordinateMatrix(diagonal, n, n)
        var bm = cm.toBlockMatrix(matrix.rowsPerBlock, matrix.colsPerBlock)
        bm = bm.setName("eye_" + key)
        bm.blocks.persist(iterativeStorageLevel)
        bm
      })
    }

    def leftPseudoInverse(limit: Int, numMidDimSplits: Int): BlockMatrix = {
      val at = matrix.transpose
      val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
      try {
        val gram = at.multiply(matrix, numMidDimSplits)
        val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
        try {
          gram.inverse(limit, numMidDimSplits).multiply(at, numMidDimSplits)
        } finally {
          if (persistedGram) {
            gram.blocks.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.blocks.unpersist(false)
        }
      }
    }

    def leftPseudoInverse(limit: Int): BlockMatrix = {
      leftPseudoInverse(limit, 1)
    }

    def leftPseudoInverse(): BlockMatrix = {
      val at = matrix.transpose
      val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
      try {
        val gram = at.multiply(matrix)
        val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
        try {
          gram.inverse().multiply(at)
        } finally {
          if (persistedGram) {
            gram.blocks.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.blocks.unpersist(false)
        }
      }
    }

    def rightPseudoInverse(limit: Int, numMidDimSplits: Int): BlockMatrix = {
      val at = matrix.transpose
      val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
      try {
        val gram = matrix.multiply(at, numMidDimSplits)
        val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
        try {
          at.multiply(gram.inverse(limit, numMidDimSplits), numMidDimSplits)
        } finally {
          if (persistedGram) {
            gram.blocks.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.blocks.unpersist(false)
        }
      }
    }

    def rightPseudoInverse(limit: Int): BlockMatrix = {
      rightPseudoInverse(limit, 1)
    }

    def rightPseudoInverse(): BlockMatrix = {
      val at = matrix.transpose
      val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
      try {
        val gram = matrix.multiply(at)
        val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
        try {
          at.multiply(gram.inverse())
        } finally {
          if (persistedGram) {
            gram.blocks.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.blocks.unpersist(false)
        }
      }
    }

    def setName (name: String): BlockMatrix = {
      matrix.blocks.setName(name)
      matrix
    }
    def persist(): BlockMatrix = {
      matrix.blocks.persist()
      matrix
    }
    def checkpoint(): BlockMatrix = {
      matrix.blocks.checkpoint()
      matrix
    }
  }


  implicit class CoordinateMatrixInverse(val matrix: CoordinateMatrix) {

    private val iterativeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
    private val cachedMatrices: ListBuffer[CoordinateMatrix] = mutable.ListBuffer.empty
    private case class CoordinateQuadrants(E: CoordinateMatrix, F: CoordinateMatrix, G: CoordinateMatrix, H: CoordinateMatrix)

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

    private def maybeCheckpoint(
      rdd: RDD[MatrixEntry],
      useCheckpoints: Boolean,
      useLocalCheckpoint: Boolean): Unit = {
      if (useCheckpoints) {
        rdd.checkpoint()
      } else if (useLocalCheckpoint) {
        rdd.localCheckpoint()
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

    private def effectiveIterativeCheckpointEvery(perf: IterativePerfConfig): Int = {
      val base = math.max(1, perf.checkpointEvery)
      if (matrix.numRows() >= perf.largeMatrixThreshold) {
        math.max(1, math.min(base, perf.largeMatrixCheckpointEvery))
      } else {
        base
      }
    }

    private def keyedEntries(entries: RDD[MatrixEntry], scale: Double = 1.0): RDD[((Long, Long), Double)] = {
      entries.map { case MatrixEntry(i, j, v) => ((i, j), v * scale) }
    }

    private def splitQuadrants(entries: RDD[MatrixEntry], m: Int): CoordinateQuadrants = {
      val E = new CoordinateMatrix(entries.filter(x => x.i < m && x.j < m)).setName("E")
      val F = new CoordinateMatrix(entries.filter(x => x.i < m && x.j >= m)
        .map { case MatrixEntry(i, j, v) => MatrixEntry(i, j - m, v) }).setName("F")
      val G = new CoordinateMatrix(entries.filter(x => x.i >= m && x.j < m)
        .map { case MatrixEntry(i, j, v) => MatrixEntry(i - m, j, v) }).setName("G")
      val H = new CoordinateMatrix(entries.filter(x => x.i >= m && x.j >= m)
        .map { case MatrixEntry(i, j, v) => MatrixEntry(i - m, j - m, v) }).setName("H")

      CoordinateQuadrants(E, F, G, H)
    }

    private def shiftAndScaleEntries(
      entries: RDD[MatrixEntry],
      rowOffset: Int = 0,
      colOffset: Int = 0,
      scale: Double = 1.0): RDD[MatrixEntry] = {
      if (rowOffset == 0 && colOffset == 0 && scale == 1.0) {
        entries
      } else {
        entries.map { case MatrixEntry(i, j, v) =>
          MatrixEntry(i + rowOffset, j + colOffset, v * scale)
        }
      }
    }

    private def invertPartition(
      partition: CoordinateMatrix,
      partitionSize: Int,
      limit: Int,
      useCheckpoints: Boolean,
      depth: Int,
      name: String,
      perf: RecursivePerfConfig): CoordinateMatrix = {
      val inv = if (partitionSize > limit) {
        partition.inverse(limit, useCheckpoints, depth = depth + 1, perf)
      } else {
        partition.localInv()
      }
      val named = inv.setName(name)
      persistAndTrack(named, useCheckpoints)
      named
    }

    /**
     * Computes the matrix inverse through the SVD method.
     * Should be used for relatively small BlockMatrices.
     *
     * @return Inverted BlockMatrix
     */
    def svdInv(): CoordinateMatrix = {
      val X = matrix.toIndexedRowMatrix()
      val n = X.numCols().toInt

      val svd = X.computeSVD(n, computeU = true, rCond = 0)
      require(svd.s.size >= n, "svdInv called on singular matrix." + X.rows.collect().mkString("Array(", ", ", ")") + svd.s.toArray.mkString("Array(", ", ", ")"))

      // Create the inv diagonal matrix from S
      val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
      val U = svd.U
      val V = svd.V

      U.multiply(invS)
        .multiply(V.transpose)
        .toCoordinateMatrix()
        .transpose()
    }

    /**
     * Computes the matrix inverse by collecting to the driver and using LU factorization.
     * Uses Breeze dense inverse on the collected local matrix.
     * Should be used for relatively small CoordinateMatrices (the base case of recursive inversion).
     *
     * @return Inverted CoordinateMatrix
     */
    def localInv(): CoordinateMatrix = {
      val localMat = matrix.toBlockMatrix().toLocalMatrix()
      val n = localMat.numRows
      val invData = luInverse(localMat.toArray, n)
      val sc = matrix.entries.sparkContext
      val entries = sc.parallelize(
        for (i <- 0 until n; j <- 0 until n)
          yield MatrixEntry(i.toLong, j.toLong, invData(i + j * n))
      )
      new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
    }

    /**
     * Return the negative of this [[CoordinateMatrix]].
     * A.negative() = -A
     *
     * @return CoordinateMatrix
     */
    def negative(): CoordinateMatrix = {
      val newblocks = matrix.entries.map { case MatrixEntry(i, j, v) => {
        MatrixEntry(i, j, -v)
      }
      }
      new CoordinateMatrix(newblocks, matrix.numRows(), matrix.numCols())
    }

    /**
     * Computes the inverse of this square [[BlockMatrix]].
     *
     * @param limit           Size limit of the block partitions that will end the recursion.
     *                        When the block partition is smaller than `limit`,
     *                        the inverse will be computed using the SVD method.
     * @param useCheckpoints  Whether to use checkpointing when applying the algorithm. This has
     *                        performance benefits since the lineage can get very large.
     * @return BlockMatrix
     */
    def inverse(limit: Int, useCheckpoints: Boolean = true, depth: Int = 0): CoordinateMatrix = {
      inverse(limit, useCheckpoints, depth, RecursivePerfConfig())
    }

    def inverse(limit: Int, useCheckpoints: Boolean, depth: Int, perf: RecursivePerfConfig): CoordinateMatrix = {
      validateInverseInputs(useCheckpoints)
      val numCols = matrix.numCols()
      val m = ((numCols + 1) / 2).toInt
      val entries: RDD[MatrixEntry] = matrix.entries
      val numParts = (matrix.entries.getNumPartitions).toInt

      trace(perf.trace, s"[perf][recursive-coo] depth=$depth inputParts=${matrix.entries.getNumPartitions}")
      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

      val CoordinateQuadrants(e, f, g, h) = splitQuadrants(entries, m)

      persistAndTrack(e, useCheckpoints)
      persistAndTrack(f, useCheckpoints)
      persistAndTrack(g, useCheckpoints)
      persistAndTrack(h, useCheckpoints)

      val E_inv = timed(perf.trace, s"[perf][recursive-coo] depth=$depth E inverse") {
        invertPartition(e, m, limit, useCheckpoints, depth, "E_inv", perf)
      }

      val GE_inv = g.multiply(E_inv).setName("GE_inv")
      val E_invF = E_inv.multiply(f).setName("E_invF")

      persistAndTrack(GE_inv, useCheckpoints)
      persistAndTrack(E_invF, useCheckpoints)

      val S = h.subtract(g.multiply(E_invF)).setName("S")
      persistAndTrack(S, useCheckpoints)

      val S_inv = timed(perf.trace, s"[perf][recursive-coo] depth=$depth S inverse") {
        invertPartition(S, m, limit, useCheckpoints, depth, "S_inv", perf)
      }

      val S_invGE_inv = S_inv.multiply(GE_inv).setName("S_invGE_inv")
      val E_invFS_inv = E_invF.multiply(S_inv).setName("E_invFS_inv")
      persistAndTrack(S_invGE_inv, useCheckpoints)
      persistAndTrack(E_invFS_inv, useCheckpoints)

      val topLeft = E_inv.add(E_invFS_inv.multiply(GE_inv))
      val sc = topLeft.entries.sparkContext
      val unionedEntries = sc.union(
        topLeft.entries,
        shiftAndScaleEntries(E_invFS_inv.entries, colOffset = m, scale = -1.0),
        shiftAndScaleEntries(S_invGE_inv.entries, rowOffset = m, scale = -1.0),
        shiftAndScaleEntries(S_inv.entries, rowOffset = m, colOffset = m))
      val defaultOutputParts = math.max(numParts, math.min(unionedEntries.getNumPartitions, numParts * 2))
      val outputParts = perf.targetOutputPartitions.getOrElse(defaultOutputParts)
      val all_blocks = maybeCoalesceNoShuffle(unionedEntries, outputParts, perf.unionCoalesceThreshold)

      val cm = new CoordinateMatrix(all_blocks, matrix.numRows(), matrix.numCols())
      if (useCheckpoints) {
        cm.entries.persist(iterativeStorageLevel)
        cm.entries.checkpoint()
      }
      cachedMatrices.foreach(cm => cm.entries.unpersist(true))
      cm
    }

    def inverse(): CoordinateMatrix = {
      inverse(4096)
    }

    def inverse(limit: Int): CoordinateMatrix = {
      inverse(limit, depth = 0)
    }

    /**
     * Compute the 1-norm (max column sum of absolute values).
     */
    def normOne(): Double = {
      matrix.entries
        .map { case MatrixEntry(_, j, v) => (j, math.abs(v)) }
        .reduceByKey(_ + _)
        .values
        .max()
    }

    /**
     * Compute the infinity-norm (max row sum of absolute values).
     */
    def normInf(): Double = {
      matrix.entries
        .map { case MatrixEntry(i, _, v) => (i, math.abs(v)) }
        .reduceByKey(_ + _)
        .values
        .max()
    }

    /**
     * Compute the squared Frobenius norm (sum of squared elements).
     */
    def frobeniusNormSquared(): Double = {
      matrix.entries.map { case MatrixEntry(_, _, v) => v * v }.sum()
    }

    /**
     * Multiply every element by a scalar.
     */
    def scalarMultiply(scalar: Double): CoordinateMatrix = {
      val newEntries = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(i, j, v * scalar) }
      new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
    }

    /**
     * Newton-Schulz iterative matrix inversion.
     *
     * Uses the iteration X_{k+1} = X_k * (2I - A * X_k) starting from
     * X_0 = alpha * A^T where alpha = 1 / (||A||_1 * ||A||_inf).
     * Converges quadratically for well-conditioned (e.g. diagonally dominant) matrices.
     *
     * @param maxIter             Maximum number of iterations
     * @param tolerance           Convergence tolerance on ||I - A*X||_F / n
     * @param useCheckpoints      Whether to use checkpointing
     * @param checkpointInterval  How often to checkpoint (every N iterations)
     * @return Inverted CoordinateMatrix
     */
    def iterativeInverse(maxIter: Int = 30, tolerance: Double = 1e-10,
                         useCheckpoints: Boolean = true, checkpointInterval: Int = 5): CoordinateMatrix = {
      iterativeInverse(
        maxIter,
        tolerance,
        useCheckpoints,
        checkpointInterval,
        IterativePerfConfig(
          persistLevel = iterativeStorageLevel,
          checkpointEvery = checkpointInterval,
          useLocalCheckpoint = false,
          trace = false))
    }

    def iterativeInverse(
      maxIter: Int,
      tolerance: Double,
      useCheckpoints: Boolean,
      checkpointInterval: Int,
      perf: IterativePerfConfig): CoordinateMatrix = {
      require(!useCheckpoints || matrix.entries.sparkContext.getCheckpointDir.isDefined,
        "Checkpointing dir has to be set when useCheckpoints=true!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val storageLevel = perf.persistLevel
      val effectiveCheckpointEvery = effectiveIterativeCheckpointEvery(perf)
      if (perf.checkpointEvery != checkpointInterval) {
        trace(perf.trace, s"[perf][iterative-coo] using checkpointEvery=${perf.checkpointEvery} (requested=$checkpointInterval)")
      }

      val shouldPersistInput = matrix.entries.getStorageLevel == StorageLevel.NONE
      if (shouldPersistInput) {
        matrix.entries.persist(storageLevel)
      }

      val n = matrix.numRows()
      val norm1 = timed(perf.trace, "[perf][iterative-coo] ||A||_1") { matrix.normOne() }
      val normInfVal = timed(perf.trace, "[perf][iterative-coo] ||A||_inf") { matrix.normInf() }
      val alpha = 1.0 / (norm1 * normInfVal)

      println(s"iterativeInverse: n=$n, ||A||_1=$norm1, ||A||_inf=$normInfVal, alpha=$alpha")

      // X_0 = alpha * A^T
      var X = matrix.transpose().scalarMultiply(alpha)
      X.entries.persist(storageLevel)
      maybeCheckpoint(X.entries, useCheckpoints, perf.useLocalCheckpoint)

      val eye = createEye(n, 1.0)
      val twoEye = createEye(n, 2.0)

      var converged = false
      var iter = 0
      while (iter < maxIter && !converged) {
        iter += 1

        // AX = A * X_k — computed once, reused for both the update and the convergence check.
        val AX = timed(perf.trace, s"[perf][iterative-coo] iter=$iter AX") { matrix.multiply(X) }
        AX.entries.persist(storageLevel)

        val residual = eye.subtract(AX)
        val frobSq = residual.frobeniusNormSquared()
        val metric = math.sqrt(frobSq) / n
        println(s"iterativeInverse iter=$iter: ||I - A*X||_F / n = $metric")
        if (metric < tolerance) {
          converged = true
        }

        if (!converged) {
          // X_{k+1} = X_k * (2I - A * X_k)
          val twoI_minus_AX = twoEye.subtract(AX)
          val X_new = timed(perf.trace, s"[perf][iterative-coo] iter=$iter X update") { X.multiply(twoI_minus_AX) }

          val oldX = X
          X = X_new
          X.entries.persist(storageLevel)
          if (iter % effectiveCheckpointEvery == 0) {
            maybeCheckpoint(X.entries, useCheckpoints, perf.useLocalCheckpoint)
            // Only force materialization when checkpointing to avoid an extra job every iteration.
            X.entries.count()
          }
          oldX.entries.unpersist(true)
        }

        AX.entries.unpersist(true)
      }

      if (!converged) {
        println(s"Warning: iterativeInverse did not converge after $maxIter iterations")
      }

      if (shouldPersistInput) {
        matrix.entries.unpersist(false)
      }

      X
    }

    /**
     * Creates identity matrix of size n and value in the diagonal
     *
     * @param n     Matrix size
     * @param value Value of the diagonal
     * @return Diagonal BlockMatrix
     */
    private def createEye(n: Long, value: Double = 1.0): CoordinateMatrix = {
      eyeCoordinateMatrixMap.getOrElseUpdate((n, value), {
        val sc = matrix.entries.sparkContext
        val diagonal = sc.range(start = 0, end = n)
          .map {
            i => {
              MatrixEntry(i, i, value)
            }
          }
        var cm = new CoordinateMatrix(diagonal, n, n)
        cm = cm.setName("eye_" + (n, value))
        cm.entries.persist(iterativeStorageLevel)
        cm
      })
    }

    def multiply(other: CoordinateMatrix): CoordinateMatrix = {
      val partitioner = new HashPartitioner(estimatedPartitionCountFor(other))
      val leftByMid = matrix.entries
        .map({ case MatrixEntry(i, j, v) => (j, (i, v)) })
        .partitionBy(partitioner)
      val rightByMid = other.entries
        .map({ case MatrixEntry(j, k, w) => (j, (k, w)) })
        .partitionBy(partitioner)
      val productEntries = leftByMid
        .join(rightByMid)
        .map({ case (_, ((i, v), (k, w))) => ((i, k), (v * w)) })
        .reduceByKey(partitioner, _ + _)
        .filter { case (_, sum) => sum != 0.0 }
        .map({ case ((i, k), sum) => MatrixEntry(i, k, sum) })
      new CoordinateMatrix(productEntries, matrix.numRows(), other.numCols())
    }

    def add(other: CoordinateMatrix): CoordinateMatrix = {
      val partitioner = new HashPartitioner(estimatedPartitionCountFor(other))
      val entries = keyedEntries(matrix.entries)
        .partitionBy(partitioner)
        .union(keyedEntries(other.entries).partitionBy(partitioner))
        .reduceByKey(partitioner, _ + _)
        .filter { case (_, sum) => sum != 0.0 }
        .map { case ((i, j), v) => MatrixEntry(i, j, v) }
      new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
    }

    def subtract(other: CoordinateMatrix): CoordinateMatrix = {
      val partitioner = new HashPartitioner(estimatedPartitionCountFor(other))
      val entries = keyedEntries(matrix.entries)
        .partitionBy(partitioner)
        .union(keyedEntries(other.entries, scale = -1.0).partitionBy(partitioner))
        .reduceByKey(partitioner, _ + _)
        .filter { case (_, sum) => sum != 0.0 }
        .map { case ((i, j), v) => MatrixEntry(i, j, v) }
      new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
    }

    def partitionBy(partitioner: Partitioner): CoordinateMatrix = {
      val newEntries = matrix.entries.keyBy(x => (x.i, x.j))
        .partitionBy(partitioner)
        .values
      new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
    }

    def transpose(): CoordinateMatrix = {
      val t = matrix.entries.map { case MatrixEntry(i, j, v) => MatrixEntry(j, i, v) }
      new CoordinateMatrix(t, matrix.numCols(), matrix.numRows())
    }

    def leftPseudoInverse(limit: Int): CoordinateMatrix = {
      val at = matrix.transpose()
      val persistedAt = persistIfNeeded(at.entries, iterativeStorageLevel)
      try {
        val gram = at.multiply(matrix)
        val persistedGram = persistIfNeeded(gram.entries, iterativeStorageLevel)
        try {
          gram.inverse(limit).multiply(at)
        } finally {
          if (persistedGram) {
            gram.entries.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.entries.unpersist(false)
        }
      }
    }

    def leftPseudoInverse(): CoordinateMatrix = {
      val at = matrix.transpose()
      val persistedAt = persistIfNeeded(at.entries, iterativeStorageLevel)
      try {
        val gram = at.multiply(matrix)
        val persistedGram = persistIfNeeded(gram.entries, iterativeStorageLevel)
        try {
          gram.inverse().multiply(at)
        } finally {
          if (persistedGram) {
            gram.entries.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.entries.unpersist(false)
        }
      }
    }

    def rightPseudoInverse(limit: Int): CoordinateMatrix = {
      val at = matrix.transpose()
      val persistedAt = persistIfNeeded(at.entries, iterativeStorageLevel)
      try {
        val gram = matrix.multiply(at)
        val persistedGram = persistIfNeeded(gram.entries, iterativeStorageLevel)
        try {
          at.multiply(gram.inverse(limit))
        } finally {
          if (persistedGram) {
            gram.entries.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.entries.unpersist(false)
        }
      }
    }

    def rightPseudoInverse(): CoordinateMatrix = {
      val at = matrix.transpose()
      val persistedAt = persistIfNeeded(at.entries, iterativeStorageLevel)
      try {
        val gram = matrix.multiply(at)
        val persistedGram = persistIfNeeded(gram.entries, iterativeStorageLevel)
        try {
          at.multiply(gram.inverse())
        } finally {
          if (persistedGram) {
            gram.entries.unpersist(false)
          }
        }
      } finally {
        if (persistedAt) {
          at.entries.unpersist(false)
        }
      }
    }

    def setName(name: String): CoordinateMatrix = {
      matrix.entries.setName(name)
      matrix
    }
    def cache(): CoordinateMatrix = {
      matrix.entries.cache()
      matrix
    }
    def persist(): CoordinateMatrix = {
      matrix.entries.persist()
      matrix
    }
    def persist(storageLevel: StorageLevel): CoordinateMatrix = {
      matrix.entries.persist(storageLevel)
      matrix
    }
    def checkpoint(): CoordinateMatrix = {
      matrix.entries.checkpoint()
      matrix
    }
  }
}
