import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object Inverse {
  private val eyeBlockMatrixMap: mutable.Map[(Long, Double, Int, Int), BlockMatrix] = mutable.Map[(Long, Double, Int, Int), BlockMatrix]()
  private val eyeCoordinateMatrixMap: mutable.Map[(Long, Double), CoordinateMatrix] = mutable.Map[(Long, Double), CoordinateMatrix]()

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
   * Invert an n×n matrix stored in column-major order using LU factorization with partial pivoting.
   * Pure JVM — no native BLAS/LAPACK dependency.
   *
   * @param data Column-major array of length n*n (not modified)
   * @param n    Matrix dimension
   * @return Column-major array of length n*n containing the inverse
   */
  private[Inverse] def luInverse(data: Array[Double], n: Int): Array[Double] = {
    val a = data.clone()
    val inv = Array.ofDim[Double](n * n)

    var i = 0
    while (i < n) {
      inv(i + i * n) = 1.0
      i += 1
    }

    var k = 0
    while (k < n) {
      var maxRow = k
      var maxVal = math.abs(a(k + k * n))
      i = k + 1
      while (i < n) {
        val v = math.abs(a(i + k * n))
        if (v > maxVal) {
          maxVal = v
          maxRow = i
        }
        i += 1
      }

      if (maxRow != k) {
        var j = 0
        while (j < n) {
          val kIdx = k + j * n
          val maxIdx = maxRow + j * n
          val tmpA = a(kIdx)
          a(kIdx) = a(maxIdx)
          a(maxIdx) = tmpA

          val tmpInv = inv(kIdx)
          inv(kIdx) = inv(maxIdx)
          inv(maxIdx) = tmpInv
          j += 1
        }
      }

      val diag = a(k + k * n)
      require(math.abs(diag) > 1e-300, s"luInverse: singular matrix (zero pivot at column $k)")

      i = k + 1
      while (i < n) {
        val ik = i + k * n
        val factor = a(ik) / diag
        a(ik) = factor

        var j = k + 1
        while (j < n) {
          a(i + j * n) -= factor * a(k + j * n)
          j += 1
        }

        j = 0
        while (j < n) {
          inv(i + j * n) -= factor * inv(k + j * n)
          j += 1
        }
        i += 1
      }

      k += 1
    }

    var j = 0
    while (j < n) {
      k = n - 1
      while (k >= 0) {
        val kk = k + k * n
        val kj = k + j * n
        inv(kj) /= a(kk)
        val xk = inv(kj)

        i = 0
        while (i < k) {
          inv(i + j * n) -= a(i + k * n) * xk
          i += 1
        }

        k -= 1
      }
      j += 1
    }

    inv
  }

  implicit class BlockMatrixInverse(val matrix: BlockMatrix) {

    private val iterativeStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
    private val cachedMatrices: ListBuffer[BlockMatrix] = mutable.ListBuffer.empty

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
     * Uses a pure-JVM Gauss-Jordan implementation — no native BLAS/LAPACK dependency.
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

      require(!useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined, "Checkpointing dir has to be set when useCheckpoints=true!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      require(colsPerBlock == rowsPerBlock, "Sub-matrices has to be square!")
      val numBlockCols = ((matrix.numCols() + colsPerBlock - 1) / colsPerBlock).toInt
      val m = (numBlockCols + 1) / 2
      val splitSize = math.min(matrix.numRows(), m.toLong * rowsPerBlock)
      val blocks = matrix.blocks
      val numParts = (matrix.blocks.getNumPartitions).toInt

      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

      val res = matrix.numRows() - splitSize

      // split into block partitions and readjust the block indices
      val E = new BlockMatrix(blocks.filter { case ((i, j), _) => i < m && j < m }, rowsPerBlock, colsPerBlock, splitSize, splitSize)
        .setName("E")
      val F = new BlockMatrix(blocks.filter { case ((i, j), _) => i < m && j >= m }
        .map { case ((i, j), matrix) => ((i, j - m), matrix) }, rowsPerBlock, colsPerBlock, splitSize, res)
        .setName("F")
      val G = new BlockMatrix(blocks.filter { case ((i, j), _) => i >= m && j < m }
        .map { case ((i, j), matrix) => ((i - m, j), matrix) }, rowsPerBlock, colsPerBlock, res, splitSize)
        .setName("G")
      val H = new BlockMatrix(blocks.filter { case ((i, j), _) => i >= m && j >= m }
        .map { case ((i, j), matrix) => ((i - m, j - m), matrix) }, rowsPerBlock, colsPerBlock, res, res)
        .setName("H")

      persistAndTrack(E, useCheckpoints)
      persistAndTrack(F, useCheckpoints)
      persistAndTrack(G, useCheckpoints)
      persistAndTrack(H, useCheckpoints)

      val recurseThresholdInBlocks = math.max(1, limit / colsPerBlock)
      val E_inv = if (m > recurseThresholdInBlocks) {
        E.inverse(limit, numMidDimSplits, useCheckpoints, depth = depth + 1)
      } else {
        E.localInv()
      }.setName("E_inv")

      persistAndTrack(E_inv, useCheckpoints)

      val GE_inv = G.multiply(E_inv, numMidDimSplits).setName("GE_inv")
      val E_invF = E_inv.multiply(F, numMidDimSplits).setName("E_invF")

      persistAndTrack(GE_inv, useCheckpoints)
      persistAndTrack(E_invF, useCheckpoints)

      val S = H.subtract(G.multiply(E_invF, numMidDimSplits)).setName("S")
      persistAndTrack(S, useCheckpoints)

      val S_inv = if (m > recurseThresholdInBlocks) {
        S.inverse(limit, numMidDimSplits, useCheckpoints, depth = depth + 1)
      } else {
        S.localInv()
      }.setName("S_inv")

      persistAndTrack(S_inv, useCheckpoints)

      val S_invGE_inv = S_inv.multiply(GE_inv, numMidDimSplits).setName("S_invGE_inv")
      val E_invFS_inv = E_invF.multiply(S_inv, numMidDimSplits).setName("E_invFS_inv")
      persistAndTrack(S_invGE_inv, useCheckpoints)
      persistAndTrack(E_invFS_inv, useCheckpoints)

      val top_left = E_inv.add(E_invFS_inv.multiply(GE_inv, numMidDimSplits)).blocks

      // Readjust the block indices
      val top_right = E_invFS_inv.blocks.map {
        case ((i, j), mat) => ((i, j + m), scaleDenseCopy(mat, -1.0))
      }
      val bottom_left = S_invGE_inv.blocks.map {
        case ((i, j), mat) => ((i + m, j), scaleDenseCopy(mat, -1.0))
      }
      val bottom_right = S_inv.blocks.map {
        case ((i, j), mat) => ((i + m, j + m), mat)
      }
      val sc = top_left.sparkContext
      val all_blocks = sc.union(top_left, top_right, bottom_left, bottom_right)
        .coalesce(numParts)

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
      require(!useCheckpoints || matrix.blocks.sparkContext.getCheckpointDir.isDefined,
        "Checkpointing dir has to be set when useCheckpoints=true!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")

      val shouldPersistInput = matrix.blocks.getStorageLevel == StorageLevel.NONE
      if (shouldPersistInput) {
        matrix.blocks.persist(iterativeStorageLevel)
      }

      val n = matrix.numRows()
      val norm1 = matrix.normOne()
      val normInfVal = matrix.normInf()
      val alpha = 1.0 / (norm1 * normInfVal)

      println(s"iterativeInverse: n=$n, ||A||_1=$norm1, ||A||_inf=$normInfVal, alpha=$alpha")

      // X_0 = alpha * A^T
      // Use MEMORY_AND_DISK_SER so Spark can spill to disk under memory pressure
      // instead of OOMing when n is large.
      var X = matrix.transpose.scalarMultiply(alpha)
      X.blocks.persist(iterativeStorageLevel)
      if (useCheckpoints) X.checkpoint()

      val eye = createEye(n, 1.0)
      val twoEye = createEye(n, 2.0)

      var converged = false
      var iter = 0
      while (iter < maxIter && !converged) {
        iter += 1

        // AX = A * X_k — computed once and reused for both the update step and
        // the convergence check, avoiding a redundant second multiply.
        val AX = matrix.multiply(X, numMidDimSplits)
        AX.blocks.persist(iterativeStorageLevel)

        // Check convergence using AX (= A * X_k) that we already have.
        // This checks the current iterate X_k one step before committing to X_{k+1}.
        val frobSq = eye.subtract(AX).frobeniusNormSquared()
        val metric = math.sqrt(frobSq) / n
        println(s"iterativeInverse iter=$iter: ||I - A*X||_F / n = $metric")
        if (metric < tolerance) {
          converged = true
        }

        if (!converged) {
          // X_{k+1} = X_k * (2I - A * X_k)
          val twoI_minus_AX = twoEye.subtract(AX)
          val X_new = X.multiply(twoI_minus_AX, numMidDimSplits)

          val oldX = X
          X = X_new
          X.blocks.persist(iterativeStorageLevel)
          if (useCheckpoints && iter % checkpointInterval == 0) X.checkpoint()
          X.blocks.count()
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
      val eyeMaybe = eyeBlockMatrixMap.get(key)
      if (eyeMaybe.isEmpty) {
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
        eyeBlockMatrixMap.put(key, bm)
        bm
      } else {
        eyeMaybe.get
      }
    }

    def leftPseudoInverse(limit: Int, numMidDimSplits: Int): BlockMatrix = {
      matrix.transpose.multiply(matrix, numMidDimSplits).inverse(limit, numMidDimSplits).multiply(matrix.transpose, numMidDimSplits)
    }

    def leftPseudoInverse(limit: Int): BlockMatrix = {
      leftPseudoInverse(limit, 1)
    }

    def leftPseudoInverse(): BlockMatrix = {
      matrix.transpose.multiply(matrix).inverse().multiply(matrix.transpose)
    }

    def rightPseudoInverse(limit: Int, numMidDimSplits: Int): BlockMatrix = {
      matrix.transpose.multiply(matrix.multiply(matrix.transpose, numMidDimSplits).inverse(limit, numMidDimSplits), numMidDimSplits)
    }

    def rightPseudoInverse(limit: Int): BlockMatrix = {
      rightPseudoInverse(limit, 1)
    }

    def rightPseudoInverse(): BlockMatrix = {
      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse())
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

      require(!useCheckpoints || matrix.entries.sparkContext.getCheckpointDir.isDefined, "Checkpointing dir has to be set when useCheckpoints=true!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val numCols = matrix.numCols()
      val m = ((numCols + 1) / 2).toInt
      val entries: RDD[MatrixEntry] = matrix.entries
      val numParts = (matrix.entries.getNumPartitions).toInt

      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

      // split into block partitions and readjust the block indices
      val E = new CoordinateMatrix(entries.filter(x => x.i < m && x.j < m))
        .setName("E")
      val F = new CoordinateMatrix(entries.filter(x => x.i < m && x.j >= m)
        .map { case MatrixEntry(i, j, matrix) => MatrixEntry(i, j - m, matrix) })
        .setName("F")
      val G = new CoordinateMatrix(entries.filter(x => x.i >= m && x.j < m)
        .map { case MatrixEntry(i, j, matrix) => MatrixEntry(i - m, j, matrix) })
        .setName("G")
      val H = new CoordinateMatrix(entries.filter(x => x.i >= m && x.j >= m)
        .map { case MatrixEntry(i, j, matrix) => MatrixEntry(i - m, j - m, matrix) })
        .setName("H")

      persistAndTrack(E, useCheckpoints)
      persistAndTrack(F, useCheckpoints)
      persistAndTrack(G, useCheckpoints)
      persistAndTrack(H, useCheckpoints)

      val E_inv = if (m > limit) {
        E.inverse(limit, useCheckpoints, depth = depth + 1)
      } else {
        E.localInv()
      }.setName("E_inv")

      persistAndTrack(E_inv, useCheckpoints)

      val GE_inv = G.multiply(E_inv).setName("GE_inv")
      val E_invF = E_inv.multiply(F).setName("E_invF")

      persistAndTrack(GE_inv, useCheckpoints)
      persistAndTrack(E_invF, useCheckpoints)

      val S = H.subtract(G.multiply(E_invF)).setName("S")
      persistAndTrack(S, useCheckpoints)

      val S_inv = if (m > limit) {
        S.inverse(limit, useCheckpoints, depth = depth + 1)
      } else {
        S.localInv()
      }.setName("S_inv")

      persistAndTrack(S_inv, useCheckpoints)

      val S_invGE_inv = S_inv.multiply(GE_inv).setName("S_invGE_inv")
      val E_invFS_inv = E_invF.multiply(S_inv).setName("E_invFS_inv")
      persistAndTrack(S_invGE_inv, useCheckpoints)
      persistAndTrack(E_invFS_inv, useCheckpoints)

      val top_left = E_inv.add(E_invFS_inv.multiply(GE_inv)).entries

      // Readjust the block indices
      val top_right = E_invFS_inv.entries.map {
        case MatrixEntry(i, j, v) => MatrixEntry(i, j + m, -v)
      }
      val bottom_left = S_invGE_inv.entries.map {
        case MatrixEntry(i, j, v) => MatrixEntry(i + m, j, -v)
      }
      val bottom_right = S_inv.entries.map {
        case MatrixEntry(i, j, v) => MatrixEntry(i + m, j + m, v)
      }
      val sc = top_left.sparkContext
      val all_blocks = sc.union(top_left, top_right, bottom_left, bottom_right)
        .coalesce(numParts)

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
      require(!useCheckpoints || matrix.entries.sparkContext.getCheckpointDir.isDefined,
        "Checkpointing dir has to be set when useCheckpoints=true!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")

      val shouldPersistInput = matrix.entries.getStorageLevel == StorageLevel.NONE
      if (shouldPersistInput) {
        matrix.entries.persist(iterativeStorageLevel)
      }

      val n = matrix.numRows()
      val norm1 = matrix.normOne()
      val normInfVal = matrix.normInf()
      val alpha = 1.0 / (norm1 * normInfVal)

      println(s"iterativeInverse: n=$n, ||A||_1=$norm1, ||A||_inf=$normInfVal, alpha=$alpha")

      // X_0 = alpha * A^T
      var X = matrix.transpose().scalarMultiply(alpha)
      X.entries.persist(iterativeStorageLevel)
      if (useCheckpoints) X.checkpoint()

      val eye = createEye(n, 1.0)
      val twoEye = createEye(n, 2.0)

      var converged = false
      var iter = 0
      while (iter < maxIter && !converged) {
        iter += 1

        // AX = A * X_k — computed once, reused for both the update and the convergence check.
        val AX = matrix.multiply(X)
        AX.entries.persist(iterativeStorageLevel)

        val frobSq = eye.subtract(AX).frobeniusNormSquared()
        val metric = math.sqrt(frobSq) / n
        println(s"iterativeInverse iter=$iter: ||I - A*X||_F / n = $metric")
        if (metric < tolerance) {
          converged = true
        }

        if (!converged) {
          // X_{k+1} = X_k * (2I - A * X_k)
          val twoI_minus_AX = twoEye.subtract(AX)
          val X_new = X.multiply(twoI_minus_AX)

          val oldX = X
          X = X_new
          X.entries.persist(iterativeStorageLevel)
          if (useCheckpoints && iter % checkpointInterval == 0) X.checkpoint()
          X.entries.count()
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
      val eyeMaybe = eyeCoordinateMatrixMap.get((n, value))
      if (eyeMaybe.isEmpty) {
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
        eyeCoordinateMatrixMap.put((n, value), cm)
        cm
      } else {
        eyeMaybe.get
      }
    }

    def multiply(other: CoordinateMatrix): CoordinateMatrix = {

      val M_ = matrix.entries
        .map({ case MatrixEntry(i, j, v) => (j, (i, v)) })
      val N_ = other.entries
        .map({ case MatrixEntry(j, k, w) => (j, (k, w)) })
      val productEntries = M_
        .join(N_)
        .map({ case (_, ((i, v), (k, w))) => ((i, k), (v * w)) })
        .reduceByKey(_ + _)
        .map({ case ((i, k), sum) => MatrixEntry(i, k, sum) })
      new CoordinateMatrix(productEntries)
    }

    def add(other: CoordinateMatrix): CoordinateMatrix = {
      val entries = matrix.entries.map { case MatrixEntry(i, j, v) => ((i, j), v) }
        .union(other.entries.map { case MatrixEntry(i, j, v) => ((i, j), v) })
        .reduceByKey(_ + _)
        .map { case ((i, j), v) => MatrixEntry(i, j, v) }
      new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
    }

    def subtract(other: CoordinateMatrix): CoordinateMatrix = {
      val entries = matrix.entries.map { case MatrixEntry(i, j, v) => ((i, j), v) }
        .union(other.entries.map { case MatrixEntry(i, j, v) => ((i, j), -v) })
        .reduceByKey(_ + _)
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
      at.multiply(matrix).inverse(limit).multiply(at)
    }

    def leftPseudoInverse(): CoordinateMatrix = {
      val at = matrix.transpose()
      at.multiply(matrix).inverse().multiply(at)
    }

    def rightPseudoInverse(limit: Int): CoordinateMatrix = {
      val at = matrix.transpose()
      at.multiply(matrix.multiply(at).inverse(limit))
    }

    def rightPseudoInverse(): CoordinateMatrix = {
      val at = matrix.transpose()
      at.multiply(matrix.multiply(at).inverse())
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
