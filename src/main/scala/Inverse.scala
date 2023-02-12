import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object Inverse {
  private val eyeMatrixMap: mutable.Map[(Long, Double), BlockMatrix] = mutable.Map[(Long, Double), BlockMatrix]()

  implicit class BlockMatrixInverse(val matrix: BlockMatrix) {

    private val cachedMatrices: ListBuffer[BlockMatrix] = mutable.ListBuffer.empty
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
     * Return the negative of this [[BlockMatrix]].
     * A.negative() = -A
     *
     * @return BlockMatrix
     */
    def negative(): BlockMatrix = {
      createEye(matrix.numRows(), -1.0).multiply(matrix)
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
    def inverse(limit: Int, numMidDimSplits: Int, useCheckpoints: Boolean = true): BlockMatrix = {

      require(matrix.blocks.sparkContext.getCheckpointDir.isDefined || useCheckpoints, "Checkpointing dir has to be set!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      require(colsPerBlock == rowsPerBlock, "Sub-matrices has to be square!")
      val m = math.ceil(matrix.numCols() / colsPerBlock / 2).toInt
      val m_bc = matrix.blocks.sparkContext.broadcast(m)
      val blocks = matrix.blocks
      val numParts = (matrix.blocks.getNumPartitions).toInt

      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols())

      val res = matrix.numRows() - colsPerBlock * m

      // split into block partitions and readjust the block indices
      val E = new BlockMatrix(blocks.filter(x => x._1._1 < m_bc.value & x._1._2 < m_bc.value), rowsPerBlock, colsPerBlock, rowsPerBlock * m, colsPerBlock * m)
        .setName("E")
      val F = new BlockMatrix(blocks.filter(x => x._1._1 < m_bc.value & x._1._2 >= m_bc.value)
        .map { case ((i, j), matrix) => ((i, j - m_bc.value), matrix) }, rowsPerBlock, colsPerBlock, rowsPerBlock * m, res)
        .setName("F")
      val G = new BlockMatrix(blocks.filter(x => x._1._1 >= m_bc.value & x._1._2 < m_bc.value)
        .map { case ((i, j), matrix) => ((i - m_bc.value, j), matrix) }, rowsPerBlock, colsPerBlock, res, colsPerBlock * m)
        .setName("G")
      val H = new BlockMatrix(blocks.filter(x => x._1._1 >= m_bc.value & x._1._2 >= m_bc.value)
        .map { case ((i, j), matrix) => ((i - m_bc.value, j - m_bc.value), matrix) }, rowsPerBlock, colsPerBlock, res, res)
        .setName("H")

      if (useCheckpoints) {
        E.cache()
        F.cache()
        G.cache()
        H.cache()
        cachedMatrices.addOne(E)
        cachedMatrices.addOne(F)
        cachedMatrices.addOne(G)
        cachedMatrices.addOne(H)

      }

      val E_inv = if (m_bc.value > (limit / colsPerBlock)) {
        E.inverse(limit, numMidDimSplits, useCheckpoints)
      } else {
        E.svdInv()
      }.setName("E_inv")

      if (useCheckpoints) {
        E_inv.cache()
        cachedMatrices.addOne(E_inv)
      }

      val mE_invF = E_inv.negative().multiply(F, numMidDimSplits).setName("mE_invF")
      val S = H.add(G.multiply(mE_invF, numMidDimSplits)).setName("S")

      if (useCheckpoints) {
        mE_invF.cache()
        S.cache()
        cachedMatrices.addOne(mE_invF)
        cachedMatrices.addOne(S)
      }

      val S_inv = if (m_bc.value > (limit / colsPerBlock)) {
        S.inverse(limit, numMidDimSplits, useCheckpoints)
      } else {
        S.svdInv()
      }.setName("S_inv")

      if (useCheckpoints) {
        S_inv.cache()
        cachedMatrices.addOne(S_inv)
      }

      val GE_inv = G.multiply(E_inv, numMidDimSplits).setName("GE_inv")
      if (useCheckpoints) {
        GE_inv.cache()
        cachedMatrices.addOne(GE_inv)
      }

      val mS_invGE_inv = S_inv.negative().multiply(GE_inv, numMidDimSplits).setName("mS_invGE_inv")
      val mE_invFS_inv = mE_invF.multiply(S_inv, numMidDimSplits).setName("mE_invFS_inv")
      if (useCheckpoints) {
        mS_invGE_inv.cache()
        mE_invFS_inv.cache()
        cachedMatrices.addOne(mS_invGE_inv)
        cachedMatrices.addOne(mE_invFS_inv)
      }

      val top_left = E_inv.subtract(mE_invFS_inv.multiply(GE_inv, numMidDimSplits)).blocks

      // Readjust the block indices
      val top_right = mE_invFS_inv.blocks.map {
        case ((i, j), mat) => ((i, j + m_bc.value), mat)
      }
      val bottom_left = mS_invGE_inv.blocks.map {
        case ((i, j), mat) => ((i + m_bc.value, j), mat)
      }
      val bottom_right = S_inv.blocks.map {
        case ((i, j), mat) => ((i + m_bc.value, j + m_bc.value), mat)
      }
      val sc = top_left.sparkContext
      val all_blocks = sc.union(top_left, top_right, bottom_left, bottom_right)
        .coalesce(numParts)

      val bm = new BlockMatrix(all_blocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
      if (useCheckpoints) {
        bm.checkpoint()
        cachedMatrices.foreach(bm => bm.blocks.unpersist(true))
      }
      bm
    }

    def inverse(): BlockMatrix = {
      inverse(4096, 1)
    }

    def inverse(limit: Int): BlockMatrix = {
      inverse(limit, 1)
    }

    /**
     * Creates identity matrix of size n and value in the diagonal
     *
     * @param n     Matrix size
     * @param value Value of the diagonal
     * @return Diagonal BlockMatrix
     */
    private def createEye(n: Long, value: Double = 1.0): BlockMatrix = {
      val eyeMaybe = eyeMatrixMap.get((n, value))
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
        bm = bm.setName("eye_" + (n, value)).cache()
        eyeMatrixMap.put((n, value), bm)
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
    def checkpoint(): BlockMatrix = {
      matrix.cache()
      matrix.blocks.checkpoint()
      matrix
    }
  }

}