import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

import scala.collection.mutable

object Inverse {
  private val eyeMatrixMap: mutable.Map[(Long, Double), BlockMatrix] = mutable.Map[(Long, Double), BlockMatrix]()

  implicit class BlockMatrixInverse(val matrix: BlockMatrix) {

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
     * @return BlockMatrix
     */
    def inverse(limit: Int, numMidDimSplits: Int, useCheckpoints: Boolean = true): BlockMatrix = {

      require(matrix.blocks.sparkContext.getCheckpointDir.isDefined, "Checkpointing dir has to be set!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      require(colsPerBlock == rowsPerBlock, "Sub-matrices has to be square!")
      val m = math.ceil(matrix.numCols() / colsPerBlock / 2).toInt
      val m_bc = matrix.blocks.sparkContext.broadcast(m)
      val blocks = matrix.blocks
      val numParts = (matrix.blocks.getNumPartitions).toInt

      val res = matrix.numRows() - colsPerBlock * m

      // split into block partitions and readjust the block indices
      val E = new BlockMatrix(blocks.filter(x => x._1._1 < m_bc.value & x._1._2 < m_bc.value), rowsPerBlock, colsPerBlock, rowsPerBlock * m, colsPerBlock * m)
      val F = new BlockMatrix(blocks.filter(x => x._1._1 < m_bc.value & x._1._2 >= m_bc.value)
        .map { case ((i, j), matrix) => ((i, j - m_bc.value), matrix) }, rowsPerBlock, colsPerBlock, rowsPerBlock * m, res)
      val G = new BlockMatrix(blocks.filter(x => x._1._1 >= m_bc.value & x._1._2 < m_bc.value)
        .map { case ((i, j), matrix) => ((i - m_bc.value, j), matrix) }, rowsPerBlock, colsPerBlock, res, colsPerBlock * m)
      val H = new BlockMatrix(blocks.filter(x => x._1._1 >= m_bc.value & x._1._2 >= m_bc.value)
        .map { case ((i, j), matrix) => ((i - m_bc.value, j - m_bc.value), matrix) }, rowsPerBlock, colsPerBlock, res, res)

      val E_inv = if (m_bc.value > (limit / colsPerBlock)) {
        E.inverse(limit, numMidDimSplits, useCheckpoints)
      } else {
        E.svdInv()
      }

      val mE_invF = E_inv.negative().multiply(F, numMidDimSplits)
      val S = H.add(G.multiply(mE_invF, numMidDimSplits))

      val S_inv = if (m_bc.value > (limit / colsPerBlock)) {
        S.inverse(limit, numMidDimSplits, useCheckpoints)
      } else {
        S.svdInv()
      }

      val GE_inv = G.multiply(E_inv, numMidDimSplits)
      val mS_invGE_inv = S_inv.negative().multiply(GE_inv, numMidDimSplits)
      val mE_invFS_inv = mE_invF.multiply(S_inv, numMidDimSplits)

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
        .repartition(numParts)

      val bm = new BlockMatrix(all_blocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
      if (useCheckpoints) {
        bm.blocks.checkpoint()
      }
      bm
    }

    def inverse(): BlockMatrix = {
      inverse(1024, 1)
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

    def pseudoInverse(limit: Int, numMidDimSplits: Int): BlockMatrix = {
//      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse(limit, numMidDimSplits))
      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse(limit, numMidDimSplits))
    }

    def pseudoInverse(limit: Int): BlockMatrix = {
      pseudoInverse(limit, 1)
    }

    def pseudoInverse(): BlockMatrix = {
//      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse())
      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse())
    }

    def setName (name: String): BlockMatrix = {
      matrix.blocks.setName(name)
      matrix
    }
  }

}