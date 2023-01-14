import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

object Inverse {

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
      val svd = X.computeSVD(n, computeU = true)
      require(svd.s.size >= n, "svdInv called on singular matrix.")

      // Create the inv diagonal matrix from S
      val invS = DenseMatrix.diag(new DenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
      val U = svd.U
      val V = svd.V

      U.multiply(invS)
        .multiply(V.transpose)
        .toBlockMatrix(colsPerBlock = colsPerBlock, rowsPerBlock = rowsPerBlock)
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
    def inverse(limit: Int, numMidDimSplits: Int): BlockMatrix = {
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      // require(colsPerBlock == rowsPerBlock, "Sub-matrices has to be square!")
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      val m = math.ceil(matrix.numCols() / matrix.colsPerBlock / 2).toInt
      val blocks = matrix.blocks
      // split into block partitions
      val E = new BlockMatrix(blocks.filter(x => x._1._1 < m & x._1._2 < m), rowsPerBlock, colsPerBlock)
      val F = new BlockMatrix(blocks.filter(x => x._1._1 < m & x._1._2 >= m)
        .map { case ((i, j), matrix) => ((i, j - m), matrix) }, rowsPerBlock, colsPerBlock)
      val G = new BlockMatrix(blocks.filter(x => x._1._1 >= m & x._1._2 < m)
        .map { case ((i, j), matrix) => ((i - m, j), matrix) }, rowsPerBlock, colsPerBlock)
      val H = new BlockMatrix(blocks.filter(x => x._1._1 >= m & x._1._2 >= m)
        .map { case ((i, j), matrix) => ((i - m, j - m), matrix) }, rowsPerBlock, colsPerBlock)

      val E_inv = if (m > (limit / matrix.colsPerBlock)) {
        E.inverse(limit, numMidDimSplits)
      } else {
        E.svdInv()
      }

      val mE_invF = E_inv.negative().multiply(F, numMidDimSplits)
      val S = H.add(G.multiply(mE_invF, numMidDimSplits))

      val S_inv = if (m > (limit / matrix.colsPerBlock)) {
        S.inverse(limit, numMidDimSplits)
      } else {
        S.svdInv()
      }

      val GE_inv = G.multiply(E_inv, numMidDimSplits)
      val mS_invGE_inv = S_inv.negative().multiply(GE_inv, numMidDimSplits)
      val mE_invFS_inv = mE_invF.multiply(S_inv, numMidDimSplits)

      val top_left = E_inv.subtract(mE_invFS_inv.multiply(GE_inv, numMidDimSplits)).blocks
      val top_right = mE_invFS_inv.blocks.map {
        case ((i, j), mat) => ((i, j + m), mat)
      }
      val bottom_left = mS_invGE_inv.blocks.map {
        case ((i, j), mat) => ((i + m, j), mat)
      }
      val bottom_right = S_inv.blocks.map {
        case ((i, j), mat) => ((i + m, j + m), mat)
      }
      val sc = top_left.sparkContext
      val all_blocks = sc.union(top_left, top_right, bottom_left, bottom_right)
      new BlockMatrix(all_blocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
    }

    def inverse(): BlockMatrix = {
      inverse(2048, 1)
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
      val sc = matrix.blocks.sparkContext
      val diagonal = sc.range(start = 0, end = n)
        .map {
          i => {
            MatrixEntry(i, i, value)
          }
        }
      val cm = new CoordinateMatrix(diagonal, n, n)
      cm.toBlockMatrix(matrix.rowsPerBlock, matrix.colsPerBlock)
    }
  }

}