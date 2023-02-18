import breeze.linalg.InjectNumericOps
import breeze.linalg.randomDouble._zero
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import breeze.linalg.{DenseMatrix => BDM, inv => BINV, pinv => PBINV}
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object Inverse {
  private val eyeBlockMatrixMap: mutable.Map[(Long, Double), BlockMatrix] = mutable.Map[(Long, Double), BlockMatrix]()
  private val eyeCoordinateMatrixMap: mutable.Map[(Long, Double), CoordinateMatrix] = mutable.Map[(Long, Double), CoordinateMatrix]()

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
//      createEye(matrix.numRows(), -1.0).multiply(matrix)
      val newblocks = matrix.blocks.map { case ((i, j), mat) => {
        val nEye = -BDM.eye[Double](mat.numCols)
        val nEyeDM = new DenseMatrix(nEye.rows, nEye.cols, nEye.data, nEye.isTranspose)
        val newMat: Matrix = mat.multiply(nEyeDM)
        ((i, j),  newMat)
      }}
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

      require(matrix.blocks.sparkContext.getCheckpointDir.isDefined || useCheckpoints, "Checkpointing dir has to be set!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val colsPerBlock = matrix.colsPerBlock
      val rowsPerBlock = matrix.rowsPerBlock
      require(colsPerBlock == rowsPerBlock, "Sub-matrices has to be square!")
      val m = math.ceil(matrix.numCols() / colsPerBlock / 2).toInt
      val m_bc = matrix.blocks.sparkContext.broadcast(m)
      val blocks = matrix.blocks
      val numParts = (matrix.blocks.getNumPartitions).toInt

      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

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

//      println("E: " + E.colsPerBlock + " " + E.rowsPerBlock + ", " + E.numCols() + " " + E.numRows())
//      println("F: " + F.colsPerBlock + " " + F.rowsPerBlock + ", " + F.numCols() + " " + F.numRows())
//      println("G: " + G.colsPerBlock + " " + G.rowsPerBlock + ", " + G.numCols() + " " + G.numRows())
//      println("H: " + H.colsPerBlock + " " + H.rowsPerBlock + ", " + H.numCols() + " " + H.numRows())


      if (useCheckpoints) {
        E.checkpoint()
        F.checkpoint()
        G.checkpoint()
        H.checkpoint()
        cachedMatrices.addOne(E)
        cachedMatrices.addOne(F)
        cachedMatrices.addOne(G)
        cachedMatrices.addOne(H)

      }

      val E_inv = if (m_bc.value > (limit / colsPerBlock)) {
        E.inverse(limit, numMidDimSplits, useCheckpoints, depth = depth +1)
      } else {
        E.svdInv()
      }.setName("E_inv")

      if (useCheckpoints) {
        E_inv.checkpoint()
        cachedMatrices.addOne(E_inv)
      }

      val mE_invF = E_inv.negative().multiply(F, numMidDimSplits).setName("mE_invF")
      val S = H.add(G.multiply(mE_invF, numMidDimSplits)).setName("S")

      if (useCheckpoints) {
        mE_invF.checkpoint()
        S.checkpoint()
        cachedMatrices.addOne(mE_invF)
        cachedMatrices.addOne(S)
      }

      val S_inv = if (m_bc.value > (limit / colsPerBlock)) {
        S.inverse(limit, numMidDimSplits, useCheckpoints, depth = depth +1)
      } else {
        S.svdInv()
      }.setName("S_inv")

      if (useCheckpoints) {
        S_inv.checkpoint()
        cachedMatrices.addOne(S_inv)
      }

      val GE_inv = G.multiply(E_inv, numMidDimSplits).setName("GE_inv")
      if (useCheckpoints) {
        GE_inv.checkpoint()
        cachedMatrices.addOne(GE_inv)
      }

      val mS_invGE_inv = S_inv.negative().multiply(GE_inv, numMidDimSplits).setName("mS_invGE_inv")
      val mE_invFS_inv = mE_invF.multiply(S_inv, numMidDimSplits).setName("mE_invFS_inv")
      if (useCheckpoints) {
        mS_invGE_inv.checkpoint()
        mE_invFS_inv.checkpoint()
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
      val eyeMaybe = eyeBlockMatrixMap.get((n, value))
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
        eyeBlockMatrixMap.put((n, value), bm)
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
      matrix.blocks.checkpoint()
      matrix
    }
  }


  implicit class CoordinateMatrixInverse(val matrix: CoordinateMatrix) {

    private val cachedMatrices: ListBuffer[CoordinateMatrix] = mutable.ListBuffer.empty

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
     * Return the negative of this [[BlockMatrix]].
     * A.negative() = -A
     *
     * @return BlockMatrix
     */
    def negative(): CoordinateMatrix = {
      //      createEye(matrix.numRows(), -1.0).multiply(matrix)
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
    def inverse(limit: Int, useCheckpoints: Boolean = true, depth: Int = 0): CoordinateMatrix = {

      require(matrix.entries.sparkContext.getCheckpointDir.isDefined || useCheckpoints, "Checkpointing dir has to be set!")
      require(matrix.numRows() == matrix.numCols(), "Matrix has to be square!")
      val numCols = matrix.numCols()
      val numRows = matrix.numRows()
      val m = math.ceil(numCols / 2).toInt
      val m_bc = matrix.entries.sparkContext.broadcast(m)
      val entries: RDD[MatrixEntry] = matrix.entries
      val numParts = (matrix.entries.getNumPartitions).toInt

      println("Input matrix shape: " + matrix.numRows() + ", " + matrix.numCols() + " At depth=" + depth)

      // split into block partitions and readjust the block indices
      val E = new CoordinateMatrix(entries.filter(x => x.i < m_bc.value & x.j < m_bc.value))
        .setName("E")
      val F = new CoordinateMatrix(entries.filter(x => x.i < m_bc.value & x.j >= m_bc.value)
        .map { case MatrixEntry(i, j, matrix) => MatrixEntry(i, j - m_bc.value, matrix) })
        .setName("F")
      val G = new CoordinateMatrix(entries.filter(x => x.i >= m_bc.value & x.j < m_bc.value)
        .map { case MatrixEntry(i, j, matrix) => MatrixEntry(i - m_bc.value, j, matrix) })
        .setName("G")
      val H = new CoordinateMatrix(entries.filter(x => x.i >= m_bc.value & x.j >= m_bc.value)
        .map { case MatrixEntry(i, j, matrix) => MatrixEntry(i - m_bc.value, j - m_bc.value, matrix) })
        .setName("H")

      if (useCheckpoints) {
        E.checkpoint()
        F.checkpoint()
        G.checkpoint()
        H.checkpoint()
        cachedMatrices.addOne(E)
        cachedMatrices.addOne(F)
        cachedMatrices.addOne(G)
        cachedMatrices.addOne(H)

      }

      val E_inv = if (m_bc.value > limit) {
        E.inverse(limit, useCheckpoints, depth = depth + 1)
      } else {
        E.svdInv()
      }.setName("E_inv")

      if (useCheckpoints) {
        E_inv.checkpoint()
        cachedMatrices.addOne(E_inv)
      }

      val mE_invF = E_inv.negative().multiply(F).setName("mE_invF")
      val S = H.add(G.multiply(mE_invF)).setName("S")

      if (useCheckpoints) {
        mE_invF.checkpoint()
        S.checkpoint()
        cachedMatrices.addOne(mE_invF)
        cachedMatrices.addOne(S)
      }

      val S_inv = if (m_bc.value > limit) {
        S.inverse(limit, useCheckpoints, depth = depth + 1)
      } else {
        S.svdInv()
      }.setName("S_inv")

      if (useCheckpoints) {
        S_inv.checkpoint()
        cachedMatrices.addOne(S_inv)
      }

      val GE_inv = G.multiply(E_inv).setName("GE_inv")
      if (useCheckpoints) {
        GE_inv.checkpoint()
        cachedMatrices.addOne(GE_inv)
      }

      val mS_invGE_inv = S_inv.negative().multiply(GE_inv).setName("mS_invGE_inv")
      val mE_invFS_inv = mE_invF.multiply(S_inv).setName("mE_invFS_inv")
      if (useCheckpoints) {
        mS_invGE_inv.checkpoint()
        mE_invFS_inv.checkpoint()
        cachedMatrices.addOne(mS_invGE_inv)
        cachedMatrices.addOne(mE_invFS_inv)
      }

      val top_left = E_inv.subtract(mE_invFS_inv.multiply(GE_inv)).entries

      // Readjust the block indices
      val top_right = mE_invFS_inv.entries.map {
        case MatrixEntry(i, j, v) => MatrixEntry(i, j + m_bc.value, v)
      }
      val bottom_left = mS_invGE_inv.entries.map {
        case MatrixEntry(i, j, v) => MatrixEntry(i + m_bc.value, j, v)
      }
      val bottom_right = S_inv.entries.map {
        case MatrixEntry(i, j, v) => MatrixEntry(i + m_bc.value, j + m_bc.value, v)
      }
      val sc = top_left.sparkContext
      val all_blocks = sc.union(top_left, top_right, bottom_left, bottom_right)
        .coalesce(numParts)

      val cm = new CoordinateMatrix(all_blocks, matrix.numRows(), matrix.numCols())
      if (useCheckpoints) {
        cm.checkpoint()
        cachedMatrices.foreach(cm => cm.entries.unpersist(true))
      }
      cm
    }

    def inverse(): CoordinateMatrix = {
      inverse(4096)
    }

    def inverse(limit: Int): CoordinateMatrix = {
      inverse(limit, depth = 0)
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
        cm = cm.setName("eye_" + (n, value)).cache()
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
      val entries = matrix.entries.keyBy{ case MatrixEntry(i, j, _) => (i, j)}
        .join(other.entries.keyBy{ case MatrixEntry(i, j, _) => (i, j)})
        .mapValues{ case (me1, me2) => me1.value + me2.value}
        .map{ case ((i, j), v)  => MatrixEntry(i, j, v)}
      new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
    }

    def subtract(other: CoordinateMatrix): CoordinateMatrix = {
      val entries = matrix.entries.keyBy { case MatrixEntry(i, j, _) => (i, j) }
        .join(other.entries.keyBy { case MatrixEntry(i, j, _) => (i, j) })
        .mapValues { case (me1, me2) => me1.value - me2.value }
        .map { case ((i, j), v) => MatrixEntry(i, j, v) }
      new CoordinateMatrix(entries, matrix.numRows(), matrix.numCols())
    }

    def partitionBy(partitioner: Partitioner): CoordinateMatrix = {
      val newEntries = matrix.entries.keyBy(x => (x.i, x.j))
        .partitionBy(partitioner)
        .values
      new CoordinateMatrix(newEntries, matrix.numRows(), matrix.numCols())
    }

//    def leftPseudoInverse(limit: Int): CoordinateMatrix = {
//      matrix.transpose.multiply(matrix).inverse(limit).multiply(matrix.transpose)
//    }
//
//    def leftPseudoInverse(limit: Int): CoordinateMatrix = {
//      leftPseudoInverse(limit)
//    }
//
//    def leftPseudoInverse(): CoordinateMatrix = {
//      matrix.transpose.multiply(matrix).inverse().multiply(matrix.transpose)
//    }
//
//    def rightPseudoInverse(limit: Int): CoordinateMatrix = {
//      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse(limit))
//    }
//
//    def rightPseudoInverse(limit: Int): CoordinateMatrix = {
//      rightPseudoInverse(limit)
//    }
//
//    def rightPseudoInverse(): CoordinateMatrix = {
//      matrix.transpose.multiply(matrix.multiply(matrix.transpose).inverse())
//    }

    def setName(name: String): CoordinateMatrix = {
      matrix.entries.setName(name)
      matrix
    }
    def cache(): CoordinateMatrix = {
      matrix.entries.cache()
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