import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.scalatest.funsuite.AnyFunSuite
import Inverse.{BlockMatrixInverse, CoordinateMatrixInverse}
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseMatrix => BDM, inv => BINV, pinv => PBINV}

import java.nio.file.Files

class TestInverse extends AnyFunSuite {
  val sc: SparkContext = setup()
  val numPartitions = 3

  def setup(): SparkContext = {
    val sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Testing"))
    sc.setLogLevel("ERROR")
    val tempDir = Files.createTempDirectory("tmpCheckpointDir").toFile.toString
    println("Using checkpointDir: " + tempDir)
    sc.setCheckpointDir(tempDir)
    sc
  }
  def breezeToDenseMatrix(dm: BDM[Double]): DenseMatrix = {
    new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
  }

  def denseMatrixToBreeze(mat: BlockMatrix): BDM[Double] = {
    val localMat = mat.toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  def denseMatrixToBreeze(mat: CoordinateMatrix): BDM[Double] = {
    val localMat = mat.toBlockMatrix().toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  test("inverse without checkpoint"){
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)

    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(sq_mat)))
    val bm_inv = sq_mat.inverse(3, 3, useCheckpoints = false)
    assert(bm_inv.numRows() === sq_mat.numRows())
    assert(bm_inv.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))
    // Test with default arguments
    val bm_inv2 = sq_mat.inverse()
    assert(bm_inv2.numRows() === sq_mat.numRows())
    assert(bm_inv2.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv2.toLocalMatrix(), expected, 1e-12))
  }

  test("inverse with checkpoint") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)

    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(sq_mat)))
    val bm_inv = sq_mat.inverse(3, 3)
    assert(bm_inv.numRows() === sq_mat.numRows())
    assert(bm_inv.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))
  }

  def testMatrixSimilarity(A: Matrix, B: DenseMatrix, tol: Double): Boolean = {
    A.toArray.zip(B.toArray).forall {case (a, b) => math.abs(a - b) < tol}
  }

  test("SVD Inverse") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 1.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(sq_mat)))
    val bm_inv = sq_mat.svdInv()

    assert(bm_inv.numRows() === sq_mat.numRows())
    assert(bm_inv.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))
  }

  test("localInv BlockMatrix") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 1.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(sq_mat)))
    val bm_inv = sq_mat.localInv()

    assert(bm_inv.numRows() === sq_mat.numRows())
    assert(bm_inv.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))
  }

  test("localInv CoordinateMatrix") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 1.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val coo_mat = sq_mat.toCoordinateMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(coo_mat)))
    val cm_inv = coo_mat.localInv()

    assert(cm_inv.numRows() === coo_mat.numRows())
    assert(cm_inv.numCols() === coo_mat.numCols())
    assert(testMatrixSimilarity(cm_inv.toBlockMatrix().toLocalMatrix(), expected, 1e-12))
  }

  test("LeftPseudoInverse") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 3, Array(1.0, 2.0, 2.0, 4.0, 3, 2))),
      ((0, 1), new DenseMatrix(2, 3, Array(4.0, 3.0, 2.0, 2.0, 1, 4))),
      ((1, 1), new DenseMatrix(2, 3, Array(1.0, 5.0, 3.0, 4.0, 2, 1))),
      ((1, 0), new DenseMatrix(2, 3, Array(-1.0, 0.0, 2.0, 1.0, 3, 4)))
    )
    val not_sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 3).transpose

    val expected = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(not_sq_mat)))
    val bm_inv = not_sq_mat.leftPseudoInverse(3, 2)

    assert(bm_inv.numRows() === not_sq_mat.numCols())
    assert(bm_inv.numCols() === not_sq_mat.numRows())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))

    // Test with default arguments
    val bm_inv2 = not_sq_mat.leftPseudoInverse()
    assert(bm_inv2.numRows() === not_sq_mat.numCols())
    assert(bm_inv2.numCols() === not_sq_mat.numRows())
    assert(testMatrixSimilarity(bm_inv2.toLocalMatrix(), expected, 1e-12))
  }

  test("RightPseudoInverse") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 3, Array(1.0, 2.0, 2.0, 4.0, 3, 2))),
      ((0, 1), new DenseMatrix(2, 3, Array(4.0, 3.0, 2.0, 2.0, 1, 4))),
      ((1, 1), new DenseMatrix(2, 3, Array(1.0, 5.0, 3.0, 4.0, 2, 1))),
      ((1, 0), new DenseMatrix(2, 3, Array(-1.0, 0.0, 2.0, 1.0, 3, 4)))
    )
    val not_sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 3)

    val expected = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(not_sq_mat)))
    val bm_inv = not_sq_mat.rightPseudoInverse(3, 2)

    assert(bm_inv.numRows() === not_sq_mat.numCols())
    assert(bm_inv.numCols() === not_sq_mat.numRows())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))

    // Test with default arguments
    val bm_inv2 = not_sq_mat.rightPseudoInverse()
    assert(bm_inv2.numRows() === not_sq_mat.numCols())
    assert(bm_inv2.numCols() === not_sq_mat.numRows())
    assert(testMatrixSimilarity(bm_inv2.toLocalMatrix(), expected, 1e-12))
  }

  test("CoordinateMatrix add with non-overlapping entries") {
    val entries1 = sc.parallelize(Seq(MatrixEntry(0, 0, 1.0), MatrixEntry(1, 1, 2.0)))
    val entries2 = sc.parallelize(Seq(MatrixEntry(0, 1, 3.0), MatrixEntry(1, 0, 4.0)))
    val cm1 = new CoordinateMatrix(entries1, 2, 2)
    val cm2 = new CoordinateMatrix(entries2, 2, 2)
    val result = cm1.add(cm2).entries.collect().sortBy(e => (e.i, e.j))
    assert(result.length === 4)
    assert(result.exists(e => e.i == 0 && e.j == 0 && math.abs(e.value - 1.0) < 1e-12))
    assert(result.exists(e => e.i == 0 && e.j == 1 && math.abs(e.value - 3.0) < 1e-12))
    assert(result.exists(e => e.i == 1 && e.j == 0 && math.abs(e.value - 4.0) < 1e-12))
    assert(result.exists(e => e.i == 1 && e.j == 1 && math.abs(e.value - 2.0) < 1e-12))
  }

  test("CoordinateMatrix subtract with non-overlapping entries") {
    val entries1 = sc.parallelize(Seq(MatrixEntry(0, 0, 5.0), MatrixEntry(1, 1, 6.0)))
    val entries2 = sc.parallelize(Seq(MatrixEntry(0, 1, 3.0), MatrixEntry(1, 0, 4.0)))
    val cm1 = new CoordinateMatrix(entries1, 2, 2)
    val cm2 = new CoordinateMatrix(entries2, 2, 2)
    val result = cm1.subtract(cm2).entries.collect().sortBy(e => (e.i, e.j))
    assert(result.length === 4)
    assert(result.exists(e => e.i == 0 && e.j == 0 && math.abs(e.value - 5.0) < 1e-12))
    assert(result.exists(e => e.i == 0 && e.j == 1 && math.abs(e.value - (-3.0)) < 1e-12))
    assert(result.exists(e => e.i == 1 && e.j == 0 && math.abs(e.value - (-4.0)) < 1e-12))
    assert(result.exists(e => e.i == 1 && e.j == 1 && math.abs(e.value - 6.0) < 1e-12))
  }

  test("CoordinateMatrix inverse with checkpoint") {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val coo_mat = sq_mat.toCoordinateMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(coo_mat)))
    val bm_inv = coo_mat.inverse(3)
    assert(bm_inv.numRows() === coo_mat.numRows())
    assert(bm_inv.numCols() === coo_mat.numCols())
    assert(testMatrixSimilarity(bm_inv.toBlockMatrix().toLocalMatrix(), expected, 1e-12))
  }

  test("BlockMatrix normOne and normInf") {
    // Matrix (column-major per block):
    // [1  3]
    // [2  4]
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 3.0, 4.0)))
    )
    val mat = new BlockMatrix(sc.parallelize(blocks, 1), 2, 2)
    // ||A||_1 = max col sum of abs = max(|1|+|2|, |3|+|4|) = 7
    assert(math.abs(mat.normOne() - 7.0) < 1e-12)
    // ||A||_inf = max row sum of abs = max(|1|+|3|, |2|+|4|) = 6
    assert(math.abs(mat.normInf() - 6.0) < 1e-12)
  }

  test("CoordinateMatrix normOne and normInf") {
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 1.0), MatrixEntry(0, 1, 3.0),
      MatrixEntry(1, 0, 2.0), MatrixEntry(1, 1, 4.0)
    ))
    val mat = new CoordinateMatrix(entries, 2, 2)
    assert(math.abs(mat.normOne() - 7.0) < 1e-12)
    assert(math.abs(mat.normInf() - 6.0) < 1e-12)
  }

  test("BlockMatrix iterativeInverse on diagonally dominant matrix") {
    // Build a 4x4 diagonally dominant matrix: 10*I + small off-diag
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(10.0, 0.5, 0.3, 10.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(0.1, 0.2, 0.4, 0.1))),
      ((1, 0), new DenseMatrix(2, 2, Array(0.2, 0.3, 0.1, 0.2))),
      ((1, 1), new DenseMatrix(2, 2, Array(10.0, 0.4, 0.2, 10.0)))
    )
    val mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(mat)))
    val inv = mat.iterativeInverse(maxIter = 30, tolerance = 1e-10)

    assert(inv.numRows() === mat.numRows())
    assert(inv.numCols() === mat.numCols())
    assert(testMatrixSimilarity(inv.toLocalMatrix(), expected, 1e-8))
  }

  test("CoordinateMatrix iterativeInverse on diagonally dominant matrix") {
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 10.0), MatrixEntry(0, 1, 0.3), MatrixEntry(0, 2, 0.1), MatrixEntry(0, 3, 0.4),
      MatrixEntry(1, 0, 0.5), MatrixEntry(1, 1, 10.0), MatrixEntry(1, 2, 0.2), MatrixEntry(1, 3, 0.1),
      MatrixEntry(2, 0, 0.2), MatrixEntry(2, 1, 0.1), MatrixEntry(2, 2, 10.0), MatrixEntry(2, 3, 0.2),
      MatrixEntry(3, 0, 0.3), MatrixEntry(3, 1, 0.2), MatrixEntry(3, 2, 0.4), MatrixEntry(3, 3, 10.0)
    ))
    val mat = new CoordinateMatrix(entries, 4, 4)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(mat)))
    val inv = mat.iterativeInverse(maxIter = 30, tolerance = 1e-10)

    assert(inv.numRows() === mat.numRows())
    assert(inv.numCols() === mat.numCols())
    assert(testMatrixSimilarity(inv.toBlockMatrix().toLocalMatrix(), expected, 1e-8))
  }

  test("BlockMatrix iterativeInverse on general (non-diagonally-dominant) matrix") {
    // Same 4x4 matrix used in the Schur complement inverse tests — large off-diagonal entries,
    // NOT diagonally dominant (e.g. row 0: |1| < |1| + |4| + |1|).
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(mat)))
    val inv = mat.iterativeInverse(maxIter = 50, tolerance = 1e-10)

    assert(inv.numRows() === mat.numRows())
    assert(inv.numCols() === mat.numCols())
    assert(testMatrixSimilarity(inv.toLocalMatrix(), expected, 1e-6))
  }

  test("CoordinateMatrix iterativeInverse on general (non-diagonally-dominant) matrix") {
    // Same general matrix as the BlockMatrix test above, expressed as entries.
    // Column-major within each 2x2 block:
    //   E=[[1,1],[2,4]]  F=[[4,2],[3,2]]  G=[[1,3],[5,4]]  H=[[-1,2],[6,1]]
    // Full matrix:
    //   [1  1  4  2]
    //   [2  4  3  2]
    //   [1  3 -1  2]
    //   [5  4  6  1]
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 1.0), MatrixEntry(0, 1, 1.0), MatrixEntry(0, 2, 4.0), MatrixEntry(0, 3, 2.0),
      MatrixEntry(1, 0, 2.0), MatrixEntry(1, 1, 4.0), MatrixEntry(1, 2, 3.0), MatrixEntry(1, 3, 2.0),
      MatrixEntry(2, 0, 1.0), MatrixEntry(2, 1, 3.0), MatrixEntry(2, 2, -1.0), MatrixEntry(2, 3, 2.0),
      MatrixEntry(3, 0, 5.0), MatrixEntry(3, 1, 4.0), MatrixEntry(3, 2, 6.0), MatrixEntry(3, 3, 1.0)
    ))
    val mat = new CoordinateMatrix(entries, 4, 4)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(mat)))
    val inv = mat.iterativeInverse(maxIter = 50, tolerance = 1e-10)

    assert(inv.numRows() === mat.numRows())
    assert(inv.numCols() === mat.numCols())
    assert(testMatrixSimilarity(inv.toBlockMatrix().toLocalMatrix(), expected, 1e-6))
  }

  test("BlockMatrix iterativeInverse on matrix with negative eigenvalues") {
    // A symmetric matrix with mixed positive/negative off-diagonal elements.
    // Eigenvalues are all positive (positive definite) but NOT diagonally dominant.
    //   [3  -1   0   0]
    //   [-1  3  -1   0]
    //   [0  -1   3  -1]
    //   [0   0  -1   3]
    // (Tridiagonal — condition number ~7, not diagonally dominant at interior rows)
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(3.0, -1.0, -1.0, 3.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(0.0, -1.0, 0.0, 0.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(0.0, 0.0, -1.0, 0.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(3.0, -1.0, -1.0, 3.0)))
    )
    val mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(mat)))
    val inv = mat.iterativeInverse(maxIter = 50, tolerance = 1e-10)

    assert(inv.numRows() === mat.numRows())
    assert(inv.numCols() === mat.numCols())
    assert(testMatrixSimilarity(inv.toLocalMatrix(), expected, 1e-6))
  }

  test("CoordinateMatrix iterativeInverse on asymmetric matrix") {
    // An asymmetric 4x4 matrix where row sums of |off-diag| exceed diagonal.
    //   [2   5  -1   1]
    //   [0   3   4  -2]
    //   [1  -1   2   3]
    //   [-3  1   0   4]
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 2.0), MatrixEntry(0, 1, 5.0), MatrixEntry(0, 2, -1.0), MatrixEntry(0, 3, 1.0),
      MatrixEntry(1, 0, 0.0), MatrixEntry(1, 1, 3.0), MatrixEntry(1, 2, 4.0), MatrixEntry(1, 3, -2.0),
      MatrixEntry(2, 0, 1.0), MatrixEntry(2, 1, -1.0), MatrixEntry(2, 2, 2.0), MatrixEntry(2, 3, 3.0),
      MatrixEntry(3, 0, -3.0), MatrixEntry(3, 1, 1.0), MatrixEntry(3, 2, 0.0), MatrixEntry(3, 3, 4.0)
    ))
    val mat = new CoordinateMatrix(entries, 4, 4)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(mat)))
    val inv = mat.iterativeInverse(maxIter = 50, tolerance = 1e-10)

    assert(inv.numRows() === mat.numRows())
    assert(inv.numCols() === mat.numCols())
    assert(testMatrixSimilarity(inv.toBlockMatrix().toLocalMatrix(), expected, 1e-6))
  }

}
