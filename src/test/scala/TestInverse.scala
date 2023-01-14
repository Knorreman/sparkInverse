import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.scalatest.funsuite.AnyFunSuite
import Inverse.BlockMatrixInverse
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseMatrix => BDM, inv => BINV}

class TestInverse extends AnyFunSuite {

  def breezeToDenseMatrix(dm: BDM[Double]): DenseMatrix = {
    new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
  }

  def denseMatrixToBreeze(mat: BlockMatrix): BDM[Double] = {
    val localMat = mat.toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  test("inverse"){
    val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("Testing"))
    sc.setLogLevel("ERROR")
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    val numPartitions = 3
    val sq_mat = new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)

    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(sq_mat)))
    val bm_inv = sq_mat.inverse(3, 3)
    assert(bm_inv.numRows() === sq_mat.numRows())
    assert(bm_inv.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv.toLocalMatrix(), expected, 1e-12))
    // Test with default arguments
    val bm_inv2 = sq_mat.inverse()
    assert(bm_inv2.numRows() === sq_mat.numRows())
    assert(bm_inv2.numCols() === sq_mat.numCols())
    assert(testMatrixSimilarity(bm_inv2.toLocalMatrix(), expected, 1e-12))
  }
  def testMatrixSimilarity(A: Matrix, B: DenseMatrix, tol: Double): Boolean = {
    A.toArray.zip(B.values).forall {case (a, b) => math.abs(a - b) < tol}
  }

}