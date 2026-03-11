package sparkinverse

import breeze.linalg.{DenseMatrix => BDM, inv => BINV, pinv => PBINV}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.funsuite.AnyFunSuite
import sparkinverse.api.{IterativeInverseConfig, MatrixInversion, RecursiveInverseConfig}
import sparkinverse.syntax.block._
import sparkinverse.syntax.coordinate._

import java.nio.file.Files

class TestInverse extends AnyFunSuite {
  val sc: SparkContext = setup()
  val numPartitions = 3

  def setup(): SparkContext = {
    val context = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("Testing"))
    context.setLogLevel("ERROR")
    val tempDir = Files.createTempDirectory("tmpCheckpointDir").toFile.toString
    println("Using checkpointDir: " + tempDir)
    context.setCheckpointDir(tempDir)
    context
  }

  def breezeToDenseMatrix(dm: BDM[Double]): DenseMatrix = new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)

  def denseMatrixToBreeze(mat: BlockMatrix): BDM[Double] = {
    val localMat = mat.toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  def denseMatrixToBreeze(mat: CoordinateMatrix): BDM[Double] = {
    val localMat = mat.toBlockMatrix().toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  def testMatrixSimilarity(a: Matrix, b: DenseMatrix, tol: Double): Boolean =
    a.toArray.zip(b.toArray).forall { case (left, right) => math.abs(left - right) < tol }

  test("block facade inverse without checkpoint") {
    val matrix = sampleBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    val config = RecursiveInverseConfig(limit = 3, numMidDimSplits = 3, useCheckpoints = false)
    val inverse = MatrixInversion.block(matrix).inverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-12))
  }

  test("block syntax inverse default") {
    val matrix = sampleBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    val inverse = matrix.inverse()
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-12))
  }

  test("coordinate facade inverse with checkpoint") {
    val matrix = sampleBlockMatrix().toCoordinateMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    val inverse = MatrixInversion.coordinate(matrix).inverse(RecursiveInverseConfig(limit = 3))
    assert(testMatrixSimilarity(inverse.toBlockMatrix().toLocalMatrix(), expected, 1e-12))
  }

  test("block svd and local inverse") {
    val matrix = sampleBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    assert(testMatrixSimilarity(MatrixInversion.block(matrix).svdInverse().toLocalMatrix(), expected, 1e-12))
    assert(testMatrixSimilarity(MatrixInversion.block(matrix).localInverse().toLocalMatrix(), expected, 1e-12))
  }

  test("coordinate local inverse and syntax methods") {
    val matrix = sampleBlockMatrix().toCoordinateMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    assert(testMatrixSimilarity(MatrixInversion.coordinate(matrix).localInverse().toBlockMatrix().toLocalMatrix(), expected, 1e-12))
    assert(testMatrixSimilarity(matrix.svdInverse().toBlockMatrix().toLocalMatrix(), expected, 1e-12))
  }

  test("pseudo inverse facade") {
    val tall = sampleRectangularBlockMatrix().transpose
    val expectedTall = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(tall)))
    val left = MatrixInversion.block(tall).leftPseudoInverse(RecursiveInverseConfig(limit = 3, numMidDimSplits = 2))
    assert(testMatrixSimilarity(left.toLocalMatrix(), expectedTall, 1e-12))

    val wide = sampleRectangularBlockMatrix()
    val expectedWide = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(wide)))
    val right = MatrixInversion.block(wide).rightPseudoInverse(RecursiveInverseConfig(limit = 3, numMidDimSplits = 2))
    assert(testMatrixSimilarity(right.toLocalMatrix(), expectedWide, 1e-12))
  }

  test("coordinate add and subtract through syntax") {
    val entries1 = sc.parallelize(Seq(MatrixEntry(0, 0, 1.0), MatrixEntry(1, 1, 2.0)))
    val entries2 = sc.parallelize(Seq(MatrixEntry(0, 1, 3.0), MatrixEntry(1, 0, 4.0)))
    val cm1 = new CoordinateMatrix(entries1, 2, 2)
    val cm2 = new CoordinateMatrix(entries2, 2, 2)
    val added = cm1.add(cm2).entries.collect().sortBy(e => (e.i, e.j))
    val subtracted = cm1.subtract(cm2).entries.collect().sortBy(e => (e.i, e.j))
    assert(added.length === 4)
    assert(subtracted.exists(e => e.i == 0 && e.j == 1 && math.abs(e.value + 3.0) < 1e-12))
  }

  test("norms through facade and syntax") {
    val block = new BlockMatrix(sc.parallelize(Seq(((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 3.0, 4.0)))), 1), 2, 2)
    val coord = new CoordinateMatrix(sc.parallelize(Seq(
      MatrixEntry(0, 0, 1.0), MatrixEntry(0, 1, 3.0), MatrixEntry(1, 0, 2.0), MatrixEntry(1, 1, 4.0)
    )), 2, 2)
    assert(math.abs(MatrixInversion.block(block).normOne() - 7.0) < 1e-12)
    assert(math.abs(block.normInf() - 6.0) < 1e-12)
    assert(math.abs(MatrixInversion.coordinate(coord).normOne() - 7.0) < 1e-12)
    assert(math.abs(coord.normInf() - 6.0) < 1e-12)
  }

  test("iterative inverse config on block and coordinate") {
    val block = diagonallyDominantBlockMatrix()
    val blockExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(block)))
    val blockInverse = MatrixInversion.block(block).iterativeInverse(IterativeInverseConfig(maxIter = 30, tolerance = 1e-10))
    assert(testMatrixSimilarity(blockInverse.toLocalMatrix(), blockExpected, 1e-8))

    val coord = diagonallyDominantCoordinateMatrix()
    val coordExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(coord)))
    val coordInverse = coord.iterativeInverse(IterativeInverseConfig(maxIter = 30, tolerance = 1e-10))
    assert(testMatrixSimilarity(coordInverse.toBlockMatrix().toLocalMatrix(), coordExpected, 1e-8))
  }

  private def sampleBlockMatrix(): BlockMatrix = {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 2.0, 1.0, 4.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(4.0, 3.0, 2.0, 2.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 5.0, 3.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(-1.0, 6.0, 2.0, 1.0)))
    )
    new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
  }

  private def sampleRectangularBlockMatrix(): BlockMatrix = {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 3, Array(1.0, 2.0, 2.0, 4.0, 3.0, 2.0))),
      ((0, 1), new DenseMatrix(2, 3, Array(4.0, 3.0, 2.0, 2.0, 1.0, 4.0))),
      ((1, 1), new DenseMatrix(2, 3, Array(1.0, 5.0, 3.0, 4.0, 2.0, 1.0))),
      ((1, 0), new DenseMatrix(2, 3, Array(-1.0, 0.0, 2.0, 1.0, 3.0, 4.0)))
    )
    new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 3)
  }

  private def diagonallyDominantBlockMatrix(): BlockMatrix = {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(10.0, 0.5, 0.3, 10.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(0.1, 0.2, 0.4, 0.1))),
      ((1, 0), new DenseMatrix(2, 2, Array(0.2, 0.3, 0.1, 0.2))),
      ((1, 1), new DenseMatrix(2, 2, Array(10.0, 0.4, 0.2, 10.0)))
    )
    new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2)
  }

  private def diagonallyDominantCoordinateMatrix(): CoordinateMatrix = {
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 10.0), MatrixEntry(0, 1, 0.3), MatrixEntry(0, 2, 0.1), MatrixEntry(0, 3, 0.4),
      MatrixEntry(1, 0, 0.5), MatrixEntry(1, 1, 10.0), MatrixEntry(1, 2, 0.2), MatrixEntry(1, 3, 0.1),
      MatrixEntry(2, 0, 0.2), MatrixEntry(2, 1, 0.1), MatrixEntry(2, 2, 10.0), MatrixEntry(2, 3, 0.2),
      MatrixEntry(3, 0, 0.3), MatrixEntry(3, 1, 0.2), MatrixEntry(3, 2, 0.4), MatrixEntry(3, 3, 10.0)
    ))
    new CoordinateMatrix(entries, 4, 4)
  }
}
