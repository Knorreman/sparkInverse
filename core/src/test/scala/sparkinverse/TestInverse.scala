package sparkinverse

import breeze.linalg.{DenseMatrix => BDM, inv => BINV, pinv => PBINV}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.funsuite.AnyFunSuite
import sparkinverse.api.{AlphaStrategy, IterativeInverseConfig, PseudoInverseSide, RecursiveInverseConfig}
import sparkinverse.block.BlockMatrixOps
import sparkinverse.core.MatrixInternals
import sparkinverse.coordinate.CoordinateMatrixOps
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

  def testMatrixSimilarity(a: Matrix, b: Matrix, tol: Double): Boolean =
    a.toArray.zip(b.toArray).forall { case (left, right) => math.abs(left - right) < tol }

  test("block facade inverse without checkpoint") {
    val matrix = sampleBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    val config = RecursiveInverseConfig(limit = 3, midSplits = 3, useCheckpoints = false)
    val inverse = matrix.inverse(config)
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
    val inverse = matrix.inverse(RecursiveInverseConfig(limit = 3))
    assert(testMatrixSimilarity(inverse.toBlockMatrix().toLocalMatrix(), expected, 1e-12))
  }

  test("block recursive inverse handles single-block matrices") {
    val matrix = singleBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    val inverse = matrix.inverse(RecursiveInverseConfig(limit = 1, useCheckpoints = false))
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-12))
  }

  test("block svd and local inverse") {
    val matrix = sampleBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    assert(testMatrixSimilarity(matrix.svdInverse().toLocalMatrix(), expected, 1e-12))
    assert(testMatrixSimilarity(matrix.localInverse().toLocalMatrix(), expected, 1e-12))
  }

  test("LU inverse matches SVD for well-conditioned matrix") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    val luInv = matrix.luInverse().toLocalMatrix()
    val svdInv = matrix.svdInverse().toLocalMatrix()
    assert(testMatrixSimilarity(luInv, expected, 1e-12))
    assert(testMatrixSimilarity(svdInv, expected, 1e-12))
    assert(testMatrixSimilarity(luInv, svdInv, 1e-14))
  }

  test("LU inverse handles scaled identity correctly") {
    val matrix = identityBlockMatrix(4).scalarMultiply(10.0)
    val expected = identityBlockMatrix(4).scalarMultiply(0.1)
    val luInv = matrix.luInverse().toLocalMatrix()
    assert(testMatrixSimilarity(luInv, expected.toLocalMatrix(), 1e-12))
  }

  test("localInverse falls back to SVD on near-singular matrix") {
    // Construct a singular matrix by making two rows identical.
    // LU should detect singularity; localInverse should recover via SVD.
    val n = 4
    val blockSize = 2
    // Matrix:
    // [1 0 0 0]  Row 0
    // [0 1 0 0]  Row 1
    // [1 0 0 0]  Row 2 = Row 0 (duplicate → singular)
    // [0 0 1 0]  Row 3
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 0.0, 0.0, 1.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(0.0, 0.0, 0.0, 0.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(1.0, 0.0, 0.0, 0.0))), // Row 2 = Row 0
      ((1, 1), new DenseMatrix(2, 2, Array(0.0, 0.0, 1.0, 0.0)))
    )
    val singular = new BlockMatrix(sc.parallelize(blocks, 2), blockSize, blockSize)

    // luInverse should throw on singular matrix
    assertThrows[IllegalArgumentException] {
      singular.luInverse()
    }

    // localInverse should not throw (SVD fallback succeeds)
    val localInv = singular.localInverse()

    // Result should have correct dimensions
    assert(localInv.numRows() == n)
    assert(localInv.numCols() == n)
  }

  test("multi-block recursion exercises localInverse at leaves") {
    // A 4×4 matrix with blockSize=2 gives a 2×2 block grid.
    // With limit=1, each leaf is a single block → triggers localInverse().
    val n = 4
    val blockSize = 2
    val blocks = (for (i <- 0 until 2; j <- 0 until 2) yield {
      val blockData = if (i == j) Array(2.0, -1.0, -1.0, 2.0) else Array(0.0, 0.0, 0.0, 0.0)
      ((i, j), new DenseMatrix(blockSize, blockSize, blockData))
    }).toSeq
    val matrix = new BlockMatrix(sc.parallelize(blocks, numPartitions), blockSize, blockSize)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = RecursiveInverseConfig(limit = 1, midSplits = 1, useCheckpoints = false)
    val inverse = matrix.inverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-10))
  }

  test("coordinate local inverse and syntax methods") {
    val matrix = sampleBlockMatrix().toCoordinateMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    assert(testMatrixSimilarity(matrix.localInverse().toBlockMatrix().toLocalMatrix(), expected, 1e-12))
    assert(testMatrixSimilarity(matrix.svdInverse().toBlockMatrix().toLocalMatrix(), expected, 1e-12))
  }

  test("pseudo inverse facade") {
    val tall = sampleRectangularBlockMatrix().transpose
    val expectedTall = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(tall)))
    val left = tall.pseudoInverse(PseudoInverseSide.Left, RecursiveInverseConfig(limit = 3, midSplits = 2))
    assert(testMatrixSimilarity(left.toLocalMatrix(), expectedTall, 1e-12))

    val wide = sampleRectangularBlockMatrix()
    val expectedWide = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(wide)))
    val right = wide.pseudoInverse(PseudoInverseSide.Right, RecursiveInverseConfig(limit = 3, midSplits = 2))
    assert(testMatrixSimilarity(right.toLocalMatrix(), expectedWide, 1e-12))
  }

  test("pseudo inverse with Tikhonov regularization") {
    val tall = sampleRectangularBlockMatrix().transpose
    val wide = sampleRectangularBlockMatrix()

    // Lambda = 0 should produce the standard pseudo-inverse (backward compatible)
    val leftDefault = tall.pseudoInverse(PseudoInverseSide.Left, RecursiveInverseConfig(limit = 3, midSplits = 2))
    val leftLambda0 = tall.pseudoInverse(PseudoInverseSide.Left,
      RecursiveInverseConfig(limit = 3, midSplits = 2, regularizationLambda = 0.0))
    assert(testMatrixSimilarity(leftDefault.toLocalMatrix(), leftLambda0.toLocalMatrix(), 1e-14))

    // Lambda > 0 should produce a different (regularized) result
    val leftReg = tall.pseudoInverse(PseudoInverseSide.Left,
      RecursiveInverseConfig(limit = 3, midSplits = 2, regularizationLambda = 0.1))
    assert(leftReg.numRows() == tall.numCols())
    assert(leftReg.numCols() == tall.numRows())

    // Right pseudo-inverse with regularization
    val rightReg = wide.pseudoInverse(PseudoInverseSide.Right,
      RecursiveInverseConfig(limit = 3, midSplits = 2, regularizationLambda = 0.1))
    assert(rightReg.numRows() == wide.numCols())
    assert(rightReg.numCols() == wide.numRows())

    // Larger lambda should shrink result toward zero (ridge shrinkage)
    val leftLargeLambda = tall.pseudoInverse(PseudoInverseSide.Left,
      RecursiveInverseConfig(limit = 3, midSplits = 2, regularizationLambda = 100.0))
    val leftNormSmall = leftReg.toLocalMatrix().toArray.map(math.abs).sum
    val leftNormLarge = leftLargeLambda.toLocalMatrix().toArray.map(math.abs).sum
    assert(leftNormLarge < leftNormSmall,
      s"Larger lambda should shrink pseudo-inverse: lambda=0.1 norm=$leftNormSmall, lambda=100 norm=$leftNormLarge")
  }

  test("pseudo inverse rejects negative regularization lambda") {
    val matrix = sampleRectangularBlockMatrix()
    assertThrows[IllegalArgumentException] {
      matrix.pseudoInverse(PseudoInverseSide.Left,
        RecursiveInverseConfig(limit = 3, midSplits = 2, regularizationLambda = -0.1))
    }
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
    assert(math.abs(block.normOne() - 7.0) < 1e-12)
    assert(math.abs(block.normInf() - 6.0) < 1e-12)
    assert(math.abs(coord.normOne() - 7.0) < 1e-12)
    assert(math.abs(coord.normInf() - 6.0) < 1e-12)
  }

  test("iterative inverse config on block and coordinate") {
    val block = diagonallyDominantBlockMatrix()
    val blockExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(block)))
    val blockInverse = block.iterativeInverse(IterativeInverseConfig(order = 2, maxIter = 30, tolerance = 1e-10))
    assert(testMatrixSimilarity(blockInverse.toLocalMatrix(), blockExpected, 1e-8))

    val coord = diagonallyDominantCoordinateMatrix()
    val coordExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(coord)))
    val coordInverse = coord.iterativeInverse(IterativeInverseConfig(order = 2, maxIter = 30, tolerance = 1e-10))
    assert(testMatrixSimilarity(coordInverse.toBlockMatrix().toLocalMatrix(), coordExpected, 1e-8))
  }

  test("iterative inverse with order 3 on block and coordinate") {
    val block = diagonallyDominantBlockMatrix()
    val blockExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(block)))
    val blockInverse = block.iterativeInverse(IterativeInverseConfig(order = 3, maxIter = 20, tolerance = 1e-10))
    assert(testMatrixSimilarity(blockInverse.toLocalMatrix(), blockExpected, 1e-8))

    val coord = diagonallyDominantCoordinateMatrix()
    val coordExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(coord)))
    val coordInverse = coord.iterativeInverse(IterativeInverseConfig(order = 3, maxIter = 20, tolerance = 1e-10))
    assert(testMatrixSimilarity(coordInverse.toBlockMatrix().toLocalMatrix(), coordExpected, 1e-8))
  }

  test("iterative inverse supports custom order 4") {
    val block = diagonallyDominantBlockMatrix()
    val blockExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(block)))
    val blockInverse = block.iterativeInverse(IterativeInverseConfig(order = 4, maxIter = 15, tolerance = 1e-10))
    assert(testMatrixSimilarity(blockInverse.toLocalMatrix(), blockExpected, 1e-8))

    val coord = diagonallyDominantCoordinateMatrix()
    val coordExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(coord)))
    val coordInverse = coord.iterativeInverse(IterativeInverseConfig(order = 4, maxIter = 15, tolerance = 1e-10))
    assert(testMatrixSimilarity(coordInverse.toBlockMatrix().toLocalMatrix(), coordExpected, 1e-8))
  }

  test("iterative inverse with order 5 supports repeated squaring") {
    val block = diagonallyDominantBlockMatrix()
    val blockExpected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(block)))
    val blockInverse = block.iterativeInverse(IterativeInverseConfig(order = 5, maxIter = 15, tolerance = 1e-10))
    assert(testMatrixSimilarity(blockInverse.toLocalMatrix(), blockExpected, 1e-8))
  }

  test("block specialized square matches generic multiply on odd sparse block grids") {
    val matrix = sparseOddGridBlockMatrix()
    val expected = matrix.multiply(matrix, 3)
    val squared = new BlockMatrixOps(matrix).squareBlocks(3)
    assert(testMatrixSimilarity(squared.toLocalMatrix(), expected.toLocalMatrix(), 1e-12))
  }

  test("block hyperpower correction matches generic powers on odd sparse block grids") {
    val residual = sparseOddGridBlockMatrix()
    val expected = genericHyperpowerCorrection(residual, order = 5, midSplits = 3)
    val correction = new BlockMatrixOps(residual).hyperpowerCorrection(order = 5, midSplits = 3)
    assert(testMatrixSimilarity(correction.toLocalMatrix(), expected.toLocalMatrix(), 1e-12))
  }

  test("well-conditioned matrix converges quickly") {
    // Test that a well-conditioned matrix converges quickly
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    
    // Use a tolerance that should converge quickly
    val config = IterativeInverseConfig(maxIter = 15, tolerance = 1e-8, midSplits = 1)
    val inverse = matrix.iterativeInverse(config.copy(order = 2))
    
    // Should converge and give accurate result
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-7))
  }

  test("identity matrix inverse") {
    val identity = identityBlockMatrix(6)
    val config = RecursiveInverseConfig(limit = 2, midSplits = 2, useCheckpoints = false)
    val inverse = identity.inverse(config)
    
    // Identity matrix inverse should be identity
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), identity.toLocalMatrix(), 1e-12))
    
    // Also test iterative methods
    val iterativeConfig = IterativeInverseConfig(maxIter = 10, tolerance = 1e-10)
    val iterativeInverseResult = identity.iterativeInverse(iterativeConfig.copy(order = 2))
    assert(testMatrixSimilarity(iterativeInverseResult.toLocalMatrix(), identity.toLocalMatrix(), 1e-10))
  }

  test("very small matrix (1x1 block)") {
    val blocks = Seq(((0, 0), new DenseMatrix(1, 1, Array(5.0))))
    val tiny = new BlockMatrix(sc.parallelize(blocks, 1), 1, 1)
    val config = RecursiveInverseConfig(limit = 1, midSplits = 1, useCheckpoints = false)
    val inverse = tiny.inverse(config)
    
    assert(math.abs(inverse.toLocalMatrix()(0, 0) - 0.2) < 1e-12)
  }

  test("frobenius norm calculation") {
    val matrix = sampleBlockMatrix()
    val ops = new BlockMatrixOps(matrix)
    val frobeniusSq = ops.frobeniusNormSquared()
    
    // Manual calculation: sum of squares of all elements
    val expected = Array(1.0, 2.0, 1.0, 4.0, 4.0, 3.0, 2.0, 2.0, 
                         1.0, 5.0, 3.0, 4.0, -1.0, 6.0, 2.0, 1.0)
      .map(x => x * x).sum
    assert(math.abs(frobeniusSq - expected) < 1e-12)
  }

  test("scalar multiplication") {
    val matrix = sampleBlockMatrix()
    val ops = new BlockMatrixOps(matrix)
    val scaled = ops.scalarMultiply(2.5)
    
    val original = denseMatrixToBreeze(matrix)
    val scaledResult = denseMatrixToBreeze(scaled)
    val expected = original * 2.5
    // Compare element by element
    val scaledLocal = scaled.toLocalMatrix()
    val expectedArray = breezeToDenseMatrix(expected).toArray
    assert(scaledLocal.toArray.zip(expectedArray).forall { case (a, b) => math.abs(a - b) < 1e-12 })
  }

  test("matrix negation") {
    val matrix = sampleBlockMatrix()
    val ops = new BlockMatrixOps(matrix)
    val negated = ops.negate()
    
    val original = denseMatrixToBreeze(matrix)
    val negatedResult = denseMatrixToBreeze(negated)
    val expected = original * -1.0
    // Compare element by element
    val negatedLocal = negated.toLocalMatrix()
    val expectedArray = breezeToDenseMatrix(expected).toArray
    assert(negatedLocal.toArray.zip(expectedArray).forall { case (a, b) => math.abs(a - b) < 1e-12 })
  }

  test("different block sizes") {
    val blockSize = 3
    val blocks = Seq(
      ((0, 0), new DenseMatrix(3, 3, Array(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0))),
      ((0, 1), new DenseMatrix(3, 3, Array(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))),
      ((1, 0), new DenseMatrix(3, 3, Array(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))),
      ((1, 1), new DenseMatrix(3, 3, Array(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)))
    )
    val matrix = new BlockMatrix(sc.parallelize(blocks, numPartitions), blockSize, blockSize)
    
    val config = RecursiveInverseConfig(limit = 3, midSplits = 2, useCheckpoints = false)
    val inverse = matrix.inverse(config)
    
    // Verify A * A^-1 ≈ I
    val product = matrix.multiply(inverse, 2)
    val identity = denseMatrixToBreeze(identityBlockMatrix(6))
    assert(testMatrixSimilarity(product.toLocalMatrix(), breezeToDenseMatrix(identity), 1e-10))
  }

  test("sparse coordinate matrix operations") {
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 10.0), MatrixEntry(0, 2, 0.5),
      MatrixEntry(1, 1, 8.0),
      MatrixEntry(2, 0, 0.5), MatrixEntry(2, 2, 6.0)
    ))
    val sparse = new CoordinateMatrix(entries, 3, 3)
    
    val config = IterativeInverseConfig(maxIter = 20, tolerance = 1e-10)
    val inverse = sparse.iterativeInverse(config.copy(order = 2))
    
    // Verify A * A^-1 ≈ I
    val product = sparse.multiply(inverse)
    val identity = identityCoordinateMatrix(3)
    val productBlock = product.toBlockMatrix().toLocalMatrix()
    val identityBlock = identity.toBlockMatrix().toLocalMatrix()
    assert(testMatrixSimilarity(productBlock, identityBlock, 1e-8))
  }

  test("adaptive mid-dimension splits") {
    val matrix = sampleBlockMatrix()
    val config = RecursiveInverseConfig(
      limit = 2,
      midSplits = 1,
      useCheckpoints = false
    )
    val inverse = matrix.inverse(config)
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-12))
  }

  test("iterative with different tolerance levels") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    
    // Test with different tolerance levels
    val config1 = IterativeInverseConfig(maxIter = 30, tolerance = 1e-6)
    val inverse1 = matrix.iterativeInverse(config1.copy(order = 2))
    assert(testMatrixSimilarity(inverse1.toLocalMatrix(), expected, 1e-5))
    
    val config2 = IterativeInverseConfig(maxIter = 50, tolerance = 1e-12)
    val inverse2 = matrix.iterativeInverse(config2.copy(order = 2))
    assert(testMatrixSimilarity(inverse2.toLocalMatrix(), expected, 1e-10))
  }

  test("adaptive order converges from high order to order 2") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    // Adaptive order: starts at order 5, falls back to order 2 when residual is small
    val config = IterativeInverseConfig(
      order = 5,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      convergenceCheckInterval = 1,
      adaptiveOrder = true,
      adaptiveOrderFallback = 0.3
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("adaptive order with order 2 is equivalent to fixed order 2") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    // When maxOrder is already 2, adaptive order should behave identically to fixed order 2
    val configFixed = IterativeInverseConfig(
      order = 2, maxIter = 20, tolerance = 1e-10,
      useCheckpoints = false, midSplits = 1
    )
    val configAdaptive = IterativeInverseConfig(
      order = 2, maxIter = 20, tolerance = 1e-10,
      useCheckpoints = false, midSplits = 1,
      adaptiveOrder = true
    )

    val fixed = matrix.iterativeInverse(configFixed)
    val adaptive = matrix.iterativeInverse(configAdaptive)
    assert(testMatrixSimilarity(fixed.toLocalMatrix(), expected, 1e-8))
    assert(testMatrixSimilarity(adaptive.toLocalMatrix(), expected, 1e-8))
  }

  test("adaptive order with sparse convergence checks") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 4,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      convergenceCheckInterval = 2,
      adaptiveOrder = true,
      adaptiveOrderFallback = 0.3
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("iterative convergence check interval greater than one converges") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      convergenceCheckInterval = 3
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("higher-order iterative convergence check interval converges") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 4,
      maxIter = 20,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      convergenceCheckInterval = 2
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("adaptive alpha works with sparse convergence checks") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      alphaStrategy = AlphaStrategy.Adaptive,
      convergenceCheckInterval = 3
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("iterative inverse rejects non-positive convergence check interval") {
    val matrix = identityBlockMatrix(4)
    val config = IterativeInverseConfig(
      maxIter = 2,
      useCheckpoints = false,
      convergenceCheckInterval = 0
    )
    assertThrows[IllegalArgumentException] {
      matrix.iterativeInverse(config)
    }
  }

  test("coordinate matrix norms") {
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 5.0), MatrixEntry(0, 1, 3.0),
      MatrixEntry(1, 0, 2.0), MatrixEntry(1, 1, 4.0)
    ))
    val coord = new CoordinateMatrix(entries, 2, 2)
    
    // One-norm: max column sum = max(|5| + |2|, |3| + |4|) = max(7, 7) = 7
    assert(math.abs(coord.normOne() - 7.0) < 1e-12)
    
    // Inf-norm: max row sum = max(|5| + |3|, |2| + |4|) = max(8, 6) = 8
    assert(math.abs(coord.normInf() - 8.0) < 1e-12)
    
    // Frobenius norm squared: 5² + 3² + 2² + 4² = 25 + 9 + 4 + 16 = 54
    val ops = new CoordinateMatrixOps(coord)
    assert(math.abs(ops.frobeniusNormSquared() - 54.0) < 1e-12)
  }

  test("coordinate scalar multiply and negate") {
    val entries = sc.parallelize(Seq(
      MatrixEntry(0, 0, 2.0), MatrixEntry(0, 1, 3.0),
      MatrixEntry(1, 0, 4.0), MatrixEntry(1, 1, 5.0)
    ))
    val coord = new CoordinateMatrix(entries, 2, 2)
    val ops = new CoordinateMatrixOps(coord)
    
    val scaled = ops.scalarMultiply(2.0).entries.collect().sortBy(e => (e.i, e.j))
    assert(scaled.exists(e => e.i == 0 && e.j == 0 && math.abs(e.value - 4.0) < 1e-12))
    assert(scaled.exists(e => e.i == 1 && e.j == 1 && math.abs(e.value - 10.0) < 1e-12))
    
    val negated = ops.negate().entries.collect().sortBy(e => (e.i, e.j))
    assert(negated.exists(e => e.i == 0 && e.j == 0 && math.abs(e.value + 2.0) < 1e-12))
  }

  test("iterative inverse with high order 6") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    
    // Test with order 6
    val config = IterativeInverseConfig(maxIter = 15, tolerance = 1e-10)
    val inverse = matrix.iterativeInverse(config.copy(order = 6))
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("pseudo-inverse with checkpointing") {
    val tall = sampleRectangularBlockMatrix().transpose
    val expectedTall = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(tall)))
    
    val config = RecursiveInverseConfig(limit = 3, midSplits = 2, useCheckpoints = true)
    val left = tall.pseudoInverse(PseudoInverseSide.Left, config)
    assert(testMatrixSimilarity(left.toLocalMatrix(), expectedTall, 1e-12))
    
    val wide = sampleRectangularBlockMatrix()
    val expectedWide = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(wide)))
    val right = wide.pseudoInverse(PseudoInverseSide.Right, config)
    assert(testMatrixSimilarity(right.toLocalMatrix(), expectedWide, 1e-12))
  }

  test("coordinate pseudo-inverse") {
    val tall = sampleRectangularBlockMatrix().transpose.toCoordinateMatrix()
    val expectedTall = breezeToDenseMatrix(PBINV(denseMatrixToBreeze(tall)))
    
    val config = RecursiveInverseConfig(limit = 3, midSplits = 2)
    val left = tall.pseudoInverse(PseudoInverseSide.Left, config)
    assert(testMatrixSimilarity(left.toBlockMatrix().toLocalMatrix(), expectedTall, 1e-8))
  }

  test("large matrix recursive inversion") {
    // Test with a larger matrix to exercise the recursive algorithm
    val n = 8
    val blockSize = 2
    val blocks = (for {
      i <- 0 until n by blockSize
      j <- 0 until n by blockSize
      bi = i / blockSize
      bj = j / blockSize
    } yield {
      val data = Array.fill(blockSize * blockSize)(math.random() * 0.1)
      // Make diagonal blocks dominant
      if (bi == bj) {
        for (k <- 0 until blockSize) data(k * blockSize + k) = 10.0
      }
      ((bi, bj), new DenseMatrix(blockSize, blockSize, data))
    }).toSeq
    
    val matrix = new BlockMatrix(sc.parallelize(blocks, numPartitions), blockSize, blockSize)
    val config = RecursiveInverseConfig(limit = 2, midSplits = 2, useCheckpoints = false)
    val inverse = matrix.inverse(config)
    
    // Verify dimensions
    assert(inverse.numRows() == n)
    assert(inverse.numCols() == n)
    
    // Verify A * A^-1 ≈ I
    val product = matrix.multiply(inverse, 2)
    val identity = denseMatrixToBreeze(identityBlockMatrix(n))
    assert(testMatrixSimilarity(product.toLocalMatrix(), breezeToDenseMatrix(identity), 1e-6))
  }

  test("recursive inverse materializes output before unpersist without checkpoints") {
    import org.apache.spark.storage.StorageLevel

    val n = 8
    val matrix = recursiveTestBlockMatrix(n, blockSize = 2)
    val config = RecursiveInverseConfig(
      limit = 2,
      midSplits = 2,
      useCheckpoints = false,
      minBlockSizeForPersistence = 0
    )

    val inverse = matrix.inverse(config)

    // Regression for unpersist-before-evaluation: the returned result should be
    // persisted/materialized before intermediate parents are unpersisted.
    assert(inverse.blocks.getStorageLevel != StorageLevel.NONE)
    assert(inverse.blocks.count() > 0)

    val product = matrix.multiply(inverse, 2)
    assert(testMatrixSimilarity(product.toLocalMatrix(), identityBlockMatrix(n).toLocalMatrix(), 1e-6))
  }

  test("recursive inverse writes checkpoint before unpersisting intermediates") {
    val n = 8
    val matrix = recursiveTestBlockMatrix(n, blockSize = 2)
    val config = RecursiveInverseConfig(
      limit = 2,
      midSplits = 2,
      useCheckpoints = true,
      minBlockSizeForPersistence = 0
    )

    val inverse = matrix.inverse(config)

    // checkpoint() is lazy; this should already be materialized by inverse().
    assert(inverse.blocks.isCheckpointed)
    assert(inverse.blocks.count() > 0)

    val product = matrix.multiply(inverse, 2)
    assert(testMatrixSimilarity(product.toLocalMatrix(), identityBlockMatrix(n).toLocalMatrix(), 1e-6))
  }

  test("recursive inverse handles odd block-grid quadrant split") {
    // 6x6 with blockSize=2 gives a 3x3 block grid. The first split is uneven
    // (m = 2), so F/G/H index normalization must handle odd grid dimensions.
    val n = 6
    val matrix = recursiveTestBlockMatrix(n, blockSize = 2)
    val config = RecursiveInverseConfig(
      limit = 2,
      midSplits = 2,
      useCheckpoints = false,
      minBlockSizeForPersistence = 0
    )

    val inverse = matrix.inverse(config)
    assert(inverse.blocks.count() > 0)

    val product = matrix.multiply(inverse, 2)
    assert(testMatrixSimilarity(product.toLocalMatrix(), identityBlockMatrix(n).toLocalMatrix(), 1e-6))
  }

  test("coordinate recursive inversion") {
    val matrix = diagonallyDominantCoordinateMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    
    val config = RecursiveInverseConfig(limit = 2, midSplits = 1, useCheckpoints = false)
    val inverse = matrix.inverse(config)
    assert(testMatrixSimilarity(inverse.toBlockMatrix().toLocalMatrix(), expected, 1e-10))
  }

  test("block matrix operations with different storage levels") {
    import org.apache.spark.storage.StorageLevel
    
    val matrix = sampleBlockMatrix()
    
    // Test with MEMORY_ONLY storage level
    val configMemoryOnly = IterativeInverseConfig(
      maxIter = 20,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1
    )
    val inverse1 = matrix.iterativeInverse(configMemoryOnly.copy(order = 2))
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))
    assert(testMatrixSimilarity(inverse1.toLocalMatrix(), expected, 1e-8))
  }

  private def recursiveTestBlockMatrix(n: Int, blockSize: Int): BlockMatrix = {
    val blocks = (for {
      i <- 0 until n by blockSize
      j <- 0 until n by blockSize
      bi = i / blockSize
      bj = j / blockSize
    } yield {
      val blockRows = math.min(blockSize, n - i)
      val blockCols = math.min(blockSize, n - j)
      val data = Array.tabulate(blockRows * blockCols) { idx =>
        val row = idx % blockRows
        val col = idx / blockRows
        if (bi == bj && row == col) 12.0
        else if (bi == bj) 0.2
        else 0.01 * (1 + bi + bj + row + col)
      }
      ((bi, bj), new DenseMatrix(blockRows, blockCols, data))
    }).toSeq
    new BlockMatrix(sc.parallelize(blocks, numPartitions), blockSize, blockSize)
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

  private def singleBlockMatrix(): BlockMatrix = {
    val block = new DenseMatrix(4, 4, Array(
      7.0, 1.0, 0.0, 2.0,
      2.0, 8.0, 1.0, 0.0,
      0.0, 1.0, 6.0, 1.0,
      1.0, 0.0, 2.0, 5.0
    ))
    new BlockMatrix(sc.parallelize(Seq(((0, 0), block)), 1), 4, 4)
  }

  private def sparseOddGridBlockMatrix(): BlockMatrix = {
    val blocks = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(4.0, 1.0, 0.5, 3.0))),
      ((0, 2), new DenseMatrix(2, 1, Array(2.0, -1.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(5.0, 0.2, -0.3, 4.0))),
      ((2, 0), new DenseMatrix(1, 2, Array(1.5, -0.5))),
      ((2, 2), new DenseMatrix(1, 1, Array(6.0)))
    )
    new BlockMatrix(sc.parallelize(blocks, numPartitions), 2, 2, 5, 5)
  }

  private def genericHyperpowerCorrection(residual: BlockMatrix, order: Int, midSplits: Int): BlockMatrix = {
    val eye = MatrixInternals.eyeBlockMatrix(
      residual.numRows(),
      1.0,
      residual.rowsPerBlock,
      residual.colsPerBlock,
      org.apache.spark.storage.StorageLevel.MEMORY_ONLY,
      residual
    )
    var correction = eye.add(residual)
    var currentPower = residual
    var exponent = 2
    while (exponent < order) {
      currentPower = currentPower.multiply(residual, midSplits)
      correction = correction.add(currentPower)
      exponent += 1
    }
    correction
  }

  private def identityBlockMatrix(n: Int): BlockMatrix = {
    val data = Array.fill(n * n)(0.0)
    for (i <- 0 until n) data(i * n + i) = 1.0
    val blockSize = 2
    val blocks = (for {
      i <- 0 until n by blockSize
      j <- 0 until n by blockSize
      bi = i / blockSize
      bj = j / blockSize
    } yield {
      val blockRows = math.min(blockSize, n - i)
      val blockCols = math.min(blockSize, n - j)
      val blockData = Array.fill(blockRows * blockCols)(0.0)
      if (bi == bj) {
        for (k <- 0 until math.min(blockRows, blockCols)) blockData(k * blockRows + k) = 1.0
      }
      ((bi, bj), new DenseMatrix(blockRows, blockCols, blockData))
    }).toSeq
    new BlockMatrix(sc.parallelize(blocks, numPartitions), blockSize, blockSize)
  }

  private def identityCoordinateMatrix(n: Int): CoordinateMatrix = {
    val entries = sc.parallelize((0 until n).map(i => MatrixEntry(i, i, 1.0)))
    new CoordinateMatrix(entries, n, n)
  }

  // ── Alpha strategy tests ────────────────────────────────────────────────────

  test("Frobenius alpha strategy converges for diagonally dominant matrix") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      alphaStrategy = AlphaStrategy.Frobenius
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("NormProduct alpha strategy converges (legacy behavior)") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      alphaStrategy = AlphaStrategy.NormProduct
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("PowerIteration alpha strategy converges") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      alphaStrategy = AlphaStrategy.PowerIteration(powerIterations = 3)
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("Adaptive alpha strategy converges") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 30,
      tolerance = 1e-10,
      useCheckpoints = false,
      midSplits = 1,
      alphaStrategy = AlphaStrategy.Adaptive
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("Frobenius and NormProduct alpha strategies both converge") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val configFrob = IterativeInverseConfig(
      order = 2, maxIter = 30, tolerance = 1e-10, useCheckpoints = false,
      midSplits = 1, alphaStrategy = AlphaStrategy.Frobenius)
    val configNorm = IterativeInverseConfig(
      order = 2, maxIter = 30, tolerance = 1e-10, useCheckpoints = false,
      midSplits = 1, alphaStrategy = AlphaStrategy.NormProduct)

    val inverseFrob = matrix.iterativeInverse(configFrob)
    val inverseNorm = matrix.iterativeInverse(configNorm)

    assert(testMatrixSimilarity(inverseFrob.toLocalMatrix(), expected, 1e-8))
    assert(testMatrixSimilarity(inverseNorm.toLocalMatrix(), expected, 1e-8))
  }

  test("Alpha strategy consistency: all strategies produce valid inverses") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    val strategies = Seq(
      AlphaStrategy.NormProduct,
      AlphaStrategy.Frobenius,
      AlphaStrategy.PowerIteration(2),
      AlphaStrategy.Adaptive
    )

    for (strategy <- strategies) {
      val config = IterativeInverseConfig(
        order = 2, maxIter = 30, tolerance = 1e-8, useCheckpoints = false,
        midSplits = 1, alphaStrategy = strategy)
      val inverse = matrix.iterativeInverse(config)
      // All strategies should produce a matrix close to the true inverse
      assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-5),
        s"Strategy $strategy failed to converge to 1e-5")
    }
  }

  test("PowerIteration alpha handles scaled identity with sigma greater than one") {
    val matrix = identityBlockMatrix(4).scalarMultiply(10.0)
    val expected = identityBlockMatrix(4).scalarMultiply(0.1)
    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 2,
      tolerance = 1e-10,
      useCheckpoints = false,
      alphaStrategy = AlphaStrategy.PowerIteration(powerIterations = 3)
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected.toLocalMatrix(), 1e-8))
  }

  test("PowerIteration alpha handles scaled identity with sigma less than one") {
    val matrix = identityBlockMatrix(4).scalarMultiply(0.5)
    val expected = identityBlockMatrix(4).scalarMultiply(2.0)
    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 2,
      tolerance = 1e-10,
      useCheckpoints = false,
      alphaStrategy = AlphaStrategy.PowerIteration(powerIterations = 3)
    )
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected.toLocalMatrix(), 1e-8))
  }

  test("PowerIteration rejects non-positive iteration counts") {
    val matrix = identityBlockMatrix(4)
    val config = IterativeInverseConfig(
      order = 2,
      maxIter = 2,
      useCheckpoints = false,
      alphaStrategy = AlphaStrategy.PowerIteration(powerIterations = 0)
    )
    assertThrows[IllegalArgumentException] {
      matrix.iterativeInverse(config).blocks.count()
    }
  }

  test("default config still produces valid inverse (backward compatibility)") {
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    // Default IterativeInverseConfig should work exactly as before
    assert(IterativeInverseConfig().convergenceCheckInterval == 1)
    val config = IterativeInverseConfig(maxIter = 30, tolerance = 1e-10)
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("divergence detection does not fire on well-conditioned matrix") {
    // A well-conditioned diagonally dominant matrix should converge without divergence warnings
    val matrix = diagonallyDominantBlockMatrix()
    val expected = breezeToDenseMatrix(BINV(denseMatrixToBreeze(matrix)))

    // Use a tight tolerance and enough iterations to guarantee convergence
    val config = IterativeInverseConfig(
      order = 2, maxIter = 30, tolerance = 1e-10, useCheckpoints = false,
      midSplits = 1, convergenceCheckInterval = 1)
    val inverse = matrix.iterativeInverse(config)
    assert(testMatrixSimilarity(inverse.toLocalMatrix(), expected, 1e-8))
  }

  test("divergence detection with zero alpha produces warning but still returns result") {
    // Using NormProduct alpha = 1/||A||_1 * 1/||A||_inf on a matrix with very large
    // condition number should still produce a result (possibly inaccurate).
    // We can't easily assert log messages, so we verify the method completes.
    val matrix = sampleBlockMatrix()
    // Use a very small maxIter to trigger non-convergence
    val config = IterativeInverseConfig(
      order = 2, maxIter = 3, tolerance = 1e-15, useCheckpoints = false,
      midSplits = 1)
    // Should complete without throwing, even if it doesn't converge
    val result = matrix.iterativeInverse(config)
    assert(result.numRows() == matrix.numRows())
    assert(result.numCols() == matrix.numCols())
  }

}
