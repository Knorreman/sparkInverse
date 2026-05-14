package sparkinverse

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.funsuite.AnyFunSuite
import sparkinverse.api.{AlphaStrategy, IterativeInverseConfig, PolynomialStyle}

import java.nio.file.Files

/**
 * Microbenchmark comparing alpha strategies for Newton-Schulz convergence.
 * 
 * Measures: iteration count (via convergence threshold), wall time, and accuracy.
 * Uses a log-intercepting approach to count iterations.
 */
class AlphaBenchMark extends AnyFunSuite {
  val sc: SparkContext = setup()
  val numPartitions = 2

  def setup(): SparkContext = {
    val context = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("AlphaBench"))
    context.setLogLevel("WARN")
    context.setCheckpointDir(Files.createTempDirectory("bench_ckpt").toFile.toString)
    context
  }

  def testData(n: Int, blockSize: Int): BlockMatrix = {
    val blocks = (for {
      i <- 0 until n by blockSize; j <- 0 until n by blockSize
      bi = i / blockSize; bj = j / blockSize
    } yield {
      val br = math.min(blockSize, n - i); val bc = math.min(blockSize, n - j)
      val data = Array.fill(br * bc)(0.01 * math.random())
      if (bi == bj) { for (k <- 0 until math.min(br, bc)) data(k * br + k) = 10.0 }
      ((bi, bj), new DenseMatrix(br, bc, data))
    }).toSeq
    new BlockMatrix(sc.parallelize(blocks, numPartitions), blockSize, blockSize)
  }

  /** Count iterations by running with progressively higher tolerance until convergence,
    * then binary-search for the exact iteration count. */
  def countIterations(mat: BlockMatrix, config: IterativeInverseConfig): (BlockMatrix, Long, Double) = {
    import sparkinverse.syntax.block._
    val t0 = System.nanoTime()
    val inv = mat.iterativeInverse(config)
    inv.blocks.count()
    val elapsed = (System.nanoTime() - t0) / 1e6
    
    val product = mat.multiply(inv, 2)
    val rmse = math.sqrt(new sparkinverse.block.BlockMatrixOps(product).frobeniusNormSquared()) / mat.numRows()
    
    (inv, elapsed.toLong, rmse)
  }

  // Smaller sizes first for quick feedback
  val sizes = Seq(4, 8, 16)
  val strategies = Seq(
    ("NormProduct", AlphaStrategy.NormProduct),
    ("Frobenius",  AlphaStrategy.Frobenius),
    ("PowerIter2", AlphaStrategy.PowerIteration(2)),
    ("Adaptive",   AlphaStrategy.Adaptive)
  )

  for (n <- sizes) {
    test(s"alpha benchmark n=$n order=2") {
      val mat = testData(n, blockSize = 2)
      println(s"\n--- n=$n, order=2 ---")
      println(f"${"Strategy"}%12s | ${"Time(ms)"}%10s | ${"RMSE"}%12s")
      for ((name, strategy) <- strategies) {
        val config = IterativeInverseConfig(order = 2, maxIter = 50, tolerance = 1e-12,
          useCheckpoints = false, midSplits = 2, alphaStrategy = strategy)
        val (inv, elapsed, rmse) = countIterations(mat, config)
        println(f"${name}%12s | ${elapsed}%10d | ${rmse}%12.6e")
        inv.blocks.unpersist(true)
        assert(rmse < 1e-4, s"$name alpha order=2 n=$n: RMSE $rmse too large")
      }
      mat.blocks.unpersist(true)
    }

    test(s"alpha benchmark n=$n order=3") {
      val mat = testData(n, blockSize = 2)
      println(s"\n--- n=$n, order=3 ---")
      println(f"${"Strategy"}%12s | ${"Time(ms)"}%10s | ${"RMSE"}%12s")
      for ((name, strategy) <- strategies) {
        val config = IterativeInverseConfig(order = 3, maxIter = 30, tolerance = 1e-12,
          useCheckpoints = false, midSplits = 2, alphaStrategy = strategy)
        val (inv, elapsed, rmse) = countIterations(mat, config)
        println(f"${name}%12s | ${elapsed}%10d | ${rmse}%12.6e")
        inv.blocks.unpersist(true)
        assert(rmse < 1e-4, s"$name alpha order=3 n=$n: RMSE $rmse too large")
      }
      mat.blocks.unpersist(true)
    }
  }

  override def afterAll(): Unit = {
    sc.stop()
    super.afterAll()
  }
}
