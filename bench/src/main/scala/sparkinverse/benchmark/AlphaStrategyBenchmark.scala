package sparkinverse.benchmark

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import sparkinverse.api.{AlphaStrategy, IterativeInverseConfig}
import sparkinverse.syntax.block._

import java.nio.file.Files

/** Standalone benchmark for iterative inverse alpha strategies.
  *
  * This intentionally lives in the `bench` module instead of the core test
  * suite because the larger cases run real Spark matrix multiplications and
  * can take minutes on local[*].
  *
  * Run locally:
  *   sbt "bench/runMain sparkinverse.benchmark.AlphaStrategyBenchmark"
  *
  * Select sizes/orders:
  *   sbt "bench/runMain sparkinverse.benchmark.AlphaStrategyBenchmark --sizes 100,500,1000 --orders 2,3"
  */
object AlphaStrategyBenchmark {
  final case class Config(
    sizes: Seq[Int] = Seq(100, 500, 1000, 2000),
    orders: Seq[Int] = Seq(2, 3),
    blockSize: Int = 100,
    midSplits: Int = 4,
    maxIterOrder2: Int = 100,
    maxIterOrder3: Int = 50,
    tolerance: Double = 1e-12,
    moderate: Boolean = false
  )

  private val strategies = Seq(
    "NormProduct" -> AlphaStrategy.NormProduct,
    "Frobenius"   -> AlphaStrategy.Frobenius,
    "PowerIter3"  -> AlphaStrategy.PowerIteration(3),
    "Adaptive"    -> AlphaStrategy.Adaptive
  )

  def main(args: Array[String]): Unit = {
    val config = parseArgs(args)
    val conf = new SparkConf()
      .setAppName("sparkInverse-alpha-strategy-benchmark")
      .setIfMissing("spark.master", "local[*]")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    sc.setCheckpointDir(Files.createTempDirectory("sparkInverse-alpha-bench-checkpoints").toString)

    try {
      println("=" * 96)
      println("sparkInverse alpha strategy benchmark")
      println(s"sizes=${config.sizes.mkString(",")}, orders=${config.orders.mkString(",")}, blockSize=${config.blockSize}, midSplits=${config.midSplits}")
      println("=" * 96)

      for {
        n <- config.sizes
        order <- config.orders
      } runCase(sc, config, n, order)
    } finally {
      sc.stop()
    }
  }

  private def parseArgs(args: Array[String]): Config = {
    val map = args.sliding(2, 2).collect { case Array(k, v) if k.startsWith("--") => k.drop(2) -> v }.toMap
    var c = Config()
    c = c.copy(
      sizes = map.get("sizes").map(_.split(",").filter(_.nonEmpty).map(_.toInt).toSeq).getOrElse(c.sizes),
      orders = map.get("orders").map(_.split(",").filter(_.nonEmpty).map(_.toInt).toSeq).getOrElse(c.orders),
      blockSize = map.get("block-size").map(_.toInt).getOrElse(c.blockSize),
      midSplits = map.get("mid-splits").map(_.toInt).getOrElse(c.midSplits),
      maxIterOrder2 = map.get("max-iter-order2").map(_.toInt).getOrElse(c.maxIterOrder2),
      maxIterOrder3 = map.get("max-iter-order3").map(_.toInt).getOrElse(c.maxIterOrder3),
      tolerance = map.get("tolerance").map(_.toDouble).getOrElse(c.tolerance),
      moderate = map.get("matrix").contains("moderate")
    )
    c
  }

  private def runCase(sc: SparkContext, config: Config, n: Int, order: Int): Unit = {
    val blockSize = math.min(config.blockSize, n)
    val matrixKind = if (config.moderate) "moderate-conditioned" else "well-conditioned"
    val mat = if (config.moderate) moderateConditionMatrix(sc, n, blockSize) else largeDenseDiagonalMatrix(sc, n, blockSize)
    try {
      println()
      println(s"Alpha benchmark: n=$n $matrixKind, order=$order")
      println(f"  ${"Strategy"}%12s | ${"Time(ms)"}%10s | ${"RMSE"}%12s")
      println("  " + "-" * 42)

      for ((name, strategy) <- strategies) {
        val maxIter = if (order == 2) config.maxIterOrder2 else config.maxIterOrder3
        val invConfig = IterativeInverseConfig(
          order = order,
          maxIter = maxIter,
          tolerance = config.tolerance,
          useCheckpoints = true,
          midSplits = config.midSplits,
          alphaStrategy = strategy
        )
        val (elapsedMs, rmse) = measure(mat, invConfig, config.midSplits)
        println(f"  $name%12s | $elapsedMs%10.0f | $rmse%12.6e")
      }
    } finally {
      mat.blocks.unpersist(true)
    }
  }

  private def measure(mat: BlockMatrix, config: IterativeInverseConfig, midSplits: Int): (Double, Double) = {
    val t0 = System.nanoTime()
    val inv = mat.iterativeInverse(config)
    inv.blocks.persist(StorageLevel.MEMORY_AND_DISK_SER)
    inv.blocks.count()
    val elapsedMs = (System.nanoTime() - t0) / 1e6

    val product = mat.multiply(inv, midSplits)
    product.blocks.persist(StorageLevel.MEMORY_AND_DISK_SER)
    val rmse = identityErrorRmse(product)

    product.blocks.unpersist(true)
    inv.blocks.unpersist(true)
    (elapsedMs, rmse)
  }

  private def identityErrorRmse(product: BlockMatrix): Double = {
    val rpb = product.rowsPerBlock
    val cpb = product.colsPerBlock
    val errSq = product.blocks.map { case ((bi, bj), mat) =>
      val arr = mat.toArray
      val rows = mat.numRows
      val cols = mat.numCols
      var sum = 0.0
      var c = 0
      while (c < cols) {
        var r = 0
        while (r < rows) {
          val globalRow = bi * rpb + r
          val globalCol = bj * cpb + c
          val expected = if (globalRow == globalCol) 1.0 else 0.0
          val e = arr(r + c * rows) - expected
          sum += e * e
          r += 1
        }
        c += 1
      }
      sum
    }.sum()
    math.sqrt(errSq) / product.numRows()
  }

  /** Large diagonally dominant matrix: diag≈10-15, off-diag≈0.15. */
  private def largeDenseDiagonalMatrix(sc: SparkContext, n: Int, blockSize: Int): BlockMatrix = {
    val rng = new scala.util.Random(42)
    val numBlocks = (n + blockSize - 1) / blockSize
    val blocks = (for {
      bi <- 0 until numBlocks
      bj <- 0 until numBlocks
    } yield block(sc, rng, n, blockSize, bi, bj, diagonal = (r: Int) => 10.0 + rng.nextDouble() * 5.0, offDiagonal = () => 0.3 * rng.nextDouble())).toSeq
    materialize(new BlockMatrix(sc.parallelize(blocks, 4), blockSize, blockSize))
  }

  /** Moderately conditioned dense matrix: diag≈2-5, off-diag≈0.5-1.5. */
  private def moderateConditionMatrix(sc: SparkContext, n: Int, blockSize: Int): BlockMatrix = {
    val rng = new scala.util.Random(123)
    val numBlocks = (n + blockSize - 1) / blockSize
    val blocks = (for {
      bi <- 0 until numBlocks
      bj <- 0 until numBlocks
    } yield block(sc, rng, n, blockSize, bi, bj, diagonal = (r: Int) => 2.0 + rng.nextDouble() * 3.0, offDiagonal = () => 0.5 + rng.nextDouble())).toSeq
    materialize(new BlockMatrix(sc.parallelize(blocks, 4), blockSize, blockSize))
  }

  private def block(
    sc: SparkContext,
    rng: scala.util.Random,
    n: Int,
    blockSize: Int,
    bi: Int,
    bj: Int,
    diagonal: Int => Double,
    offDiagonal: () => Double
  ): ((Int, Int), DenseMatrix) = {
    val startRow = bi * blockSize
    val startCol = bj * blockSize
    val rows = math.min(blockSize, n - startRow)
    val cols = math.min(blockSize, n - startCol)
    val data = new Array[Double](rows * cols)
    var idx = 0
    for (r <- 0 until rows; c <- 0 until cols) {
      val globalRow = startRow + r
      val globalCol = startCol + c
      data(idx) = if (globalRow == globalCol) diagonal(globalRow) else offDiagonal()
      idx += 1
    }
    ((bi, bj), new DenseMatrix(rows, cols, data))
  }

  private def materialize(mat: BlockMatrix): BlockMatrix = {
    mat.blocks.persist(StorageLevel.MEMORY_AND_DISK_SER)
    mat.blocks.count()
    mat
  }
}
