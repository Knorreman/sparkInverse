import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.{SparkConf, SparkContext}
import Inverse.{BlockMatrixInverse, CoordinateMatrixInverse}
import org.apache.spark.storage.StorageLevel

object Main {

  case class BenchResult(n: Int, schurBlockSize: Int, nsBlockSize: Int, schurLimit: Int,
                         schurMidSplits: Int, nsMidSplits: Int,
                         schurTimeSec: Double, schurRmse: Double,
                         iterTimeSec: Double, iterRmse: Double, iterIters: Int)

  /**
   * Build a well-conditioned n×n BlockMatrix from a sparse random base + n*I diagonal.
   * The diagonal dominance guarantees both algorithms converge well.
   */
  def buildMatrix(sc: SparkContext, n: Int, blockSize: Int, seed: Long): BlockMatrix = {
    val lnrdd = RandomRDDs.poissonVectorRDD(sc, mean = 0.01, numRows = n, numCols = n,
        seed = seed, numPartitions = 8)
      .zipWithIndex()
      .map(_.swap)
      .map(x => IndexedRow(x._1, x._2))
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
    lnrdd.count() // materialise

    val base = new IndexedRowMatrix(lnrdd).toCoordinateMatrix()
    val diag = new CoordinateMatrix(sc.range(0, n).map(i => MatrixEntry(i, i, n.toDouble)), n, n)
    val mat = base.add(diag).toBlockMatrix(blockSize, blockSize)
    mat.blocks.persist(StorageLevel.MEMORY_AND_DISK_SER)
    mat.blocks.count() // materialise before timing starts
    lnrdd.unpersist(true)
    mat
  }

  private def chooseMidDimSplits(n: Int, blockSize: Int): Int = {
    val blocksPerDim = (n + blockSize - 1) / blockSize
    math.max(1, math.min(8, blocksPerDim / 8))
  }

  def computeRmse(matrix: BlockMatrix, inv: BlockMatrix, numMidDimSplits: Int): Double = {
    val n = matrix.numRows()
    val product = matrix.multiply(inv, numMidDimSplits)
    val rpb = product.rowsPerBlock
    val cpb = product.colsPerBlock
    val errorSq = product.blocks.map { case ((bi, bj), mat) =>
      val arr = mat.toArray
      val nRows = mat.numRows
      val nCols = mat.numCols
      var blockErr = 0.0
      var c = 0
      while (c < nCols) {
        val globalCol = bj.toLong * cpb + c
        var r = 0
        while (r < nRows) {
          val globalRow = bi.toLong * rpb + r
          val expected = if (globalRow == globalCol) 1.0 else 0.0
          val diff = arr(r + c * nRows) - expected
          blockErr += diff * diff
          r += 1
        }
        c += 1
      }
      blockErr
    }.sum()
    math.sqrt(errorSq / (n.toDouble * n.toDouble))
  }

  def runBenchmark(sc: SparkContext, n: Int, schurBlockSize: Int, nsBlockSize: Int,
                   schurLimit: Int, iterMaxIter: Int, iterTol: Double): BenchResult = {
    println(s"\n${"=" * 60}")
    val schurMidSplits = chooseMidDimSplits(n, schurBlockSize)
    val nsMidSplits = chooseMidDimSplits(n, nsBlockSize)
    println(s"  n=$n  schurBlockSize=$schurBlockSize  nsBlockSize=$nsBlockSize  schurLimit=$schurLimit")
    println(s"  schurMidSplits=$schurMidSplits  nsMidSplits=$nsMidSplits")
    println(s"${"=" * 60}")

    // --- Schur complement recursive inversion ---
    println("\n[Schur complement recursive inversion]")
    val mat1 = buildMatrix(sc, n, schurBlockSize, seed = 42)
    val t1 = System.nanoTime()
    val inv1 = mat1.inverse(schurLimit, schurMidSplits)
    val schurTime = (System.nanoTime() - t1) / 1e9
    val schurRmse = computeRmse(mat1, inv1, schurMidSplits)
    println(f"  time=${schurTime}%.2fs  RMSE=$schurRmse%.3e")
    mat1.blocks.unpersist(true)
    inv1.blocks.unpersist(true)

    // --- Newton-Schulz (larger blockSize = fewer, bigger Spark tasks = less overhead) ---
    println("\n[Newton-Schulz iterative inversion]")
    val mat2 = buildMatrix(sc, n, nsBlockSize, seed = 42)
    val t2 = System.nanoTime()
    val inv2 = mat2.iterativeInverse(maxIter = iterMaxIter, tolerance = iterTol, numMidDimSplits = nsMidSplits)
    val iterTime = (System.nanoTime() - t2) / 1e9
    val iterRmse = computeRmse(mat2, inv2, nsMidSplits)
    println(f"  time=${iterTime}%.2fs  RMSE=$iterRmse%.3e")
    mat2.blocks.unpersist(true)
    inv2.blocks.unpersist(true)

    BenchResult(n, schurBlockSize, nsBlockSize, schurLimit, schurMidSplits, nsMidSplits,
      schurTime, schurRmse, iterTime, iterRmse, 0)
  }

  def printTable(results: Seq[BenchResult]): Unit = {
    println("\n" + "=" * 100)
    println(f"  ${"n"}%6s  ${"sBsz"}%5s  ${"nBsz"}%5s  ${"sSplit"}%6s  ${"nSplit"}%6s  | ${"Schur time(s)"}%14s  ${"Schur RMSE"}%12s  | ${"N-S time(s)"}%12s  ${"N-S RMSE"}%10s  ${"Speedup"}%8s")
    println("-" * 100)
    for (r <- results) {
      val speedup = r.schurTimeSec / r.iterTimeSec
      println(f"  ${r.n}%6d  ${r.schurBlockSize}%5d  ${r.nsBlockSize}%5d  ${r.schurMidSplits}%6d  ${r.nsMidSplits}%6d  | ${r.schurTimeSec}%14.2f  ${r.schurRmse}%12.3e  | ${r.iterTimeSec}%12.2f  ${r.iterRmse}%10.3e  ${speedup}%8.2fx")
    }
    println("=" * 100)
  }

  def main(args: Array[String]): Unit = {
    val sparkMemory = sys.props.getOrElse("sparkInverse.memory", "56g")
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .set("spark.driver.memory", sparkMemory)
      .set("spark.executor.memory", sparkMemory)
      .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      .set("spark.local.dir", "/tmp/spark_tmp")
      .setAppName("sparkInverse-benchmark"))

    sc.setLogLevel("WARN")
    sc.setCheckpointDir("/tmp/spark_checkpoints")

    // Sizes to benchmark. Each row: (n, schurBlockSize, nsBlockSize, schurLimit)
    //
    // Schur complement uses blockSize=100 throughout:
    //   - schurLimit=200 → limit/colsPerBlock = 2 → localInv base cases ≤ 400×400
    //   - keeping base cases small is critical because luInverse is O(n^3)
    //
    // Newton-Schulz uses larger block sizes that scale with n:
    //   - each N-S iteration is 2 full n×n matrix multiplications
    //   - larger blocks → fewer, bigger Spark tasks → less scheduler overhead on a single machine
    val configs = Seq(
      (500,   100, 100, 200),
      (1000,  100, 100, 200),
      (2000,  100, 100, 200),
      (4000,  100, 200, 200),
      (8000,  100, 400, 200),
    )

    val results = configs.map { case (n, schurBlockSize, nsBlockSize, schurLimit) =>
      runBenchmark(sc, n, schurBlockSize, nsBlockSize, schurLimit,
                   iterMaxIter = 30, iterTol = 1e-10)
    }

    printTable(results)
    sc.stop()
  }
}
