import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix, SparseVector, Vector}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.{SparkConf, SparkContext}
import Inverse.{BlockMatrixInverse, CoordinateMatrixInverse, IterativePerfConfig, RecursivePerfConfig}
import org.apache.spark.storage.StorageLevel
import java.net.URI
import java.nio.file.{Files, Paths}

object Main {

  case class BenchResult(n: Int, schurBlockSize: Int, nsBlockSize: Int, schurLimit: Int,
                         schurMidSplits: Int, nsMidSplits: Int,
                         warmupRuns: Int, measuredRuns: Int,
                         schurMedianSec: Double, schurP95Sec: Double, schurRmse: Double,
                         iterMedianSec: Double, iterP95Sec: Double, iterRmse: Double)

  private val kryoClasses: Array[Class[_]] = Array(
    classOf[DenseVector],
    classOf[SparseVector],
    classOf[DenseMatrix],
    classOf[SparseMatrix],
    classOf[Vector],
    classOf[Matrix],
    classOf[MatrixEntry],
    classOf[IndexedRow],
    classOf[CoordinateMatrix],
    classOf[BlockMatrix],
    classOf[Array[Double]],
    classOf[Array[Int]],
    classOf[Array[Long]]
  )

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

  private def percentile(values: Seq[Double], p: Double): Double = {
    val sorted = values.sorted
    val idx = math.min(sorted.length - 1, math.max(0, math.ceil(p * sorted.length).toInt - 1))
    sorted(idx)
  }

  private def benchmarkRuns(
    warmupRuns: Int,
    measuredRuns: Int)(
    runOnce: Boolean => (Double, Double)): (Seq[Double], Seq[Double]) = {
    val totalRuns = warmupRuns + measuredRuns
    val times = scala.collection.mutable.ArrayBuffer.empty[Double]
    val rmses = scala.collection.mutable.ArrayBuffer.empty[Double]
    var i = 0
    while (i < totalRuns) {
      val shouldComputeRmse = i == totalRuns - 1
      val (secs, rmse) = runOnce(shouldComputeRmse)
      if (i >= warmupRuns) {
        times += secs
        if (!rmse.isNaN) {
          rmses += rmse
        }
      }
      i += 1
    }
    (times.toSeq, rmses.toSeq)
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
                   schurLimit: Int, iterMaxIter: Int, iterTol: Double,
                   warmupRuns: Int, measuredRuns: Int, perfTrace: Boolean): BenchResult = {
    println(s"\n${"=" * 60}")
    val schurMidSplits = chooseMidDimSplits(n, schurBlockSize)
    val nsMidSplits = chooseMidDimSplits(n, nsBlockSize)
    println(s"  n=$n  schurBlockSize=$schurBlockSize  nsBlockSize=$nsBlockSize  schurLimit=$schurLimit")
    println(s"  schurMidSplits=$schurMidSplits  nsMidSplits=$nsMidSplits")
    println(s"  warmupRuns=$warmupRuns  measuredRuns=$measuredRuns  perfTrace=$perfTrace")
    println(s"${"=" * 60}")

    // --- Schur complement recursive inversion ---
    println("\n[Schur complement recursive inversion]")
    val schurPerf = RecursivePerfConfig(trace = perfTrace, adaptiveMidDimSplits = true)
    val (schurTimes, schurRmses) = benchmarkRuns(warmupRuns, measuredRuns) { computeRmseOnThisRun =>
      val mat = buildMatrix(sc, n, schurBlockSize, seed = 42)
      val t = System.nanoTime()
      val inv = mat.inverse(schurLimit, schurMidSplits, useCheckpoints = true, depth = 0, perf = schurPerf)
      val secs = (System.nanoTime() - t) / 1e9
      val rmse = if (computeRmseOnThisRun) computeRmse(mat, inv, schurMidSplits) else Double.NaN
      mat.blocks.unpersist(true)
      inv.blocks.unpersist(true)
      (secs, rmse)
    }
    val schurMedian = percentile(schurTimes, 0.50)
    val schurP95 = percentile(schurTimes, 0.95)
    val schurRmse = schurRmses.lastOption.getOrElse(Double.NaN)
    println(f"  median=${schurMedian}%.2fs  p95=${schurP95}%.2fs  RMSE(sample)=$schurRmse%.3e")

    // --- Newton-Schulz (larger blockSize = fewer, bigger Spark tasks = less overhead) ---
    println("\n[Newton-Schulz iterative inversion]")
    val iterPerf = IterativePerfConfig(trace = perfTrace, checkpointEvery = 5)
    val (iterTimes, iterRmses) = benchmarkRuns(warmupRuns, measuredRuns) { computeRmseOnThisRun =>
      val mat = buildMatrix(sc, n, nsBlockSize, seed = 42)
      val t = System.nanoTime()
      val inv = mat.iterativeInverse(
        maxIter = iterMaxIter,
        tolerance = iterTol,
        useCheckpoints = true,
        checkpointInterval = 5,
        numMidDimSplits = nsMidSplits,
        perf = iterPerf)
      val secs = (System.nanoTime() - t) / 1e9
      val rmse = if (computeRmseOnThisRun) computeRmse(mat, inv, nsMidSplits) else Double.NaN
      mat.blocks.unpersist(true)
      inv.blocks.unpersist(true)
      (secs, rmse)
    }
    val iterMedian = percentile(iterTimes, 0.50)
    val iterP95 = percentile(iterTimes, 0.95)
    val iterRmse = iterRmses.lastOption.getOrElse(Double.NaN)
    println(f"  median=${iterMedian}%.2fs  p95=${iterP95}%.2fs  RMSE(sample)=$iterRmse%.3e")

    BenchResult(n, schurBlockSize, nsBlockSize, schurLimit, schurMidSplits, nsMidSplits,
      warmupRuns, measuredRuns, schurMedian, schurP95, schurRmse, iterMedian, iterP95, iterRmse)
  }

  def printTable(results: Seq[BenchResult]): Unit = {
    println("\n" + "=" * 132)
    println(f"  ${"n"}%6s  ${"sBsz"}%5s  ${"nBsz"}%5s  ${"sSplit"}%6s  ${"nSplit"}%6s  ${"runs"}%6s  | ${"Schur med(s)"}%12s  ${"Schur p95"}%10s  ${"Schur RMSE"}%12s  | ${"N-S med(s)"}%10s  ${"N-S p95"}%10s  ${"N-S RMSE"}%10s  ${"Speedup"}%8s")
    println("-" * 132)
    for (r <- results) {
      val speedup = r.schurMedianSec / r.iterMedianSec
      println(f"  ${r.n}%6d  ${r.schurBlockSize}%5d  ${r.nsBlockSize}%5d  ${r.schurMidSplits}%6d  ${r.nsMidSplits}%6d  ${r.measuredRuns}%6d  | ${r.schurMedianSec}%12.2f  ${r.schurP95Sec}%10.2f  ${r.schurRmse}%12.3e  | ${r.iterMedianSec}%10.2f  ${r.iterP95Sec}%10.2f  ${r.iterRmse}%10.3e  ${speedup}%8.2fx")
    }
    println("=" * 132)
  }

  def main(args: Array[String]): Unit = {
    val sparkMemory = sys.props.getOrElse("sparkInverse.memory", "56g")
    val warmupRuns = sys.props.getOrElse("sparkInverse.bench.warmups", "1").toInt
    val measuredRuns = sys.props.getOrElse("sparkInverse.bench.runs", "3").toInt
    val perfTrace = sys.props.getOrElse("sparkInverse.perfTrace", "false").toBoolean
    val openBlasThreads = sys.props.getOrElse("sparkInverse.openblasThreads",
      sys.env.getOrElse("OPENBLAS_NUM_THREADS", "1"))
    val eventLogDirRaw = sys.props.getOrElse("sparkInverse.eventLogDir", "spark-events")
    val eventLogUri =
      if (eventLogDirRaw.startsWith("file:")) eventLogDirRaw
      else Paths.get(eventLogDirRaw).toAbsolutePath.toUri.toString
    Files.createDirectories(Paths.get(URI.create(eventLogUri)))
    println(s"[native] OPENBLAS_NUM_THREADS=$openBlasThreads")
    println(s"[spark] eventLogDir=$eventLogUri")
    println(s"[spark] registerKryoClasses=${kryoClasses.length}")

    val conf = new SparkConf()
      .setMaster("local[*]")
      .set("spark.driver.memory", sparkMemory)
      .set("spark.executor.memory", sparkMemory)
      .set("spark.executorEnv.OPENBLAS_NUM_THREADS", openBlasThreads)
      .set("spark.executorEnv.OMP_NUM_THREADS", openBlasThreads)
      .set("spark.executorEnv.GOTO_NUM_THREADS", openBlasThreads)
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      .set("spark.eventLog.enabled", "true")
      .set("spark.eventLog.compress", "true")
      .set("spark.eventLog.dir", eventLogUri)
      .set("spark.history.fs.logDirectory", eventLogUri)
      .set("spark.local.dir", "/tmp/spark_tmp")
      .setAppName("sparkInverse-benchmark")
      .registerKryoClasses(kryoClasses)

    val sc = new SparkContext(conf)

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
    )

    val results = configs.map { case (n, schurBlockSize, nsBlockSize, schurLimit) =>
      runBenchmark(sc, n, schurBlockSize, nsBlockSize, schurLimit,
                   iterMaxIter = 30, iterTol = 1e-10,
                   warmupRuns = warmupRuns, measuredRuns = measuredRuns, perfTrace = perfTrace)
    }

    printTable(results)
    sc.stop()
  }
}
