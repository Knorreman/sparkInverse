package sparkinverse.benchmark

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix, SparseVector, Vector}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import sparkinverse.api.{IterativeInverseConfig, MatrixInversion, RecursiveInverseConfig, RecursiveTuning}
import sparkinverse.syntax.coordinate._

import java.net.URI
import java.nio.file.{Files, Paths}

object Main {
  case class BenchConfig(
    n: Int,
    schurBlockSize: Int,
    nsBlockSize: Int,
    schurLimit: Int,
    schurMidSplits: Int,
    nsMidSplits: Int
  )

  case class BenchResult(
    n: Int,
    schurBlockSize: Int,
    nsBlockSize: Int,
    schurLimit: Int,
    schurMidSplits: Int,
    nsMidSplits: Int,
    schurSec: Double,
    schurRmse: Double,
    iterSec: Double,
    iterRmse: Double
  )

  private val kryoClasses: Array[Class[_]] = Array(
    classOf[DenseVector], classOf[SparseVector], classOf[DenseMatrix], classOf[SparseMatrix],
    classOf[Vector], classOf[Matrix], classOf[MatrixEntry], classOf[IndexedRow],
    classOf[CoordinateMatrix], classOf[BlockMatrix], classOf[Array[Double]], classOf[Array[Int]], classOf[Array[Long]]
  )

  def buildMatrix(sc: SparkContext, n: Int, blockSize: Int, seed: Long): BlockMatrix = {
    val lnrdd = RandomRDDs.poissonVectorRDD(sc, mean = 0.01, numRows = n, numCols = n, seed = seed, numPartitions = 8)
      .zipWithIndex()
      .map(_.swap)
      .map(x => IndexedRow(x._1, x._2))
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
    lnrdd.count()

    val base = new IndexedRowMatrix(lnrdd).toCoordinateMatrix()
    val diag = new CoordinateMatrix(sc.range(0, n).map(i => MatrixEntry(i, i, n.toDouble)), n, n)
    val mat = base.add(diag).toBlockMatrix(blockSize, blockSize)
    mat.blocks.persist(StorageLevel.MEMORY_AND_DISK_SER)
    mat.blocks.count()
    lnrdd.unpersist(true)
    mat
  }

  private def runOnce(sc: SparkContext, n: Int, blockSize: Int)(compute: BlockMatrix => BlockMatrix): (Double, BlockMatrix, BlockMatrix) = {
    val mat = buildMatrix(sc, n, blockSize, seed = 42)
    val t0 = System.nanoTime()
    val inv = compute(mat)
    inv.blocks.count()
    val secs = (System.nanoTime() - t0) / 1e9
    (secs, mat, inv)
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

  def runBenchmark(sc: SparkContext, config: BenchConfig, iterMaxIter: Int, iterTol: Double): BenchResult = {
    println(s"\n${"=" * 60}")
    println(s"  n=${config.n}")
    println(s"  schur: blockSize=${config.schurBlockSize} limit=${config.schurLimit} midSplits=${config.schurMidSplits}")
    println(s"  n-s  : blockSize=${config.nsBlockSize} midSplits=${config.nsMidSplits}")
    println(s"${"=" * 60}")

    println("\n[Schur complement recursive inversion]")
    val schurConfig = RecursiveInverseConfig(
      limit = config.schurLimit,
      numMidDimSplits = config.schurMidSplits,
      useCheckpoints = true,
      tuning = RecursiveTuning(adaptiveMidDimSplits = false)
    )
    val (schurSec, schurMat, schurInv) =
      runOnce(sc, config.n, config.schurBlockSize)(mat => MatrixInversion.block(mat).inverse(schurConfig))
    val schurRmse = computeRmse(schurMat, schurInv, config.schurMidSplits)
    println(f"  time=${schurSec}%.2fs  RMSE=$schurRmse%.3e")
    schurMat.blocks.unpersist(true)
    schurInv.blocks.unpersist(true)

    println("\n[Newton-Schulz iterative inversion]")
    val iterConfig = IterativeInverseConfig(
      maxIter = iterMaxIter,
      tolerance = iterTol,
      useCheckpoints = true,
      checkpointInterval = 5,
      numMidDimSplits = config.nsMidSplits
    )
    val (iterSec, iterMat, iterInv) =
      runOnce(sc, config.n, config.nsBlockSize)(mat => MatrixInversion.block(mat).iterativeInverse(iterConfig))
    val iterRmse = computeRmse(iterMat, iterInv, config.nsMidSplits)
    println(f"  time=${iterSec}%.2fs  RMSE=$iterRmse%.3e")
    iterMat.blocks.unpersist(true)
    iterInv.blocks.unpersist(true)

    BenchResult(
      n = config.n,
      schurBlockSize = config.schurBlockSize,
      nsBlockSize = config.nsBlockSize,
      schurLimit = config.schurLimit,
      schurMidSplits = config.schurMidSplits,
      nsMidSplits = config.nsMidSplits,
      schurSec = schurSec,
      schurRmse = schurRmse,
      iterSec = iterSec,
      iterRmse = iterRmse
    )
  }

  def printTable(results: Seq[BenchResult]): Unit = {
    println("\n" + "=" * 114)
    println(f"  ${"n"}%6s  ${"sBsz"}%5s  ${"nBsz"}%5s  ${"sLim"}%5s  ${"sSplit"}%6s  ${"nSplit"}%6s  | ${"Schur(s)"}%10s  ${"Schur RMSE"}%12s  | ${"N-S(s)"}%10s  ${"N-S RMSE"}%10s  ${"Speedup"}%8s")
    println("-" * 114)
    for (r <- results) {
      val speedup = r.schurSec / r.iterSec
      println(f"  ${r.n}%6d  ${r.schurBlockSize}%5d  ${r.nsBlockSize}%5d  ${r.schurLimit}%5d  ${r.schurMidSplits}%6d  ${r.nsMidSplits}%6d  | ${r.schurSec}%10.2f  ${r.schurRmse}%12.3e  | ${r.iterSec}%10.2f  ${r.iterRmse}%10.3e  ${speedup}%8.2fx")
    }
    println("=" * 114)
  }

  def main(args: Array[String]): Unit = {
    val sparkMemory = sys.props.getOrElse("sparkInverse.memory", "56g")
    val openBlasThreads = sys.props.getOrElse("sparkInverse.openblasThreads", sys.env.getOrElse("OPENBLAS_NUM_THREADS", "1"))
    val eventLogDirRaw = sys.props.getOrElse("sparkInverse.eventLogDir", "spark-events")
    val eventLogUri = if (eventLogDirRaw.startsWith("file:")) eventLogDirRaw else Paths.get(eventLogDirRaw).toAbsolutePath.toUri.toString
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

    val configs = Seq(
      BenchConfig(n = 500, schurBlockSize = 100, nsBlockSize = 100, schurLimit = 200, schurMidSplits = 1, nsMidSplits = 1),
      BenchConfig(n = 2000, schurBlockSize = 100, nsBlockSize = 200, schurLimit = 300, schurMidSplits = 4, nsMidSplits = 4),
      BenchConfig(n = 4000, schurBlockSize = 100, nsBlockSize = 200, schurLimit = 300, schurMidSplits = 4, nsMidSplits = 4),
      BenchConfig(n = 6000, schurBlockSize = 100, nsBlockSize = 400, schurLimit = 400, schurMidSplits = 8, nsMidSplits = 4)
    )

    val results = configs.map { config =>
      runBenchmark(sc, config, iterMaxIter = 30, iterTol = 1e-10)
    }

    printTable(results)
    sc.stop()
  }
}
