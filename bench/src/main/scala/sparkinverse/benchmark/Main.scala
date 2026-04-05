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
    hp3Sec: Double,
    hp3Rmse: Double,
    hp4Sec: Double,
    hp4Rmse: Double,
    iterSec: Double,
    iterRmse: Double
  )

  private val kryoClasses: Array[Class[_]] = Array(
    classOf[DenseVector], classOf[SparseVector], classOf[DenseMatrix], classOf[SparseMatrix],
    classOf[Vector], classOf[Matrix], classOf[MatrixEntry], classOf[IndexedRow],
    classOf[CoordinateMatrix], classOf[BlockMatrix], classOf[Array[Double]], classOf[Array[Int]], classOf[Array[Long]]
  )

  def buildMatrix(sc: SparkContext, n: Int, blockSize: Int, seed: Long): BlockMatrix = {
    val lnrdd = RandomRDDs.normalVectorRDD(sc, numRows = n, numCols = n, seed = seed, numPartitions = 8)
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
    println(s"  hp3  : blockSize=${config.nsBlockSize} midSplits=${config.nsMidSplits}")
    println(s"  hp4  : blockSize=${config.nsBlockSize} midSplits=${config.nsMidSplits}")
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

    println("\n[Third-order iterative inversion (order=3)]")
    val hp3Config = IterativeInverseConfig(
      maxIter = iterMaxIter,
      tolerance = iterTol,
      useCheckpoints = true,
      checkpointInterval = 5,
      numMidDimSplits = config.nsMidSplits
    )
    val (hp3Sec, hp3Mat, hp3Inv) =
      runOnce(sc, config.n, config.nsBlockSize)(mat => MatrixInversion.block(mat).iterativeInverse(3, hp3Config))
    val hp3Rmse = computeRmse(hp3Mat, hp3Inv, config.nsMidSplits)
    println(f"  time=${hp3Sec}%.2fs  RMSE=$hp3Rmse%.3e")
    hp3Mat.blocks.unpersist(true)
    hp3Inv.blocks.unpersist(true)

    println("\n[Fourth-order iterative inversion (order=4)]")
    val hp4Config = IterativeInverseConfig(
      maxIter = iterMaxIter,
      tolerance = iterTol,
      useCheckpoints = true,
      checkpointInterval = 5,
      numMidDimSplits = config.nsMidSplits
    )
    val (hp4Sec, hp4Mat, hp4Inv) =
      runOnce(sc, config.n, config.nsBlockSize)(mat => MatrixInversion.block(mat).iterativeInverse(4, hp4Config))
    val hp4Rmse = computeRmse(hp4Mat, hp4Inv, config.nsMidSplits)
    println(f"  time=${hp4Sec}%.2fs  RMSE=$hp4Rmse%.3e")
    hp4Mat.blocks.unpersist(true)
    hp4Inv.blocks.unpersist(true)

    println("\n[Newton-Schulz iterative inversion (order=2)]")
    val iterConfig = IterativeInverseConfig(
      maxIter = iterMaxIter,
      tolerance = iterTol,
      useCheckpoints = true,
      checkpointInterval = 5,
      numMidDimSplits = config.nsMidSplits
    )
    val (iterSec, iterMat, iterInv) =
      runOnce(sc, config.n, config.nsBlockSize)(mat => MatrixInversion.block(mat).iterativeInverse(2, iterConfig))
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
      hp3Sec = hp3Sec,
      hp3Rmse = hp3Rmse,
      hp4Sec = hp4Sec,
      hp4Rmse = hp4Rmse,
      iterSec = iterSec,
      iterRmse = iterRmse
    )
  }

  def printTable(results: Seq[BenchResult]): Unit = {
    println("\n" + "=" * 186)
    println(f"  ${"n"}%6s  ${"sBsz"}%5s  ${"iBsz"}%5s  ${"sLim"}%5s  ${"sSplit"}%6s  ${"iSplit"}%6s  | ${"Schur(s)"}%10s  ${"Schur RMSE"}%12s  | ${"HP3(s)"}%10s  ${"HP3 RMSE"}%10s  | ${"HP4(s)"}%10s  ${"HP4 RMSE"}%10s  | ${"N-S(s)"}%10s  ${"N-S RMSE"}%10s  ${"HP4/HP3"}%9s  ${"N-S/HP3"}%9s")
    println("-" * 186)
    for (r <- results) {
      val hp4VsHp3 = r.hp4Sec / r.hp3Sec
      val nsVsHp3 = r.iterSec / r.hp3Sec
      println(f"  ${r.n}%6d  ${r.schurBlockSize}%5d  ${r.nsBlockSize}%5d  ${r.schurLimit}%5d  ${r.schurMidSplits}%6d  ${r.nsMidSplits}%6d  | ${r.schurSec}%10.2f  ${r.schurRmse}%12.3e  | ${r.hp3Sec}%10.2f  ${r.hp3Rmse}%10.3e  | ${r.hp4Sec}%10.2f  ${r.hp4Rmse}%10.3e  | ${r.iterSec}%10.2f  ${r.iterRmse}%10.3e  ${hp4VsHp3}%9.2fx  ${nsVsHp3}%9.2fx")
    }
    println("=" * 186)
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
      BenchConfig(n = 1000, schurBlockSize = 125, nsBlockSize = 125, schurLimit = 250, schurMidSplits = 2, nsMidSplits = 2),
      BenchConfig(n = 4000, schurBlockSize = 250, nsBlockSize = 250, schurLimit = 500, schurMidSplits = 4, nsMidSplits = 4),
      BenchConfig(n = 6000, schurBlockSize = 300, nsBlockSize = 300, schurLimit = 600, schurMidSplits = 4, nsMidSplits = 4)
    )

    val results = configs.map { config =>
      runBenchmark(sc, config, iterMaxIter = 30, iterTol = 1e-10)
    }

    printTable(results)
    sc.stop()
  }
}
