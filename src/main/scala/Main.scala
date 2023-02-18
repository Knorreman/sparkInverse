import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.{SparkConf, SparkContext}
import Inverse.{BlockMatrixInverse, CoordinateMatrixInverse}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.storage.StorageLevel

object Main {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .set("spark.driver.memory", "64g")
      .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      .set("spark.kryoserializer.buffer.max", "2047m")
      .set("spark.local.dir", "/mnt/h/spark_tmp")
//      .set("spark.local.dir", "/mnt/d/spark_tmp_debian/,/mnt/c/spark_tmp_debian/")
//      .set("spark.io.compression.lz4.blockSize", "128k")
//      .set("spark.shuffle.file.buffer", "2MB")
      .registerKryoClasses(Array(classOf[IndexedRow], classOf[MatrixEntry], classOf[Matrix], classOf[DenseMatrix]))
      .setAppName("Main"))

    sc.setLogLevel("WARN")
    sc.setCheckpointDir("/mnt/h/spark_checkpoints")
//      sc.setCheckpointDir("/mnt/d/spark_checkpoints_debian")
    import dev.ludovic.netlib.NativeBLAS
    println(NativeBLAS.getInstance())
    val n = 1001 // math.pow(2, 11).intValue - 1

//    val lnrdd = RandomRDDs.exponentialVectorRDD(sc, 0.01, n, n, seed = 42, numPartitions = 8)
//      .zipWithIndex()
//      .map(_.swap)
//      .map(x => IndexedRow(x._1, x._2))
//      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val lnrdd = RandomRDDs.poissonVectorRDD(sc, 0.01, n, n, seed = 42, numPartitions = 4)
      .zipWithIndex()
      .map(_.swap)
      .map(x => IndexedRow(x._1, x._2))
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val nonZeroCount = lnrdd.flatMap(ir => ir.vector.toArray).filter(d => d != 0.0).count()
    println("Sparsity: " + nonZeroCount.toDouble / (n*n))

    val matrix = new IndexedRowMatrix(lnrdd)
      .toBlockMatrix(100, 100)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
    println("Matrix shape: " + matrix.numRows() + ", " + matrix.numCols())
    matrix.blocks.count()
    val t = System.nanoTime()
    val inverted = matrix.inverse(250)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val eyeMaybe = matrix.multiply(inverted).toCoordinateMatrix().entries
    val errorSum = eyeMaybe
      .map {
        me => {
          if (me.j == me.i) {
            math.pow(me.value - 1.0, 2)
          } else {
            math.pow(me.value, 2)
          }
        }
      }
      .sum()
    println("Time taken: " + (System.nanoTime() - t) / 1e9 + "s")

    val absErrorSum = eyeMaybe
      .map {
        me => {
          if (me.j == me.i) {
            math.abs(me.value - 1.0)
          } else {
            math.abs(me.value)
          }
        }
      }
      .sum()

    println("Time taken2: " + (System.nanoTime() - t) / 1e9 + "s")
    println("input partitions: " + matrix.blocks.getNumPartitions)
    println("inverted output partitions: " + inverted.blocks.getNumPartitions)
    println("Error squared sum: " + errorSum)
    println("Error average squared sum: " + errorSum/(n*n))
    println("RMSE: " + math.sqrt(errorSum / (n * n)))
    println("Error abs sum: " + absErrorSum)
    println("Error average abs sum: " + absErrorSum / (n * n))

    matrix.blocks.unpersist(true)
    inverted.blocks.unpersist(true)

  }
}