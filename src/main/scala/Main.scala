import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.{SparkConf, SparkContext}
import Inverse.BlockMatrixInverse
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.storage.StorageLevel

import java.lang

object Main {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .set("spark.driver.memory", "64g")
      .set("spark.executor.memory", "64g")
      .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      .registerKryoClasses(Array(classOf[IndexedRow], classOf[MatrixEntry], classOf[Matrix]))
      .setAppName("Main"))

    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("D:\\spark_checkpoints")

    val n = 3720 // math.pow(2, 11).intValue - 1
    val lnrdd = RandomRDDs.normalVectorRDD(sc, n, n, seed = 42, numPartitions = 16)
      .zipWithIndex()
      .map(_.swap)
      .map(x => IndexedRow(x._1, x._2))



    val matrix = new IndexedRowMatrix(lnrdd)
      .toBlockMatrix(512, 512)
    println("Matrix shape: " + matrix.numRows() + ", " + matrix.numCols())
    val t = System.nanoTime()
    val inverted = matrix.inverse(1250, 1)
      .cache()

    val errorSum = matrix.multiply(inverted).toCoordinateMatrix().entries
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
    println("input partitions: " + matrix.blocks.getNumPartitions)
    println("inverted output partitions: " + inverted.blocks.getNumPartitions)
    println("Error squared sum: " + errorSum)
    println("Error average squared sum: " + errorSum/(n*n))
    println("RMSE: " + math.sqrt(errorSum / (n * n)))

    matrix.blocks.unpersist(true)
    inverted.blocks.unpersist(true)

  }
}