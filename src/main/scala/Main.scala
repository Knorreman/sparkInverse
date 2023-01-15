import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.{SparkConf, SparkContext}
import Inverse.BlockMatrixInverse
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.storage.StorageLevel

object Main {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf()
      .setMaster("local[*]")
      .set("spark.driver.memory", "64g")
      .set("spark.executor.memory", "64g")
      .registerKryoClasses(Array(classOf[IndexedRow], classOf[MatrixEntry], classOf[Matrix]))
      .setAppName("Main"))
    sc.setLogLevel("ERROR")

    val n = math.pow(2, 11).intValue - 1
    val lnrdd = RandomRDDs.logNormalVectorRDD(sc, 0.0, 1.0, n, n, seed = 42, numPartitions = 32)
      .zipWithIndex()
      .map(_.swap)
      .map(x=> IndexedRow(x._1, x._2))

    val matrix = new IndexedRowMatrix(lnrdd)
      .toBlockMatrix(512, 512)
      .persist(StorageLevel.MEMORY_ONLY_SER)
    val inverted = matrix.inverse(1024, 2)
      .persist(StorageLevel.MEMORY_ONLY_SER)

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

    println("Error squared sum: " + errorSum)
    println("Error average squared sum: " + errorSum/(n*n))

  }
}