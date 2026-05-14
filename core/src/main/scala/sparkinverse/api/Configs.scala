package sparkinverse.api

import org.apache.spark.storage.StorageLevel

sealed trait PseudoInverseSide

object PseudoInverseSide {
  case object Left extends PseudoInverseSide
  case object Right extends PseudoInverseSide
}

final case class RecursiveInverseConfig(
  limit: Int = 4096,
  midSplits: Int = 1,
  useCheckpoints: Boolean = true,
  targetOutputPartitions: Option[Int] = None,
  unionCoalesceThreshold: Int = 8,
  minBlockSizeForPersistence: Int = 1000000
)

final case class IterativeInverseConfig(
  order: Int = 2,
  maxIter: Int = 30,
  tolerance: Double = 1e-15,
  useCheckpoints: Boolean = true,
  checkpointEvery: Int = 5,
  midSplits: Int = 1,
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER
)
