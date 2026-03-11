package sparkinverse.api

import org.apache.spark.storage.StorageLevel

final case class RecursiveTuning(
  targetOutputPartitions: Option[Int] = None,
  unionCoalesceThreshold: Int = 8,
  adaptiveMidDimSplits: Boolean = true
)

final case class IterativeTuning(
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
  checkpointEvery: Int = 5,
  useLocalCheckpoint: Boolean = false,
  adaptiveMidDimSplits: Boolean = true,
  maxAdaptiveMidDimSplits: Int = 16,
  largeMatrixCheckpointEvery: Int = 2,
  largeMatrixThreshold: Int = 4000
)

final case class RecursiveInverseConfig(
  limit: Int = 4096,
  numMidDimSplits: Int = 1,
  useCheckpoints: Boolean = true,
  tuning: RecursiveTuning = RecursiveTuning()
)

final case class IterativeInverseConfig(
  maxIter: Int = 30,
  tolerance: Double = 1e-10,
  useCheckpoints: Boolean = true,
  checkpointInterval: Int = 5,
  numMidDimSplits: Int = 1,
  tuning: IterativeTuning = IterativeTuning()
)
