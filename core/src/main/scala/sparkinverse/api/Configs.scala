package sparkinverse.api

import org.apache.spark.storage.StorageLevel

final case class RecursiveTuning(
  targetOutputPartitions: Option[Int] = None,
  unionCoalesceThreshold: Int = 8,
  adaptiveMidDimSplits: Boolean = true,
  adaptivePersistence: Boolean = true,
  minBlockSizeForPersistence: Int = 1000000, // Minimum elements for persistence
  conditionNumberThreshold: Double = 1e6, // Threshold for ill-conditioning detection
  divergenceDetection: Boolean = true,
  maxDivergenceCount: Int = 3
)

final case class IterativeTuning(
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
  checkpointEvery: Int = 5,
  useLocalCheckpoint: Boolean = false,
  adaptiveMidDimSplits: Boolean = true,
  maxAdaptiveMidDimSplits: Int = 16,
  largeMatrixCheckpointEvery: Int = 2,
  largeMatrixThreshold: Int = 4000,
  adaptiveStepSize: Boolean = true,
  conditionNumberThreshold: Double = 1e6,
  divergenceDetection: Boolean = true,
  maxDivergenceCount: Int = 3,
  conservativeInitialization: Boolean = true
)

final case class RecursiveInverseConfig(
  limit: Int = 4096,
  numMidDimSplits: Int = 1,
  useCheckpoints: Boolean = true,
  tuning: RecursiveTuning = RecursiveTuning()
)

final case class IterativeInverseConfig(
  maxIter: Int = 30,
  tolerance: Double = 1e-15,
  useCheckpoints: Boolean = true,
  checkpointInterval: Int = 5,
  numMidDimSplits: Int = 1,
  tuning: IterativeTuning = IterativeTuning()
)
