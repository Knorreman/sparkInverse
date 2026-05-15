# TODO — Efficiency, Correctness & Algorithmic Improvements

Each item includes: description, code location, root cause, proposed fix, effort estimate, and priority.

---

## ✅ Done

### 1. Unpersist Before Lazy Evaluation — COMPLETED

**Priority:** 🔴 Critical  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/test/scala/sparkinverse/TestInverse.scala`

**Implementation:**

- `inverseInternal()` now persists and materializes the assembled output before unpersisting tracked intermediates
- For checkpointed runs: `persist → checkpoint → count → unpersist parents`
- For non-checkpointed runs with tracked intermediates: `persist → count → unpersist parents`
- Prevents repeated recomputation and ensures checkpoint lineage is written before parent caches are released

**Validation:**

- Added recursive tests for checkpointed and non-checkpointed materialization with forced intermediate persistence
- All 49 core tests pass

---

### 3. Single-Pass Quadrant Split — COMPLETED

**Priority:** 🟠 High  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/test/scala/sparkinverse/TestInverse.scala`

**Implementation:**

- `splitQuadrants()` can now tag and cache block quadrants in a single materialized pass
- Shared tagged split RDD is tracked and cleaned up after recursive output materialization
- Avoids unsafe early unpersist of lazy quadrant parents
- Avoids `partitionBy` on four quadrant keys to prevent unnecessary shuffle/skew
- Gated by checkpointing or `minBlockSizeForPersistence` to avoid extra count/cache overhead on small matrices

**Validation:**

- Added odd block-grid recursive inversion test for uneven quadrant splits
- Existing recursive checkpoint/non-checkpoint materialization tests exercise shared split lifecycle
- All 50 core tests pass

---

### 4. LU-Based Local Inversion (Replace SVD Base Case) — COMPLETED

**Priority:** 🟠 High  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/syntax/block.scala`, `build.sbt`

**Implementation:**

- Added `commons-math3` as explicit dependency
- Implemented `luInverse()` using `LUDecomposition` with 1e-12 singularity threshold
- `localInverse()` now tries LU first, falls back to SVD on `IllegalArgumentException`
- ~2× faster for base case matrices (≤4096 elements / ≤64×64)
- Added tests verifying LU matches SVD for well-conditioned matrices

**Validation:**

- All 45 core tests pass
- New tests: "LU inverse matches SVD" and "LU inverse handles scaled identity"

---

### 7. Convergence Check Frequency — COMPLETED

**Priority:** 🟡 Medium  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/api/Configs.scala`, `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/test/scala/sparkinverse/TestInverse.scala`, `README.md`

**Implementation:**

- Added `IterativeInverseConfig.convergenceCheckInterval` with default `1` for backward-compatible per-iteration checks
- Validates interval is positive
- Iterative inverse now computes the expensive Frobenius convergence metric only on iteration 1 and every N iterations
- Adaptive alpha still performs its first-iteration metric/refinement path
- Non-convergence warning now reports the last checked metric

**Validation:**

- Added tests for sparse convergence checks, higher-order hyperpower, adaptive alpha, invalid interval, and default compatibility
- All 54 core tests pass

---

### 13. Dead Code — `previousMetric` — COMPLETED

**Priority:** 🟢 Low  
**Effort:** Minimal  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/test/scala/sparkinverse/TestInverse.scala`

**Implementation:**

- Renamed `previousMetric` → `lastMetric` (already done via convergenceCheckInterval refactor)
- Added divergence detection: `metric > lastMetric * 2 && lastMetric > tolerance` inside `metricOpt.foreach`
- Added alpha-refinement divergence warning when `metricNew > metric * 2`
- Fixed `1 % checkpointEvery` bug → `iter % checkpointEvery` in adaptive alpha branch
- Non-convergence warning updated to "Last checked metric"

**Validation:**

- Added tests: divergence detection on well-conditioned matrix, non-convergence still returns result
- All 56 core tests pass

---

### 5. A Caching in Iterative Loop — COMPLETED

**Priority:** 🟢 Low  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

**Implementation:**

- Added `matrix.blocks.count()` after `persist()` in `iterativeInverseInternal` to force cache materialization before the iterative loop starts
- Input matrix A is now guaranteed cached before the first multiply
- Skipped `pseudoInverse` — no loop, no repeated recomputation benefit

**Validation:**

- All 56 core tests pass
- bench/compile passes

---

### 12. No Regularization for Pseudo-Inverse — COMPLETED

**Priority:** 🟡 Medium  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/api/Configs.scala`, `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/test/scala/sparkinverse/TestInverse.scala`, `README.md`

**Implementation:**

- Added `regularizationLambda: Double = 0.0` to `RecursiveInverseConfig`
- Validates lambda >= 0
- When lambda > 0, `pseudoInverse` adds λI to the Gram matrix before inversion (Tikhonov regularization)
- Identity matrix constructed via `MatrixInternals.eyeBlockMatrix` and unpersisted after use
- Backward compatible: lambda = 0.0 preserves standard Moore-Penrose behavior

**Validation:**

- Added tests: backward compatibility (lambda=0), regularization changes result, Left/Right sides, ridge shrinkage, negative lambda rejection
- All 58 core tests pass

---

## 1. Unpersist Before Lazy Evaluation (Correctness Bug) — COMPLETED

**Priority:** 🔴 Critical  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

### Problem

In `inverseInternal()`, line ~251:

```scala
val bm = new BlockMatrix(allBlocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
if (config.useCheckpoints) {
  bm.blocks.persist(iterativeStorageLevel)
  bm.blocks.checkpoint()
}
cachedMatrices.foreach(cached => cached.blocks.unpersist(true))
bm
```

Spark RDDs are lazy. The intermediates (eInv, geInv, schur, sInv, etc.) are all persisted via `persistAndTrack()` and added to `cachedMatrices`. But `cachedMatrices.foreach(cached => cached.blocks.unpersist(true))` is called **before** `bm` has been materialized by any action. Since the returned `bm` lineage still depends on those intermediates, unpersisting them forces Spark to either:

1. Recompute them from scratch (best case — severe performance regression), or
2. Fail entirely if parent RDDs have also been unpersisted (worst case — `SparkException`).

When `useCheckpoints = true`, the `bm.blocks.checkpoint()` call _schedules_ a checkpoint but the checkpoint only takes effect after an action forces materialization. The unpersist happens before that action.

The same pattern exists recursively — each `inverseInternal()` call creates its own `cachedMatrices` list and unpersists at the end, but the parent call may still need the child's intermediate results.

### Fix

Force materialization before unpersisting:

```scala
val bm = new BlockMatrix(allBlocks, rowsPerBlock, colsPerBlock, matrix.numRows(), matrix.numCols())
bm.blocks.persist(iterativeStorageLevel)
if (config.useCheckpoints) {
  bm.blocks.checkpoint()
}
bm.blocks.count()  // Force materialization BEFORE unpersisting intermediates
cachedMatrices.foreach(cached => cached.blocks.unpersist(true))
bm
```

Note: this adds one `count()` action per recursion level. The cost is minimal since it forces computation that would happen anyway when the caller consumes the result, and it ensures intermediates are available.

For `iterativeInverseInternal`, the same issue doesn't exist because intermediates are unpersisted within the loop after they're consumed, and the final `x` is returned directly. However, the caller must trigger an action on the result before any parent unpersists input data.

### Validation

Write a test that inverts a large matrix with `useCheckpoints = false` and calls `inverse.blocks.count()` on the result. Without the fix, this should show excessive recomputation or fail. With the fix, it should complete normally.

---

## 2. Suboptimal Alpha Scaling in Newton-Schulz

**Priority:** 🟠 High  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/api/Configs.scala`

### Problem

In `iterativeInverseInternal()`, lines ~455-458:

```scala
val norm1 = normOne()
val normInfValue = normInf()
val alpha = 1.0 / (norm1 * normInfValue)
```

The initial scaling `α = 1/(‖A‖₁ · ‖A‖_∞)` ensures convergence but is suboptimal. The optimal scaling for Newton-Schulz is `α* = 1/σ₁²` where `σ₁` is the largest singular value of `A`.

The relationship is: `σ₁² ≤ ‖A‖₁ · ‖A‖_∞`, so `α_current ≤ α_optimal`. Since the spectral radius `ρ(I - αAᵀA) = max|1 - ασ²|` is minimized at `α*`, using a smaller alpha leaves the spectral radius larger, causing slower convergence.

**Concrete example:** For a matrix with `‖A‖₁ = 10, ‖A‖_∞ = 10` but `σ₁ = 5` (common when rows/columns have balanced norms):

- `α_current = 1/100`, giving `ρ ≈ 1 - σₙ²/100`
- `α_optimal = 1/25`, giving `ρ ≈ 1 - σₙ²/25`
- The optimal scaling requires ~2× fewer iterations

**Cost of current approach:** Two extra Spark shuffles for `normOne()` (flatMap → reduceByKey → max) and `normInf()` (flatMap → reduceByKey → max).

### Fix Option A — Frobenius norm (cheapest, improved)

```scala
// Replace normOne/normInf with frobeniusNormSquared (already available, no shuffle)
val frobSq = frobeniusNormSquared()  // single map().sum(), no shuffle
val alpha = 1.0 / frobSq  // α = 1/‖A‖²_F ≥ 1/σ₁² since ‖A‖²_F = Σσᵢ² ≥ σ₁²
```

This is still suboptimal but tighter than `1/(‖A‖₁ · ‖A‖_∞)` in many cases, and it costs **zero shuffles** (just a map + reduce) vs two shuffles.

### Fix Option B — Power iteration estimate (most accurate)

```scala
// Run 2-3 power iterations to estimate σ₁²
def estimateLargestSquaredSingularValue(mat: BlockMatrix, midSplits: Int, iterations: Int = 3): Double = {
  // v₀ = random unit vector (block form)
  var v = randomBlockVector(mat.numCols().toInt, mat.colsPerBlock)
  var sigmaSqEstimate = 1.0
  for (_ <- 1 to iterations) {
    val Av = mat.multiply(v, midSplits)
    val AtAv = mat.transpose.multiply(Av, midSplits)
    sigmaSqEstimate = frobeniusNormSquared(AtAv) / frobeniusNormSquared(v)
    v = scaleBlockVector(AtAv, 1.0 / math.sqrt(sigmaSqEstimate))
  }
  sigmaSqEstimate
}

val alpha = 1.0 / estimateLargestSquaredSingularValue(matrix, midSplits)
```

Costs 2×iterations distributed multiplies but can save many Newton-Schulz iterations.

### Fix Option C — Gelfand's estimate (free, from CANS paper)

```scala
// Gelfand's formula: σ₁(A) ≤ ‖(AᵀA)^k‖_F^{1/(2k)}
// During the first Newton-Schulz iteration, we already compute AᵀA·X₀ = α·AᵀA·Aᵀ
// which is (α^{1/2}·Aᵀ)·(α^{1/2}·A). Use this for a free estimate.
//
// After the first iteration: R = I - αAᵀA, so ‖R‖_F² ≈ σ₁²·α²·(n-1) for initial estimate
// More precisely: α₂ = α / ‖R + I‖_F² · n  (refinement)
```

This costs zero extra multiplications since the residual R is already computed.

### Config Change

Add `alphaStrategy` to `IterativeInverseConfig`:

```scala
sealed trait AlphaStrategy
object AlphaStrategy {
  case object NormProduct extends AlphaStrategy    // current: 1/(‖A‖₁·‖A‖_∞)
  case object Frobenius extends AlphaStrategy      // 1/‖A‖²_F
  case object PowerIteration extends AlphaStrategy  // 1/σ₁² via power iteration
  case object Adaptive extends AlphaStrategy        // Gelfand estimate from first iteration
}

final case class IterativeInverseConfig(
  order: Int = 2,
  maxIter: Int = 30,
  tolerance: Double = 1e-15,
  useCheckpoints: Boolean = true,
  checkpointEvery: Int = 5,
  midSplits: Int = 1,
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
  alphaStrategy: AlphaStrategy = AlphaStrategy.Frobenius  // changed default
)
```

---

## 3. Single-Pass Quadrant Split — COMPLETED

**Priority:** 🟠 High  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

### Problem

In `splitQuadrants()`, lines ~98-112:

```scala
val e = withName(new BlockMatrix(
  blocks.filter { case ((i, j), _) => i < m && j < m }, ...), "E")
val f = withName(new BlockMatrix(
  blocks.filter { case ((i, j), _) => i < m && j >= m }.map { ... }, ...), "F")
val g = withName(new BlockMatrix(
  blocks.filter { case ((i, j), _) => i >= m && j < m }.map { ... }, ...), "G")
val h = withName(new BlockMatrix(
  blocks.filter { case ((i, j), _) => i >= m && j >= m }.map { ... }, ...), "H")
```

Each `filter` scans the entire input RDD. With 4 filters, the data is read **4 times**. At recursion depth d, the base-level data is read 4^d times total (through nested calls). For a 4-level recursion, that's 256 scans of the original data.

Even with Spark's caching, each filter creates a separate RDD dependency. The 4 filter operations don't share computation.

### Fix

Use a single `map` + `partitionBy` to classify and redistribute blocks by quadrant in one pass:

```scala
private def splitQuadrants(m: Int, splitSize: Long, res: Long): BlockQuadrants = {
  val rowsPerBlock = matrix.rowsPerBlock
  val colsPerBlock = matrix.colsPerBlock
  val blocks = matrix.blocks

  // Single-pass: tag each block with its quadrant, then partition once
  val tagged = blocks.map { case ((i, j), block) =>
    val qi = if (i < m) 0 else 1
    val qj = if (j < m) 0 else 1
    val ni = i - qi * m.toLong
    val nj = j - qj * m.toLong
    ((qi, qj), (ni, nj, block))
  }.persist(StorageLevel.MEMORY_AND_DISK_SER)

  // Force single scan before filtering
  tagged.count()

  val eBlocks = tagged.filter(_._1 == (0, 0))
    .map { case (_, (ni, nj, block)) => ((ni, nj), block) }
  val fBlocks = tagged.filter(_._1 == (0, 1))
    .map { case (_, (ni, nj, block)) => ((ni, nj), block) }
  val gBlocks = tagged.filter(_._1 == (1, 0))
    .map { case (_, (ni, nj, block)) => ((ni, nj), block) }
  val hBlocks = tagged.filter(_._1 == (1, 1))
    .map { case (_, (ni, nj, block)) => ((ni, nj), block) }

  val result = BlockQuadrants(
    withName(new BlockMatrix(eBlocks, rowsPerBlock, colsPerBlock, splitSize, splitSize), "E"),
    withName(new BlockMatrix(fBlocks, rowsPerBlock, colsPerBlock, splitSize, res), "F"),
    withName(new BlockMatrix(gBlocks, rowsPerBlock, colsPerBlock, res, splitSize), "G"),
    withName(new BlockMatrix(hBlocks, rowsPerBlock, colsPerBlock, res, res), "H")
  )

  tagged.unpersist()
  result
}
```

**Important:** The `tagged.count()` is needed to materialize the cached RDD before the 4 `filter` calls read from it. Without it, the filters would each re-scan the original `blocks` RDD. This pairs with the fix for Point 1 (unpersist-before-evaluation).

Alternative: avoid `count()` by relying on the persisted intermediates being consumed downstream, but that requires careful lifecycle management.

### Expected improvement

For a matrix with N non-zero blocks and recursion depth d:

- **Before:** 4 × (2^d) scans of the data across all recursion levels
- **After:** 1 scan per level, with cached `tagged` RDD shared across 4 filters

This is roughly a **2-4× I/O reduction** in the quadrant-splitting phase.

---

## 4. LU-Based Local Inversion (Replace SVD Base Case)

**Priority:** 🟠 High  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

### Problem

`localInverse()` delegates to `svdInverse()`, which:

1. Converts `BlockMatrix` → `IndexedRowMatrix` (shuffle)
2. Calls `computeSVD(n, computeU = true, rCond = 0)` (full SVD — ~⁴⁄₃ n³ flops)
3. Inverts all singular values, even near-zero ones
4. Reconstructs via `U · diag(σ⁻¹) · Vᵀ` (extra multiply + transpose + block conversion)

For the base case of the recursive inversion (matrices up to `limit=4096` elements by default), SVD is ~2× more expensive than LU decomposition.

SVD cost: `⁴⁄₃ n³ + O(n²)` for the decomposition itself, plus `O(n³)` for reconstruction.
LU cost: `²⁄₃ n³ + O(n²)` for the decomposition, `²⁄₃ n³` for the solve.

### Fix Option A — Commons Math LU (zero new dependencies)

Commons Math (`commons-math3`) is already a transitive dependency of Apache Spark:

```scala
import org.apache.commons.math3.linear.{LUDecomposition, Array2DRowRealMatrix}

def luInverse(): BlockMatrix = {
  val local = matrix.toLocalMatrix()
  val n = local.numRows
  require(n == local.numCols, s"Matrix must be square for inversion, got ${local.numRows}x${local.numCols}")

  val realMatrix = new Array2DRowRealMatrix(local.toArray.map(_.toDouble), false)
  val luDecomp = new LUDecomposition(realMatrix, 1e-20)  // singular threshold

  if (!luDecomp.getSolver.isNonSingular) {
    throw new IllegalArgumentException(
      s"Matrix is singular (det ≈ 0). Consider using pseudo-inverse or SVD inverse with regularization.")
  }

  val inverseData = luDecomp.getSolver.getInverse.getData
  val result = new DenseMatrix(n, n, inverseData.flatten)
  // Convert back to BlockMatrix with same block sizing...
  sc.parallelize(Seq(((0, 0), result))).toBlockMatrix(matrix.rowsPerBlock, matrix.colsPerBlock)
}
```

### Fix Option B — Breeze LU (already in test classpath)

Breeze is already a test dependency (`scalatest` pulls it in). To use it in production, add it to `libraryDependencies`:

```scala
"org.scalanlp" %% "breeze" % "2.0"  // in core's libraryDependencies
```

```scala
import breeze.linalg.{DenseMatrix => BDM, inv => BINV}

def luInverse(): BlockMatrix = {
  val local = matrix.toLocalMatrix()
  val n = local.numRows
  val breezeMat = new BDM[Double](n, n, local.toArray)
  val invBreeze = BINV(breezeMat)  // Uses LU decomposition internally
  // Convert back...
}
```

### Fix Option C — Combined approach with condition-number-based fallback

```scala
def localInverse(): BlockMatrix = {
  val local = matrix.toLocalMatrix()
  val n = local.numRows

  // For small matrices, LU decomposition is faster and sufficient
  // For ill-conditioned matrices, fall back to SVD
  val conditionEstimate = estimateConditionNumber(local)

  if (conditionEstimate < 1e10) {
    luInverse()
  } else {
    logger.warn("Matrix appears ill-conditioned (κ ≈ {}). Falling back to SVD inverse.", conditionEstimate)
    svdInverse()
  }
}
```

### Build change for Option A

No build change needed — `commons-math3` is already available via Spark's transitive dependencies. However, it's better to add it explicitly:

```scala
// In build.sbt, core project's libraryDependencies:
"org.apache.commons" % "commons-math3" % "3.6.1"  // already a transitive dep of Spark
```

For Option B, add Breeze as a production dependency:

```scala
"org.scalanlp" %% "breeze" % "2.0"
```

### Expected improvement

- ~2× speedup for the base case of recursive inversion
- Better numerical stability for well-conditioned matrices (LU with partial pivoting)
- SVD remains available as fallback for ill-conditioned matrices

---

## 5. A Caching in Iterative Loop — COMPLETED

**Priority:** 🟢 Low  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

**Implementation:**

- Added `matrix.blocks.count()` after `persist()` in `iterativeInverseInternal` to force cache materialization before the iterative loop starts
- Input matrix A is now guaranteed cached before the first multiply
- Skipped `pseudoInverse` — no loop, no repeated recomputation benefit

**Validation:**

- All 56 core tests pass
- bench/compile passes

---

**Priority:** 🟢 Low  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

### Problem

In `iterativeInverseInternal()`, the matrix `A` is used in every iteration via `matrix.multiply(x, midSplits)`. The code handles persisting:

```scala
val shouldPersistInput = matrix.blocks.getStorageLevel == StorageLevel.NONE
if (shouldPersistInput) {
  matrix.blocks.persist(storageLevel)
}
```

But there are two issues:

1. **If A was already persisted**, `shouldPersistInput` is false and we don't add our own persist. This is correct for memory management but means we rely on the caller having persisted A appropriately.

2. **Shuffle reuse**: Each `matrix.multiply(x, midSplits)` creates a new shuffle for A's blocks even though A hasn't changed. Spark's DAG scheduler can't automatically detect that the same matrix is being shuffled identically across iterations, so it creates new shuffle files each time.

The real performance bottleneck is the **shuffle**, not the persistence. Each `multiply` triggers two shuffles (one for the left operand, one for the right), and the left operand (`matrix`) is shuffled identically every iteration.

### Fix Option A — Pre-partition the matrix

```scala
// Before the loop, partition A once using the same partitioner
// that multiply will use internally
val prePartitionedA = new BlockMatrix(
  matrix.blocks.partitionBy(new HashPartitioner(numPartitions)),
  matrix.rowsPerBlock, matrix.colsPerBlock, matrix.numRows(), matrix.numCols()
)
prePartitionedA.blocks.persist(storageLevel)
prePartitionedA.blocks.count()  // Force materialization
```

This doesn't directly help because `BlockMatrix.multiply` creates its own partitioned RDDs internally and doesn't accept pre-partitioned inputs.

### Fix Option B — Custom multiply with shuffle file reuse

Implement a version of the iterative loop that maintains accumulated shuffle files for A and reuses them:

```scala
// First iteration:
val ax = matrix.multiply(x, midSplits)  // creates shuffle files for A and X

// Subsequent iterations:
// x changes but A doesn't. If we could reuse A's shuffle map output,
// we'd save one shuffle per iteration. This requires lower-level
// Spark RDD programming (saveAsHadoopDataset, etc.) or maintaining
// reference to the ShuffledRDD.
```

This is complex and would require overriding `BlockMatrix.multiply`. The practical benefit depends on the cluster and data size.

### Practical recommendation

For now, ensure A is persisted and document that callers should persist A before calling `iterativeInverse`:

```scala
val shouldPersistInput = matrix.blocks.getStorageLevel == StorageLevel.NONE
if (shouldPersistInput) {
  matrix.blocks.persist(storageLevel)
  matrix.blocks.count()  // Force immediate materialization
}
```

The `count()` is cheap if A is small relative to the iterative computations, and it ensures A is in cache before the loop starts.

---

## 6. Adaptive Iteration Order

**Priority:** 🟡 Medium  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/api/Configs.scala`

### Problem

The current implementation uses a fixed `order` for all iterations:

```scala
val order = config.order  // Fixed for entire loop
while (iter < config.maxIter && !converged) {
  val (correction, extraPowers) = buildHyperpowerCorrection(eye, residual, order, midSplits, storageLevel)
  // ...
}
```

From the analysis:

- **High order (3, 4, 5)** is better when far from the solution (large residual) because the correction polynomial captures more information per iteration
- **Low order (2, i.e., Newton-Schulz)** is more efficient when close to convergence because higher-order terms are negligible but still cost matrix multiplications

The cost per iteration scales with order:

- Order 2: 2 multiplies (1 for R², 1 for X·C)
- Order 3: 3 multiplies (1 for R², 1 for R³, 1 for X·C)
- Order 4: 4 multiplies (1 for R², 1 for R⁴ via squaring, 1 for R³= R⁴/R, 1 for X·C)
- Order 5: 4 multiplies (via repeated squaring) + 1 for X·C

The convergence rate per iteration is approximately ρ^order where ρ is the spectral radius of the residual. So the **efficiency** (convergence per multiply) is:

- Order 2: ρ² / 2 per multiply
- Order 3: ρ³ / 3 per multiply
- When ρ ≈ 0.9: order 2 gives 0.81/2 = 0.405, order 3 gives 0.729/3 = 0.243 → order 3 converges faster per multiply
- When ρ ≈ 0.1: order 2 gives 0.01/2 = 0.005, order 3 gives 0.001/3 = 0.000333 → order 2 converges faster per multiply

### Fix — Adaptive order with metric thresholds

```scala
final case class IterativeInverseConfig(
  order: Int = 2,
  maxIter: Int = 30,
  tolerance: Double = 1e-15,
  useCheckpoints: Boolean = true,
  checkpointEvery: Int = 5,
  midSplits: Int = 1,
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,
  alphaStrategy: AlphaStrategy = AlphaStrategy.Frobenius,
  adaptiveOrder: Boolean = false  // NEW: enable adaptive order selection
)
```

Implementation:

```scala
var currentOrder = config.order
var metric = Double.MaxValue

while (iter < config.maxIter && !converged) {
  // Adaptive order: high order when far, low order when close
  if (config.adaptiveOrder) {
    currentOrder = if (metric > 0.5) math.min(config.order, 5)
    else if (metric > 0.1) math.min(config.order, 4)
    else if (metric > 0.01) math.min(config.order, 3)
    else 2  // Near convergence: use Newton-Schulz (order 2)
  }

  val (correction, extraPowers) = buildHyperpowerCorrection(eye, residual, currentOrder, midSplits, storageLevel)
  // ...
}
```

Alternatively, implement CANS-style (from Grishina et al. 2026) Chebyshev-optimal polynomial selection where the polynomial coefficients are adapted based on estimated singular value bounds at each iteration.

### CANS Integration (more ambitious)

The CANS paper shows that for order-3 polynomials, the optimal coefficients are given by Proposition 3.3:

```
p_{2,a,b}(x) = (2/(a² + ab + b²))^(3/2) / (2(a²+ab+b²)^(3/2) + a²b + b²a) · (x - x³)
```

where `[a, b]` is the current spectral range estimated from the iteration. This gives provably optimal convergence per iteration for degree-3 polynomials.

This would require modifying `buildHyperpowerCorrection` to accept custom polynomial coefficients instead of the fixed binomial expansion `(I + R + R² + ... + R^{p-1})`.

---

## 7. Convergence Check Frequency — COMPLETED

**Priority:** 🟡 Medium  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/api/Configs.scala`

### Problem

Every iteration computes:

```scala
val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
```

`frobeniusNormSquared()` triggers a Spark action (`.sum()`), which means every iteration submits a separate Spark job just to check convergence. This has two costs:

1. **Spark job scheduling overhead**: Each action triggers DAG scheduling, task serialization, and result collection. For small matrices, this overhead can dominate the actual computation.
2. **Unnecessary materialization**: The `residual` matrix is materialized just for the norm check, even though it would be needed anyway for the hyperpower correction.

However, the silver lining is that the `frobeniusNormSquared()` call forces `residual` to be materialized before `buildHyperpowerCorrection` needs it, which can be beneficial for caching.

### Fix Option A — Check every K iterations

Add a config parameter:

```scala
final case class IterativeInverseConfig(
  // ... existing fields ...
  convergenceCheckInterval: Int = 1  // Check convergence every N iterations
)
```

```scala
var lastMetric = Double.MaxValue
while (iter < config.maxIter && !converged) {
  iter += 1
  val ax = matrix.multiply(x, midSplits)
  ax.blocks.persist(storageLevel)

  val residual = eye.subtract(ax)
  residual.blocks.persist(storageLevel)

  // Check convergence only every K iterations
  if (iter % config.convergenceCheckInterval == 0 || iter == 1) {
    val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
    lastMetric = metric
    if (metric < config.tolerance) converged = true
  }

  if (!converged) {
    // continue iteration...
  }
  residual.blocks.unpersist(true)
  ax.blocks.unpersist(true)
}
```

The risk is running 1-2 extra iterations when convergence happens between checks, but the savings from fewer Spark actions can be significant.

### Fix Option B — Cheap convergence proxy

Instead of the full Frobenius norm, use a cheaper estimate:

```scala
// Option B1: Diagonal-only residual check
// The identity matrix has 1s on the diagonal, so check ||diag(I - A·X)||_2
val diagonalResidualNorm = ax.blocks.filter { case ((i, j), _) => i == j }
  .map { case ((i, _), mat) =>
    val arr = mat.toArray
    val nRows = mat.numRows
    var sum = 0.0
    for (d <- 0 until math.min(mat.numRows, mat.numCols)) {
      val v = arr(d + d * nRows)
      sum += (1.0 - v) * (1.0 - v)
    }
    sum
  }.sum()

// Option B2: Sampled residual norm
// Random projection: ||I - A·X|| ≈ ||(I - A·X)·v|| for a random vector v
val randomVec = MatrixInternals.randomBlockVector(n.toInt, matrix.rowsPerBlock, matrix.colsPerBlock)
val sampleResidual = residual.multiply(randomVec, midSplits)  // single multiply
val sampleNorm = new BlockMatrixOps(sampleResidual).frobeniusNormSquared() / n
```

Option B1 only scans diagonal blocks (O(n) instead of O(n²)) and Option B2 costs one matrix-vector multiply (O(n²) but no shuffle, since BlockMatrix × local vector can be done without shuffling if the vector is broadcast).

---

## 8. CoordinateMatrix Always Densifies

**Priority:** 🟢 Low (high impact for sparse matrices, but high implementation effort)  
**Effort:** High  
**Files:** `core/src/main/scala/sparkinverse/coordinate/CoordinateMatrixOps.scala`

### Problem

```scala
private def toBlock = matrix.toBlockMatrix(defaultBlockSize, defaultBlockSize)

private def defaultBlockSize: Int = {
  val maxSize = math.min(matrix.numRows(), matrix.numCols())
  math.max(1, math.min(1024L, maxSize).toInt)
}
```

Every `CoordinateMatrixOps` operation converts to `BlockMatrix` (dense blocks) via `toBlockMatrix()`. For a sparse 10000×10000 matrix with 0.1% density:

- Original data: ~100KB as `CoordinateMatrix`
- After `toBlockMatrix(1024, 1024)`: ~100 dense 1024×1024 blocks × 8MB each = ~800MB

The `defaultBlockSize` of 1024 creates very large dense blocks regardless of sparsity. For sparse data, smaller blocks or sparse representations would be much more efficient.

### Fix Option A — Adaptive block size based on sparsity

```scala
private def toBlock = {
  val numEntries = matrix.entries.count()  // one action — expensive but informs block size
  val nnz = numEntries.toDouble
  val totalElements = matrix.numRows() * matrix.numCols()
  val density = nnz / totalElements

  val blockSize = if (density < 0.001) {
    // Very sparse: small blocks to avoid dense waste
    math.max(1, math.min(64, math.min(matrix.numRows(), matrix.numCols()).toInt))
  } else if (density < 0.05) {
    math.max(1, math.min(256, math.min(matrix.numRows(), matrix.numCols()).toInt))
  } else if (density < 0.3) {
    math.max(1, math.min(512, math.min(matrix.numRows(), matrix.numCols()).toInt))
  } else {
    math.max(1, math.min(1024, math.min(matrix.numRows(), matrix.numCols()).toInt))
  }

  matrix.toBlockMatrix(blockSize, blockSize)
}
```

The `count()` action is expensive, but it only happens once per operation chain.

### Fix Option B — Sparse Newton-Schulz path (major implementation)

Implement the iterative inverse entirely in `CoordinateMatrix` format, avoiding densification:

```scala
def iterativeInverse(config: IterativeInverseConfig): CoordinateMatrix = {
  val density = estimateDensity()
  if (density < 0.05) {
    // Sparse path: stay in coordinate form
    val alpha = coordinateFrobeniusNormSquared()  // cheap for sparse
    var x = transpose().scalarMultiply(alpha)       // CoordinateMatrix
    for (iter <- 1 to config.maxIter) {
      val ax = multiply(x)                          // CoordinateMatrix × CoordinateMatrix
      val residual = identity().subtract(ax)         // CoordinateMatrix
      if (converged(residual)) return x
      val correction = buildCoordHyperpowerCorrection(residual, config.order)
      x = x.multiply(correction)                    // CoordinateMatrix × CoordinateMatrix
    }
    x
  } else {
    // Dense path: convert to BlockMatrix (existing implementation)
    fromBlock(_.iterativeInverse(config))
  }
}
```

This leverages the existing `multiply`, `add`, `subtract`, `transpose` operations on `CoordinateMatrix` that are already implemented in `CoordinateMatrixOps`. The key missing pieces are:

- `identity(): CoordinateMatrix` (create identity in coordinate form — already available in `MatrixInternals.eyeCoordinateMatrix`)
- `buildCoordHyperpowerCorrection`: CoordinateMatrix version of residual power computation
- Convergence check using coordinate Frobenius norm (already available)

### Recommendation

Start with Option A (adaptive block size) as it's low-effort and handles the worst cases. Option B is a larger project that should be prioritized based on user demand for sparse matrix support.

---

## 9: Partition Count Explosion in Recursive Inverse

**Priority:** 🟡 Medium  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/api/Configs.scala`

### Problem

In `inverseInternal()`, the final assembly:

```scala
val unionedBlocks = sc.union(
  topLeft.blocks,           // P₁ partitions
  shiftAndScaleBlocks(eInvFSInv.blocks, colOffset = m, scale = -1.0),  // P₂ partitions
  shiftAndScaleBlocks(sInvGeInv.blocks, rowOffset = m, scale = -1.0),  // P₃ partitions
  shiftAndScaleBlocks(sInv.blocks, rowOffset = m, colOffset = m)       // P₄ partitions
)
val defaultOutputParts = math.max(numParts, math.min(unionedBlocks.getNumPartitions, numParts * 2))
val outputParts = config.targetOutputPartitions.getOrElse(defaultOutputParts)
val allBlocks = MatrixInternals.maybeCoalesceNoShuffle(unionedBlocks, outputParts, config.unionCoalesceThreshold)
```

The `union` of 4 RDDs creates an RDD with P₁ + P₂ + P₃ + P₄ partitions. At recursion depth d, each sub-problem's output has 4 × (parent partitions) partitions from the sub-union. The total partition count grows exponentially.

With `unionCoalesceThreshold = 8`, `maybeCoalesceNoShuffle` only triggers when `currentPartitions - targetPartitions > 8`. For example:

- Input: 8 partitions
- E quadrant: 8 partitions (from filter), F: 8, G: 8, H: 8
- After sub-inversions and multiplications: each intermediate has ~8-16 partitions
- Union: ~40-64 partitions
- Threshold check: `64 - max(8, min(64, 16)) = 64 - 16 = 48 > 8` → coalesce to 16
- But `coalesce(16, shuffle=false)` on 64 partitions can only merge adjacent partitions without shuffling, which may produce unbalanced data distribution

At deeper recursion levels, the partition count compounds. A 4096×4096 matrix with `limit=256` has ~4 levels of recursion, producing intermediate RDDs with hundreds of partitions.

### Fix Option A — Aggressive output partition targeting

```scala
final case class RecursiveInverseConfig(
  // ... existing fields ...
  targetOutputPartitions: Option[Int] = None  // Already exists
  // Add auto-computation if not specified
)
```

Add a smarter default for `targetOutputPartitions`:

```scala
val autoTargetPartitions = {
  val n = matrix.numRows()
  val blockSize = matrix.rowsPerBlock
  val numBlocks = (n / blockSize) * (n / blockSize)
  // Target ~128KB-1MB per partition
  val targetElementsPerPartition = 128 * 1024 / 8  // 128KB / 8 bytes per double
  val targetPartitions = math.max(2, numBlocks / (targetElementsPerPartition / (blockSize * blockSize)))
  math.min(targetPartitions, numParts * 4)  // Don't exceed 4x input parallelism
}
val outputParts = config.targetOutputPartitions.getOrElse(autoTargetPartitions)
```

### Fix Option B — Use repartition (with shuffle) for better balancing

```scala
val allBlocks = if (unionedBlocks.getNumPartitions > outputParts * 2) {
  // Too many partitions: reshuffle for balance
  unionedBlocks.repartition(outputParts)
} else if (unionedBlocks.getNumPartitions > outputParts + config.unionCoalesceThreshold) {
  // Moderate excess: coalesce without shuffle
  unionedBlocks.coalesce(outputParts, shuffle = false)
} else {
  unionedBlocks
}
```

The `repartition` (which shuffles) is more expensive than `coalesce(no-shuffle)` but produces balanced partitions, which is critical for subsequent operations.

### Fix Option C — Single-rdd quadrant assembly

Instead of union of 4 RDDs, emit all 4 quadrants into a single RDD with a quadrant tag, then use a single partitioning:

```scala
// All 4 quadrants in one RDD, partitioned by quadrant
val allBlocks = quadrantBlocks.partitionBy(new HashPartitioner(outputParts))
  .map { case (key, ((ni, nj), block)) => ((ni, nj), block) }
```

This produces a single well-partitioned RDD with exactly `outputParts` partitions, no coalescing needed.

---

## 10. O(n³) Block Multiplication (Strassen Opportunity)

**Priority:** 🟢 Low (high impact for very large matrices, but high implementation effort)  
**Effort:** High  
**Files:** New file in `core/src/main/scala/sparkinverse/block/StrassenMultiply.scala`

### Problem

The recursive inversion performs standard O(n³) matrix multiplications at each level using either Spark's `BlockMatrix.multiply()` or the custom `squareMultiply()`. The SPIN paper (arXiv:1801.04723) demonstrated that using Strassen's algorithm for the multiplication sub-problems reduces total inversion complexity from O(n³) to O(n^log₂7) ≈ O(n^2.807).

Strassen's key insight: two n×n matrices can be multiplied using only 7 multiplications of n/2×n/2 sub-problems (instead of 8), giving:

```
T(n) = 7·T(n/2) + O(n²)
T(n) = O(n^log₂7) ≈ O(n^2.807)
```

For the inversion context, this applies to all intermediate multiplications in the Schur complement formula.

### Implementation Sketch

```scala
object StrassenMultiply {
  val STRASSEN_THRESHOLD = 512  // Switch to standard multiply below this size

  def multiply(a: BlockMatrix, b: BlockMatrix, midSplits: Int): BlockMatrix = {
    if (a.numRows() <= STRASSEN_THRESHOLD || a.numCols() <= STRASSEN_THRESHOLD) {
      return a.multiply(b, midSplits)  // Base case: standard multiply
    }

    // Split a and b into 2×2 block form
    val (a11, a12, a21, a22) = splitQuadrants(a)
    val (b11, b12, b21, b22) = splitQuadrants(b)

    // Strassen's 7 multiplications
    val m1 = multiply(a11, b12.subtract(b22, midSplits), midSplits)    // M1 = A11·(B12 - B22)
    val m2 = multiply(a11.add(a12, midSplits), b22, midSplits)        // M2 = (A11 + A12)·B22
    val m3 = multiply(a21.add(a22, midSplits), b11, midSplits)        // M3 = (A21 + A22)·B11
    val m4 = multiply(a22, b21.subtract(b11, midSplits), midSplits)   // M4 = A22·(B21 - B11)
    val m5 = multiply(a11.add(a22, midSplits),
                       b11.add(b22, midSplits), midSplits)              // M5 = (A11 + A22)·(B11 + B22)
    val m6 = multiply(a12.subtract(a22, midSplits),
                       b21.add(b22, midSplits), midSplits)             // M6 = (A12 - A22)·(B21 + B22)
    val m7 = multiply(a11.subtract(a21, midSplits),
                       b11.add(b12, midSplits), midSplits)             // M7 = (A11 - A21)·(B11 + B12)

    // Combine results
    val c11 = m5.add(m4, midSplits).subtract(m2, midSplits).add(m6, midSplits)
    val c12 = m1.add(m2, midSplits)
    val c21 = m3.add(m4, midSplits)
    val c22 = m5.add(m1, midSplits).subtract(m3, midSplits).subtract(m7, midSplits)

    combineQuadrants(c11, c12, c21, c22)
  }
}
```

### Considerations

1. **Crossover point**: Strassen's algorithm has higher constant factors than standard multiplication. The crossover where it becomes faster depends on the distributed setting — network overhead from additional intermediate RDDs may dominate. SPIN paper suggests n > 2000-5000 in distributed settings.

2. **Numerical stability**: Strassen's algorithm has worse numerical stability than standard multiplication, with error bounds that grow as O(n^log₂7) vs O(n²) for standard. This is usually acceptable for double precision.

3. **Cache friendliness**: The recursive splitting creates many intermediate BlockMatrix objects. Each intermediate needs persistence management (Points 1 and 9 compound here).

4. **Integration**: Could be used as a drop-in replacement for `matrix.multiply()` in the recursive inversion, controlled by a config flag:

```scala
final case class RecursiveInverseConfig(
  // ... existing fields ...
  useStrassenMultiply: Boolean = false,  // NEW
  strassenThreshold: Int = 2000          // NEW: minimum dimension for Strassen
)
```

### References

- Strassen, V. (1969). "Gaussian elimination is not optimal." _Numerische Mathematik_, 13(4), 354-356.
- Misra, C. (2018). "SPIN: A Fast and Scalable Matrix Inversion Method in Apache Spark." arXiv:1801.04723.
- Ballard, G. et al. (2012). "Communication-Optimal Parallel Algorithm for Strassen's Matrix Multiplication."

---

## 11. No Preconditioning for Iterative Method

**Priority:** 🟢 Low (enables convergence for wider matrix classes, but medium implementation effort)  
**Effort:** Medium  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/api/Configs.scala`

### Problem

The Newton-Schulz iteration converges when `ρ(I - αAᵀA) < 1`, which requires `σε(AᵀA) ⊂ (0, 2/α)`. The convergence rate is:

```
convergence per iteration ≈ (1 - 1/κ²(A))^order
```

where `κ(A) = σ₁/σₙ` is the condition number.

For condition numbers > 10³, convergence is extremely slow:

- κ = 10²: ρ ≈ 1 - 10⁻⁴ → needs ~10⁴ iterations for order 2
- κ = 10⁴: ρ ≈ 1 - 10⁻⁸ → needs ~10⁸ iterations for order 2

The current implementation simply hits `maxIter` with a warning for such matrices.

### Fix: Jacobi (diagonal) preconditioning

The simplest preconditioner scales A by its diagonal, transforming A → D⁻¹A where D = diag(A):

```scala
final case class IterativeInverseConfig(
  // ... existing fields ...
  preconditioner: Preconditioner = Preconditioner.None  // NEW
)

sealed trait Preconditioner
object Preconditioner {
  case object None extends Preconditioner
  case object Jacobi extends Preconditioner      // Scale by diagonal: D⁻¹A
  case object BlockJacobi extends Preconditioner // Scale by diagonal blocks
}
```

Implementation for Jacobi:

```scala
private def applyJacobiPreconditioner(matrix: BlockMatrix): (BlockMatrix, BlockMatrix, BlockMatrix) = {
  // Extract diagonal D = diag(A), compute D⁻¹
  val diagonalBlocks = matrix.blocks.filter { case ((i, j), _) => i == j }

  val dInverseBlocks = diagonalBlocks.map { case ((i, j), mat) =>
    // Invert the diagonal of each block
    val arr = mat.toArray
    val nRows = mat.numRows
    val nCols = mat.numCols
    val invArr = new Array[Double](arr.length)
    java.util.Arrays.fill(invArr, 0.0)
    for (d <- 0 until math.min(nRows, nCols)) {
      invArr(d + d * nRows) = 1.0 / arr(d + d * nRows)
    }
    ((i, j), new DenseMatrix(nRows, nCols, invArr))
  }

  val dInverse = new BlockMatrix(dInverseBlocks, matrix.rowsPerBlock, matrix.colsPerBlock,
    matrix.numRows(), matrix.numCols())

  // Compute D⁻¹A (preconditioned matrix)
  val preconditioned = dInverse.multiply(matrix, midSplits)

  // Return (D⁻¹A, D⁻¹, A) so that A⁻¹ = D⁻¹ · (D⁻¹A)⁻¹
  (preconditioned, dInverse, matrix)
}
```

Then in `iterativeInverseInternal`:

```scala
val (workingMatrix, leftInverse) = config.preconditioner match {
  case Preconditioner.None => (matrix, None)
  case Preconditioner.Jacobi =>
    val (precond, dInv, _) = applyJacobiPreconditioner(matrix)
    (precond, Some(dInv))
}

// ... run Newton-Schulz on workingMatrix ...

// Post-process: A⁻¹ = D⁻¹ · (D⁻¹A)⁻¹
leftInverse match {
  case Some(dInv) => dInv.multiply(result, midSplits)
  case None => result
}
```

For **Block Jacobi**, invert each diagonal block of A (small dense inversion) rather than just the diagonal entries. This provides better preconditioning for block-structured matrices.

### Expected improvement

For a diagonally dominant matrix with κ = 10⁴:

- Without preconditioning: essentially no convergence in 30 iterations
- With Jacobi preconditioning: κ(D⁻¹A) ≈ √κ(A) ≈ 100, convergence in ~10-20 iterations

---

## 12. No Regularization for Pseudo-Inverse — COMPLETED

**Priority:** 🟡 Medium  
**Effort:** Low  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, `core/src/main/scala/sparkinverse/api/Configs.scala`

### Problem

The pseudo-inverse computes:

```scala
// Left: (AᵀA)⁻¹Aᵀ
// Right: Aᵀ(AAᵀ)⁻¹
```

The Gram matrix `AᵀA` (for left) or `AAᵀ` (for right) has condition number `κ(A)²`, which is much worse than `κ(A)`. For example, if A has κ = 10³, then AᵀA has κ = 10⁶, and the recursive inverse will struggle to converge.

Even for full-rank matrices, near-singular Gram matrices amplify noise from small singular values. The result is numerically unstable.

### Fix: Tikhonov regularization parameter

```scala
final case class RecursiveInverseConfig(
  limit: Int = 4096,
  midSplits: Int = 1,
  useCheckpoints: Boolean = true,
  targetOutputPartitions: Option[Int] = None,
  unionCoalesceThreshold: Int = 8,
  minBlockSizeForPersistence: Int = 1000000,
  regularizationLambda: Double = 0.0  // NEW: Tikhonov regularization parameter
)
```

Implementation in `pseudoInverse`:

```scala
def pseudoInverse(side: PseudoInverseSide, config: RecursiveInverseConfig): BlockMatrix = {
  val at = matrix.transpose
  val persistedAt = persistIfNeeded(at.blocks, iterativeStorageLevel)
  try {
    val midSplits = math.max(1, config.midSplits)
    val gram = side match {
      case PseudoInverseSide.Left  => at.multiply(matrix, midSplits)
      case PseudoInverseSide.Right => matrix.multiply(at, midSplits)
    }
    val persistedGram = persistIfNeeded(gram.blocks, iterativeStorageLevel)
    try {
      val gramInverted = if (config.regularizationLambda > 0.0) {
        // Tikhonov regularization: invert (Gram + λI) instead of Gram
        val regGram = gram.add(
          MatrixInternals.eyeBlockMatrix(
            gram.numRows(), config.regularizationLambda,
            gram.rowsPerBlock, gram.colsPerBlock,
            iterativeStorageLevel, gram
          )
        )
        new BlockMatrixOps(regGram).inverse(config)
      } else {
        new BlockMatrixOps(gram).inverse(config)
      }

      side match {
        case PseudoInverseSide.Left  => gramInverted.multiply(at, midSplits)
        case PseudoInverseSide.Right => at.multiply(gramInverted, midSplits)
      }
    } finally {
      if (persistedGram) gram.blocks.unpersist(false)
    }
  } finally {
    if (persistedAt) at.blocks.unpersist(false)
  }
}
```

This changes `(AᵀA)⁻¹Aᵀ` to `(AᵀA + λI)⁻¹Aᵀ` which is the **ridge regression / Tikhonov** regularized pseudo-inverse. It:

- Damps small singular values: (`σᵢ/(σᵢ² + λ)` instead of `1/σᵢ`)
- Improves condition number from `κ(A)²` to `(σ₁² + λ)/(σₙ² + λ)`
- Introduces bias proportional to `λ` (larger λ → more stable but more biased)

### Guidance for choosing λ

- `λ = 0`: Standard pseudo-inverse (current behavior)
- `λ = ε·σ₁²`: Minimal regularization (ε = machine epsilon), effective for numerical stability
- `λ = σₙ²` (estimated): Near-optimal for noise suppression
- GCV (Generalized Cross-Validation): Automatically select λ, but expensive to compute

A practical default could be `λ = 1e-10 * ‖AᵀA‖_F²/n²`, which is negligible for well-conditioned matrices but significant for near-singular ones.

Also add a condition number warning:

```scala
// After computing gram:
val gramDiagMin = gram.blocks.filter { case ((i, j), _) => i == j }
  .map { case ((i, _), mat) =>
    val arr = mat.toArray
    var minDiag = Double.MaxValue
    for (d <- 0 until math.min(mat.numRows, mat.numCols)) {
      minDiag = math.min(minDiag, math.abs(arr(d + d * mat.numRows)))
    }
    minDiag
  }.min()

if (gramDiagMin < 1e-6 && config.regularizationLambda == 0.0) {
  logger.warn("Gram matrix may be ill-conditioned (min diagonal element = {}). " +
    "Consider setting regularizationLambda > 0 in RecursiveInverseConfig for numerical stability.",
    gramDiagMin)
}
```

---

## 13. Dead Code — `previousMetric` — COMPLETED

**Priority:** 🟢 Low  
**Effort:** Minimal  
**Files:** `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`

### Problem

In `iterativeInverseInternal()`, `previousMetric` is assigned but only used in the non-convergence warning:

```scala
var previousMetric = Double.MaxValue  // line ~466

while (iter < config.maxIter && !converged) {
  // ...
  val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
  previousMetric = metric  // assigned here, line ~479

  if (metric < config.tolerance) {
    converged = true
  }
  // ... metric is never compared to previousMetric for logic
}

if (!converged) {
  logger.warn("{} did not converge after {} iterations. Last metric: {}",
    algorithmName, config.maxIter, previousMetric)  // only use
}
```

The name `previousMetric` suggests it holds the previous iteration's metric, but it actually holds the **current** metric (assigned from `metric`). This is misleading. Additionally, `previousMetric` is never compared to the current `metric` for divergence detection.

### Fix Option A — Rename and add divergence detection (recommended)

```scala
var lastMetric = Double.MaxValue

while (iter < config.maxIter && !converged) {
  iter += 1
  // ...
  val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
  logger.debug("{} iter={}: ||I - A*X||_F / n = {}", algorithmName, iter, metric)

  // Divergence detection: if metric is increasing significantly
  if (metric > lastMetric * 100 && iter > 1) {
    logger.warn("{}: possible divergence detected at iter={} (metric={}, prev={}). " +
      "Consider using a smaller alpha or preconditioning.",
      algorithmName, iter, metric, lastMetric)
  }
  lastMetric = metric

  if (metric < config.tolerance) {
    converged = true
  }
  // ...
}

if (!converged) {
  logger.warn("{} did not converge after {} iterations. Last metric: {}",
    algorithmName, config.maxIter, lastMetric)
}
```

### Fix Option B — Remove entirely

If divergence detection isn't desired:

```scala
// Remove var previousMetric = Double.MaxValue
// Remove previousMetric = metric
// Change the warning to use a local variable captured at loop exit:

var finalMetric = Double.MaxValue
while (iter < config.maxIter && !converged) {
  // ...
  val metric = math.sqrt(new BlockMatrixOps(residual).frobeniusNormSquared()) / n
  finalMetric = metric
  // ...
}

if (!converged) {
  logger.warn("{} did not converge after {} iterations. Last metric: {}",
    algorithmName, config.maxIter, finalMetric)
}
```

---

## 14. Example job: EASE Recommender

**Priority:** 🟡 Nice-to-have  
**Effort:** Medium  
**Files:** New — `examples/` or `bench/`  
**Tag:** **[both]** — correctness of EASE weights: local. End-to-end training time: cluster

### Problem

The library has no example usage that demonstrates real-world value. An EASE (Embarrassingly Shallow Autoencoder) model is the ideal showcase because:

1. **It's essentially a distributed matrix inversion** — the closed-form EASE weights solve `B = (XᵀX + λI)⁻¹` for the Gram matrix inverse, exactly what `sparkInverse` does
2. **It's a real recommendation algorithm** used in production at companies like Spotify and Zalando
3. **It operates on sparse user–item interaction matrices** that are exactly the size (10⁵–10⁷ users × items) where distributed inversion matters

### EASE Model

The EASE algorithm (Steck, 2019) computes item-to-item recommendation weights:

1. Compute the Gram matrix: `G = XᵀX` where `X` is the user–item interaction matrix
2. Regularize and invert: `B = (G + λI)⁻¹`
3. Zero-out diagonal: `B ← B - diag(B) · I`
4. Normalize: `B ← B / -diag(B)ᵀ` (element-wise division by negative diagonal)
5. Predictions: `Ŷ = XB` for scoring unseen items

### Proposed Example

A self-contained Spark job in `bench/src/main/scala/sparkinverse/benchmark/EaseJob.scala`:

```scala
// Read interaction matrix X (users × items) as CoordinateMatrix
val X = spark.read.parquet("hdfs://...").as[Interaction].toCoordinateMatrix()

// Step 1: Gram matrix
val G = X.transpose().multiply(X, midSplits = 4)

// Step 2: Regularize and invert using sparkInverse
val lambda = 500.0
val regularized = G.add(MatrixInternals.eyeBlockMatrix(..., lambda, ...))
val B = regularized.inverse(IterativeInverseConfig(
  order = 2, maxIter = 50, tolerance = 1e-8,
  alphaStrategy = AlphaStrategy.Frobenius
))

// Step 3-4: Zero diagonal and normalize (local operations on blocks)
val B_normalized = EaseModel.normalizeWeights(B)

// Step 5: Score for all user–item pairs
val predictions = X.multiply(B_normalized, midSplits = 4)
```

### What this validates

| Aspect                 | How the example tests it                                                            |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Correctness of inverse | Compare EASE recall@K against a known-good NumPy/SciPy baseline on a small dataset  |
| AlphaStrategy choice   | Show convergence difference on the (G + λI) matrix which has κ ≈ λ/σ_min            |
| Performance at scale   | Run on MovieLens 10M+ or a synthetic 1M×100K matrix on a K8s cluster                |
| Partition behavior     | The Gram matrix of a 1M×100K sparse matrix is 100K×100K dense — perfect stress test |
| Checkpoint + unpersist | Long-running job (many iterations) exposes the correctness bug in TODO #1           |

### Dataset suggestions

- **MovieLens 25M** — well-known, ~160K users × ~60K items
- **Synthetic** — generate a sparse user–item matrix with configurable κ to test alpha strategies on ill-conditioned Gram matrices

---

## 15. CANS / Chebyshev-Optimal Polynomial Style

**Priority:** 🟡 Research / future work  
**Effort:** High  
**Files:** `core/src/main/scala/sparkinverse/api/Configs.scala`, `core/src/main/scala/sparkinverse/block/BlockMatrixOps.scala`, tests  
**Tag:** **[local]** — coefficient correctness and iteration count are locally measurable; cluster benchmarking still needed for wall-clock impact

### Problem

A preliminary `PolynomialStyle.CANS` API was removed because it was only an alias for the standard hyperpower correction while documentation implied Chebyshev/CANS-optimal coefficients were implemented. Reintroducing this must wait until the math is implemented and validated.

### Required work before reintroducing

1. Derive the correct minimax polynomial for the **matrix inverse correction** form used here:

   ```text
   X_{k+1} = X_k · p(R_k),   R_k = I - A X_k
   ```

   Do not directly transplant CANS formulas for polar decomposition unless the transformation to inverse iteration is proven.

2. Define a public API only after the coefficients are real, e.g.:

   ```scala
   sealed trait PolynomialStyle
   object PolynomialStyle {
     case object Binomial extends PolynomialStyle
     case object ChebyshevMinimax extends PolynomialStyle
   }
   ```

3. Implement coefficient construction for at least `order = 3`, with explicit fallback or rejection for unsupported orders.

4. Add focused tests:
   - coefficient values for known intervals
   - equality with Binomial only when mathematically expected
   - iteration count improvement on a small matrix where minimax coefficients should help
   - no misleading no-op behavior

5. Update README only after implementation is complete. Documentation must say exactly which orders and assumptions are supported.

### Non-goal for current branch

Do **not** expose a placeholder `CANS` or `PolynomialStyle` API that behaves exactly like Binomial. That is misleading for users and hard to validate.

---

## Benchmarking Mandate

Every improvement implemented from this TODO list **must** be followed by:

### 1. Benchmark comparison

- **Before:** run the existing benchmark on the pre-change commit
- **After:** run the same benchmark on the post-change commit
- **Measure:** both wall-clock time **and** iteration count (the latter is logged at `logger.info` level in the iterative inverse loop)
- **Report:** the delta (e.g. "−12%", "+3 iterations", "no measurable difference")
- **No commit without numbers.** If the change does not improve anything, document that and keep it anyway if there's a correctness reason — but the numbers must exist.

### 2. Assess whether results are locally meaningful

| Matrix size       | What you **can** measure locally (`local[*]`)                   | What you **need** a cluster for                                                         |
| ----------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **n ≤ 500**       | Correctness, basic algorithm behavior, RMSE                     | Nothing — too small for network I/O to matter                                           |
| **n ≈ 1000–5000** | Whether iteration count changes, whether the fix works          | Shuffle cost, I/O patterns, partition behavior                                          |
| **n ≥ 10 000**    | Nothing meaningful — local Spark is a loopback, no real shuffle | Actual performance differences, shuffle vs no-shuffle, partition explosion, network I/O |

### 3. Tag each improvement accordingly

Add a one-line tag to the TODO item:

- **`[local]`** — the improvement can be verified on `local[*]` (correctness fixes, algorithm changes that affect iteration count)
- **`[cluster]`** — the improvement is only visible on a real distributed cluster (shuffle optimizations, partition tuning, I/O reductions)
- **`[both]`** — correctness can be verified locally, performance needs a cluster

### Current tag assignments

| #   | Item                               | Tag           | Notes                                        |
| --- | ---------------------------------- | ------------- | -------------------------------------------- |
| 1   | Unpersist-before-evaluation bug    | **[both]**    | Correctness: local. Perf regression: cluster |
| 2   | Alpha scaling strategies           | **[both]**    | Convergence: local. Shuffle cost: cluster    |
| 3   | Single-pass quadrant split         | **[both]**    | Correctness: local. I/O reduction: cluster   |
| 4   | LU-based local inversion           | **[local]**   | Pure computation speedup                     |
| 5   | CoordinateMatrix repeated squaring | **[cluster]** | Shuffles only matter at scale                |
| 6   | Adaptive iteration order           | **[local]**   | Iteration count measurable locally           |
| 7   | Checkpoint tuning                  | **[both]**    | Correctness: local. I/O: cluster             |
| 8   | Pseudo-inverse SVD path            | **[local]**   | Algorithm change visible locally             |
| 9   | Partition count explosion          | **[cluster]** | Only visible with many workers               |
| 10  | Adaptive checkpointing             | **[cluster]** | I/O patterns need real storage               |
| 11  | Tolerance-based early exit         | **[local]**   | Iteration count measurable locally           |
| 12  | Pseudo-inverse regularization      | **[local]**   | Correctness + convergence visible locally    |
| 13  | Dead code `previousMetric`         | **[local]**   | Code cleanup only                            |
| 14  | EASE recommender example job       | **[both]**    | Correctness: local. End-to-end time: cluster |
| 15  | CANS / Chebyshev polynomial style  | **[local]**   | Coefficients + iteration count locally       |
