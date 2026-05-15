package sparkinverse.api

import org.apache.spark.storage.StorageLevel

sealed trait PseudoInverseSide

object PseudoInverseSide {
  case object Left extends PseudoInverseSide
  case object Right extends PseudoInverseSide
}

// ── Alpha scaling strategies for iterative inversion ────────────────────────
//
// The initial approximation is X₀ = α·Aᵀ. The value of α determines the
// spectral radius ρ(I − α·AᵀA) and thus the convergence rate.
//
// The optimal safe choice is α* = 1/σ₁² where σ₁ is the largest singular value
// of A. The strategies below produce conservative estimates of that scale with
// different distributed costs.
//
sealed trait AlphaStrategy

object AlphaStrategy {

  /** α = 1 / (‖A‖₁ · ‖A‖_∞) — the original strategy.
    *
    * Guaranteed upper bound: σ₁² ≤ ‖A‖₁·‖A‖_∞, so α ≤ 1/σ₁².
    * Requires two Spark shuffles/actions (normOne + normInf).
    */
  case object NormProduct extends AlphaStrategy

  /** α = 1 / ‖A‖²_F — Frobenius norm squared.
    *
    * Since ‖A‖²_F = Σ σᵢ² ≥ σ₁², α ≤ 1/σ₁².
    * This is safe, simple, and avoids the row/column norm shuffles used by
    * NormProduct. It is often a good distributed default, but it is not
    * universally tighter than NormProduct.
    */
  case object Frobenius extends AlphaStrategy

  /** α = 1 / σ₁² estimated via power iteration on AᵀA.
    *
    * After `powerIterations` steps of v_{k+1} = (AᵀA)·v_k / ‖(AᵀA)·v_k‖,
    * the Rayleigh quotient vᵀ(AᵀA)v / vᵀv estimates σ₁².
    * Costs 2 distributed multiplies per power iteration.
    */
  case class PowerIteration(powerIterations: Int = 3) extends AlphaStrategy

  /** Experimental conservative refinement from the first residual.
    *
    * Starts from Frobenius and, after the first iteration, may shrink α based
    * on the observed residual. This can improve stability but should not be
    * interpreted as an optimal or accelerating strategy.
    */
  case object Adaptive extends AlphaStrategy
}

final case class RecursiveInverseConfig(
  limit: Int = 4096,
  midSplits: Int = 1,
  useCheckpoints: Boolean = true,
  targetOutputPartitions: Option[Int] = None,
  unionCoalesceThreshold: Int = 8,
  minBlockSizeForPersistence: Int = 1000000,
  /** Tikhonov regularization parameter for pseudo-inverse.
    * When > 0, adds λI to the Gram matrix before inversion,
    * improving numerical stability for ill-conditioned matrices.
    * Only affects `pseudoInverse`, not `inverse`.
    * Default 0.0 preserves standard Moore-Penrose behavior.
    */
  regularizationLambda: Double = 0.0
)

final case class IterativeInverseConfig(
  order: Int = 2,
  maxIter: Int = 30,
  tolerance: Double = 1e-15,
  useCheckpoints: Boolean = true,
  checkpointEvery: Int = 5,
  midSplits: Int = 1,
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,

  /** Strategy for computing the initial scaling α in X₀ = α·Aᵀ.
    * Default is Frobenius (1/‖A‖²_F), a safe zero-shuffle distributed default.
    */
  alphaStrategy: AlphaStrategy = AlphaStrategy.Frobenius,

  /** Check convergence every N iterations. Default 1 preserves the historical
    * behavior of checking ‖I - A·X‖_F / n on every iteration.
    */
  convergenceCheckInterval: Int = 1
)
