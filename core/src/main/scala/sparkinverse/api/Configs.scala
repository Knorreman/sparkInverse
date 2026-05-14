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
// The optimal choice is α* = 1/σ₁² where σ₁ is the largest singular value
// of A. All strategies below produce α ≤ 1/σ₁² (guaranteeing convergence)
// but differ in tightness and computational cost.
//
sealed trait AlphaStrategy

object AlphaStrategy {

  /** α = 1 / (‖A‖₁ · ‖A‖_∞) — the original strategy.
    *
    * Guaranteed upper bound: σ₁² ≤ ‖A‖₁·‖A‖_∞, so α ≤ 1/σ₁².
    * Requires two Spark shuffles (normOne + normInf).
    * Often significantly smaller than optimal, leading to slower convergence.
    */
  case object NormProduct extends AlphaStrategy

  /** α = 1 / ‖A‖²_F — Frobenius norm squared.
    *
    * Since ‖A‖²_F = Σ σᵢ² ≥ σ₁², we have α ≤ 1/σ₁².
    * Usually tighter than NormProduct for matrices with moderate singular values.
    * Requires only a single Spark action (map + sum, no shuffle).
    */
  case object Frobenius extends AlphaStrategy

  /** α = 1 / σ₁² estimated via power iteration on AᵀA.
    *
    * The most accurate strategy. After `powerIterations` steps of
    * v_{k+1} = (AᵀA)·v_k / ‖(AᵀA)·v_k‖, the Rayleigh quotient
    * gives σ₁² with exponential convergence rate.
    * Costs 2 distributed multiplies per power iteration.
    */
  case class PowerIteration(powerIterations: Int = 3) extends AlphaStrategy

  /** α = 1 / σ₁² estimated from the first Newton-Schulz iteration.
    *
    * Uses Gelfand's spectral radius estimate on the residual of the
    * first iteration to refine α for subsequent iterations.
    * Costs zero extra distributed multiplications — the estimate
    * comes from residual data already computed during iteration 1.
    *
    * After the first iteration, α is refined:
    *   σ₁² ≈ n / ‖A·X₀‖²_F   (since X₀ = α₀·Aᵀ and A·X₀ = α₀·A·Aᵀ)
    *   α_refined = 1 / σ₁²
    *
    * If the initial α₀ is too far off (first iteration diverges), falls
    * back to the Frobenius estimate.
    */
  case object Adaptive extends AlphaStrategy
}

// ── Iteration polynomial style ───────────────────────────────────────────────
//
// Controls how the correction polynomial is built at each iteration step.
//
sealed trait PolynomialStyle

object PolynomialStyle {

  /** Standard binomial (hyperpower) expansion:
    * C = I + R + R² + R³ + ... + R^{order-1}
    *
    * This is the classical Newton-Schulz generalization.
    * Coefficients are all 1.
    */
  case object Binomial extends PolynomialStyle

  /** CANS-optimal coefficients for order 3 (Chebyshev-accelerated Newton-Schulz).
    *
    * From Grishina, Smirnov & Rakhuba (2026), Proposition 3.3:
    * The optimal degree-3 odd polynomial approximating f≡1 on [a,b] is:
    *
    *   p_{2,a,b}(x) = α·(x - x³)
    *
    * where α = 3 / (2·(a²+ab+b²)^(3/2) + a²b + ab²)
    *
    * This gives the correction C = α₁·I + α₃·R + α₅·R²
    * with CANS-optimal coefficients that minimize the worst-case
    * spectral error, yielding provably faster convergence than
    * the binomial expansion for order 3.
    *
    * For order ≠ 3, falls back to Binomial.
    */
  case object CANS extends PolynomialStyle
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
  persistLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER,

  /** Strategy for computing the initial scaling α in X₀ = α·Aᵀ.
    * Default is Frobenius (1/‖A‖²_F) which is cheap and tighter
    * than the legacy NormProduct (1/(‖A‖₁·‖A‖_∞)).
    */
  alphaStrategy: AlphaStrategy = AlphaStrategy.Frobenius,

  /** Polynomial style for the hyperpower correction.
    * Default is Binomial (the classical expansion with all-1 coefficients).
    * Set to CANS for Chebyshev-optimal coefficients on order 3.
    */
  polynomialStyle: PolynomialStyle = PolynomialStyle.Binomial
)
