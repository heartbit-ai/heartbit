use chrono::{DateTime, Utc};

/// Weights for composite memory scoring (Park et al., 2023).
///
/// `alpha * recency + beta * importance + gamma * relevance + delta * strength`
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for recency component (default: 0.25).
    pub alpha: f64,
    /// Weight for importance component (default: 0.25).
    pub beta: f64,
    /// Weight for relevance component (default: 0.3).
    pub gamma: f64,
    /// Weight for strength component (default: 0.2).
    pub delta: f64,
    /// Exponential decay rate for recency (default: 0.01, ~69h half-life).
    pub decay_rate: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            beta: 0.25,
            gamma: 0.3,
            delta: 0.2,
            decay_rate: 0.01,
        }
    }
}

/// Recency score: `e^(-decay_rate * hours)`, returns `[0.0, 1.0]`.
///
/// Clamps negative durations (future timestamps) to 1.0.
pub fn recency_score(created_at: DateTime<Utc>, now: DateTime<Utc>, decay_rate: f64) -> f64 {
    let duration = now.signed_duration_since(created_at);
    let hours = duration.num_seconds() as f64 / 3600.0;
    if hours <= 0.0 {
        return 1.0;
    }
    (-decay_rate * hours).exp()
}

/// Normalize importance `[1, 10]` to `[0.0, 1.0]`.
///
/// Values outside `[1, 10]` are clamped.
pub fn importance_score(importance: u8) -> f64 {
    let clamped = importance.clamp(1, 10);
    (clamped as f64 - 1.0) / 9.0
}

/// Strength score: identity mapping (already in `[0.0, 1.0]`).
///
/// Clamps to `[0.0, 1.0]` for safety.
pub fn strength_score(strength: f64) -> f64 {
    strength.clamp(0.0, 1.0)
}

/// Ebbinghaus strength decay rate (per hour). Default: 0.005.
///
/// Half-life ≈ ln(2)/0.005 ≈ 139 hours ≈ 5.8 days.
/// This means unused memories lose half their strength every ~6 days.
pub const STRENGTH_DECAY_RATE: f64 = 0.005;

/// Compute effective strength with Ebbinghaus decay from `last_accessed`.
///
/// `effective = stored_strength * e^(-decay_rate * hours_since_last_access)`
///
/// This gives recently-accessed memories their full stored strength, while
/// memories not accessed in a long time decay toward zero. Reinforcement
/// on access (+0.2, capped at 1.0) resets the decay clock.
pub fn effective_strength(
    strength: f64,
    last_accessed: DateTime<Utc>,
    now: DateTime<Utc>,
    decay_rate: f64,
) -> f64 {
    let hours = now
        .signed_duration_since(last_accessed)
        .num_seconds()
        .max(0) as f64
        / 3600.0;
    let decayed = strength * (-decay_rate * hours).exp();
    decayed.clamp(0.0, 1.0)
}

/// Composite score: `alpha * recency + beta * importance + gamma * relevance + delta * strength`.
pub fn composite_score(
    weights: &ScoringWeights,
    created_at: DateTime<Utc>,
    now: DateTime<Utc>,
    importance: u8,
    relevance: f64,
    strength: f64,
) -> f64 {
    let r = recency_score(created_at, now, weights.decay_rate);
    let i = importance_score(importance);
    let rel = relevance.clamp(0.0, 1.0);
    let s = strength_score(strength);
    weights.alpha * r + weights.beta * i + weights.gamma * rel + weights.delta * s
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn recency_score_now_is_one() {
        let now = Utc::now();
        let score = recency_score(now, now, 0.01);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recency_score_decays_over_time() {
        let now = Utc::now();
        let one_day_ago = now - Duration::hours(24);
        let score = recency_score(one_day_ago, now, 0.01);
        // e^(-0.01 * 24) ≈ 0.787
        assert!(score < 1.0);
        assert!(score > 0.5);
    }

    #[test]
    fn recency_score_very_old_approaches_zero() {
        let now = Utc::now();
        let long_ago = now - Duration::hours(10000);
        let score = recency_score(long_ago, now, 0.01);
        assert!(score < 0.001);
    }

    #[test]
    fn recency_score_negative_duration_clamps() {
        let now = Utc::now();
        let future = now + Duration::hours(5);
        let score = recency_score(future, now, 0.01);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn importance_score_range() {
        assert!((importance_score(1) - 0.0).abs() < f64::EPSILON);
        assert!((importance_score(10) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn importance_score_midpoint() {
        // importance 5 → (5-1)/9 ≈ 0.444
        let score = importance_score(5);
        assert!((score - 4.0 / 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn importance_score_clamps_out_of_range() {
        // 0 clamps to 1 → 0.0
        assert!((importance_score(0) - 0.0).abs() < f64::EPSILON);
        // 15 clamps to 10 → 1.0
        assert!((importance_score(15) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn strength_score_identity() {
        assert!((strength_score(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((strength_score(1.0) - 1.0).abs() < f64::EPSILON);
        assert!((strength_score(0.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn strength_score_clamps() {
        assert!((strength_score(-0.5) - 0.0).abs() < f64::EPSILON);
        assert!((strength_score(1.5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn composite_score_all_max() {
        let now = Utc::now();
        let weights = ScoringWeights::default();
        let score = composite_score(&weights, now, now, 10, 1.0, 1.0);
        // recency=1.0, importance=1.0, relevance=1.0, strength=1.0
        // 0.25 + 0.25 + 0.3 + 0.2 = 1.0
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn composite_score_all_zero() {
        let now = Utc::now();
        let old = now - Duration::hours(100_000);
        let weights = ScoringWeights::default();
        let score = composite_score(&weights, old, now, 1, 0.0, 0.0);
        // recency≈0, importance=0.0, relevance=0.0, strength=0.0
        assert!(score < 0.01);
    }

    #[test]
    fn composite_score_importance_dominates() {
        let now = Utc::now();
        let old = now - Duration::hours(100_000);
        let weights = ScoringWeights {
            alpha: 0.0,
            beta: 1.0,
            gamma: 0.0,
            delta: 0.0,
            decay_rate: 0.01,
        };
        let score = composite_score(&weights, old, now, 10, 0.0, 0.0);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn composite_score_strength_contributes() {
        let now = Utc::now();
        let weights = ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            delta: 1.0,
            decay_rate: 0.01,
        };
        let score = composite_score(&weights, now, now, 1, 0.0, 0.8);
        assert!((score - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn low_strength_entries_rank_lower() {
        let now = Utc::now();
        let weights = ScoringWeights::default();
        let strong = composite_score(&weights, now, now, 5, 0.5, 1.0);
        let weak = composite_score(&weights, now, now, 5, 0.5, 0.1);
        assert!(
            strong > weak,
            "higher strength should yield higher composite score"
        );
    }

    #[test]
    fn default_weights_sum_to_one() {
        let w = ScoringWeights::default();
        assert!((w.alpha + w.beta + w.gamma + w.delta - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn effective_strength_no_decay_when_just_accessed() {
        let now = Utc::now();
        let eff = effective_strength(0.8, now, now, STRENGTH_DECAY_RATE);
        assert!((eff - 0.8).abs() < 1e-10);
    }

    #[test]
    fn effective_strength_decays_over_time() {
        let now = Utc::now();
        let week_ago = now - Duration::hours(7 * 24);
        let eff = effective_strength(1.0, week_ago, now, STRENGTH_DECAY_RATE);
        // e^(-0.005 * 168) ≈ e^(-0.84) ≈ 0.432
        assert!(eff < 0.5, "should decay significantly after a week: {eff}");
        assert!(eff > 0.3, "should not fully decay after a week: {eff}");
    }

    #[test]
    fn effective_strength_approaches_zero_for_very_old() {
        let now = Utc::now();
        let month_ago = now - Duration::hours(30 * 24);
        let eff = effective_strength(1.0, month_ago, now, STRENGTH_DECAY_RATE);
        // e^(-0.005 * 720) ≈ e^(-3.6) ≈ 0.027
        assert!(eff < 0.05, "should be near zero after a month: {eff}");
    }

    #[test]
    fn effective_strength_clamps_negative_duration() {
        let now = Utc::now();
        let future = now + Duration::hours(5);
        let eff = effective_strength(0.8, future, now, STRENGTH_DECAY_RATE);
        assert!((eff - 0.8).abs() < 1e-10);
    }
}
