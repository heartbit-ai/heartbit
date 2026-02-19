use chrono::{DateTime, Utc};

/// Weights for composite memory scoring (Park et al., 2023).
///
/// `alpha * recency + beta * importance + gamma * relevance`
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for recency component (default: 0.3).
    pub alpha: f64,
    /// Weight for importance component (default: 0.3).
    pub beta: f64,
    /// Weight for relevance component (default: 0.4).
    pub gamma: f64,
    /// Exponential decay rate for recency (default: 0.01, ~69h half-life).
    pub decay_rate: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.3,
            gamma: 0.4,
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

/// Composite score: `alpha * recency + beta * importance + gamma * relevance`.
pub fn composite_score(
    weights: &ScoringWeights,
    created_at: DateTime<Utc>,
    now: DateTime<Utc>,
    importance: u8,
    relevance: f64,
) -> f64 {
    let r = recency_score(created_at, now, weights.decay_rate);
    let i = importance_score(importance);
    let rel = relevance.clamp(0.0, 1.0);
    weights.alpha * r + weights.beta * i + weights.gamma * rel
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
    fn composite_score_all_max() {
        let now = Utc::now();
        let weights = ScoringWeights::default();
        let score = composite_score(&weights, now, now, 10, 1.0);
        // recency=1.0, importance=1.0, relevance=1.0
        // 0.3*1 + 0.3*1 + 0.4*1 = 1.0
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn composite_score_all_zero() {
        let now = Utc::now();
        let old = now - Duration::hours(100_000);
        let weights = ScoringWeights::default();
        let score = composite_score(&weights, old, now, 1, 0.0);
        // recency≈0, importance=0.0, relevance=0.0
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
            decay_rate: 0.01,
        };
        let score = composite_score(&weights, old, now, 10, 0.0);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn default_weights_sum_to_one() {
        let w = ScoringWeights::default();
        assert!((w.alpha + w.beta + w.gamma - 1.0).abs() < f64::EPSILON);
    }
}
