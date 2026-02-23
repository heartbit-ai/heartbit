use std::sync::Mutex;

/// Tracks cumulative importance of stored memories to trigger reflection.
///
/// When the sum of importance values since the last trigger exceeds `threshold`,
/// `record()` returns `true` and resets the accumulator. This follows the
/// Generative Agents (Park et al., 2023) reflection pattern.
pub struct ReflectionTracker {
    threshold: u32,
    accumulated: Mutex<u32>,
}

impl ReflectionTracker {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            accumulated: Mutex::new(0),
        }
    }

    /// Record an importance value. Returns `true` if the threshold was exceeded,
    /// triggering a reflection. The accumulator is reset after triggering.
    pub fn record(&self, importance: u8) -> bool {
        let mut acc = self.accumulated.lock().expect("reflection lock poisoned");
        *acc += importance as u32;
        if *acc >= self.threshold {
            *acc = 0;
            true
        } else {
            false
        }
    }

    /// Current accumulated importance (for testing/debugging).
    pub fn accumulated(&self) -> u32 {
        *self.accumulated.lock().expect("reflection lock poisoned")
    }
}

/// Hint text appended to store tool output when a reflection is triggered.
pub const REFLECTION_HINT: &str = "\n\n[Reflection suggested] You have stored several important memories recently. \
     Take a moment to reflect on what you've learned. Use memory_store with high importance \
     to record any insights, patterns, or connections you notice across your recent memories.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_triggers_at_threshold() {
        let tracker = ReflectionTracker::new(10);

        // Accumulate to just below threshold
        assert!(!tracker.record(5)); // 5
        assert!(!tracker.record(4)); // 9
        // This should trigger
        assert!(tracker.record(1)); // 10 >= 10
    }

    #[test]
    fn tracker_resets_after_trigger() {
        let tracker = ReflectionTracker::new(10);

        // Trigger
        assert!(tracker.record(10));
        assert_eq!(tracker.accumulated(), 0);

        // Should need full threshold again
        assert!(!tracker.record(5));
        assert_eq!(tracker.accumulated(), 5);
    }

    #[test]
    fn tracker_exceeds_threshold() {
        let tracker = ReflectionTracker::new(10);

        // Single large value exceeding threshold
        assert!(tracker.record(15));
        assert_eq!(tracker.accumulated(), 0);
    }

    #[test]
    fn tracker_many_small_values() {
        let tracker = ReflectionTracker::new(10);

        for _ in 0..9 {
            assert!(!tracker.record(1));
        }
        // 9 accumulated, one more should trigger
        assert!(tracker.record(1));
    }

    #[test]
    fn tracker_triggers_multiple_times() {
        let tracker = ReflectionTracker::new(5);

        assert!(tracker.record(5)); // first trigger
        assert!(tracker.record(5)); // second trigger
        assert!(!tracker.record(3)); // not yet
        assert!(tracker.record(2)); // third trigger
    }
}
