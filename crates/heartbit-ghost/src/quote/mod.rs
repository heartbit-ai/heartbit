//! Quote-tweet pipeline — polls curated source accounts, drafts
//! opinionated-but-charitable quote-tweets via the `quote_writer`
//! agent, routes through Telegram review, posts via `twitter_quote`.
//!
//! Task 4 introduces the source polling + dedup store. The pipeline
//! runtime (`run_quote_pipeline`, `QuoteConfig`, `QuoteReviewDelivery`,
//! outcomes) is added in Task 5.

pub mod sources;

pub use sources::{
    InMemoryQuoteSeenStore, JsonlQuoteSeenStore, QuoteCandidate, QuoteSeenStore, QuoteSource,
    QuoteStoreError, XUserTimelineSource,
};
