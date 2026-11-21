//! Wires the Cursor-facing API routes.

pub mod bidi;
pub mod bidi_poll;
mod handlers;
pub mod proxy;
mod run_sse;

pub use handlers::router;
