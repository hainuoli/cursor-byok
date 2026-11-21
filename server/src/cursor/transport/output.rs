//! Buffers, replays, broadcasts, and atomically closes downstream output.

use bytes::Bytes;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::mpsc;

/// Exclusive lease for the single active `RunPoll` stream of a transport.
///
/// The poll delivery cursor is shared state: two concurrent poll streams would
/// hand the same `seqno` to two clients and advance the cursor twice, so the
/// second poll is rejected until the first one ends.
pub struct PollLease {
    hub: Arc<OutputHub>,
}

impl Drop for PollLease {
    fn drop(&mut self) {
        self.hub.poll_active.store(false, Ordering::SeqCst);
    }
}

#[derive(Default)]
pub struct OutputHub {
    state: parking_lot::Mutex<OutputState>,
    /// Index of the next frame a BidiPoll client has not received yet.
    poll_cursor: AtomicUsize,
    poll_active: AtomicBool,
    notify: tokio::sync::Notify,
}

#[derive(Default)]
struct OutputState {
    history: Vec<Bytes>,
    subscribers: Vec<mpsc::UnboundedSender<Bytes>>,
    closed: bool,
}

impl OutputHub {
    pub fn emit(&self, frame: Bytes) -> bool {
        let mut state = self.state.lock();
        if state.closed {
            return false;
        }
        state.history.push(frame.clone());
        state
            .subscribers
            .retain(|subscriber| subscriber.send(frame.clone()).is_ok());
        drop(state);
        self.notify.notify_one();
        true
    }

    pub fn subscribe(&self) -> mpsc::UnboundedReceiver<Bytes> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let mut state = self.state.lock();
        for frame in &state.history {
            let _ = sender.send(frame.clone());
        }
        if !state.closed {
            state.subscribers.push(sender);
        }
        receiver
    }

    /// Returns buffered frames the polling client has not seen yet together with
    /// the index the first returned frame occupies and whether output is closed.
    ///
    /// The cursor, the history snapshot and `closed` are read under one lock so
    /// a run finishing between two reads cannot hide its final frames.
    pub fn poll_frames(&self) -> (usize, Vec<Bytes>, bool) {
        let state = self.state.lock();
        let start = self.poll_cursor.load(Ordering::SeqCst);
        let available = state.history.len();
        let frames = if start < available {
            state.history[start..].to_vec()
        } else {
            Vec::new()
        };
        let closed = state.closed;
        drop(state);
        (start, frames, closed)
    }

    pub fn advance_poll_cursor(&self, by: usize) {
        self.poll_cursor.fetch_add(by, Ordering::SeqCst);
    }

    /// Claims the exclusive poll consumer slot for this transport.
    pub fn try_begin_poll(self: &Arc<Self>) -> Option<PollLease> {
        if self.poll_active.swap(true, Ordering::SeqCst) {
            return None;
        }
        Some(PollLease {
            hub: Arc::clone(self),
        })
    }

    /// Resolves when a new frame is emitted or the hub closes.
    ///
    /// Emitters call `notify_one`, which stores a permit when no waiter is
    /// parked, so a frame emitted between the caller's `poll_frames` check and
    /// this wait is not lost (plain `notify_waiters` stores no permit and that
    /// wakeup would be missed, stalling a poll until its idle timeout).
    pub async fn wait_for_output(&self) {
        self.notify.notified().await;
    }

    pub fn close(&self) -> bool {
        let mut state = self.state.lock();
        if state.closed {
            return false;
        }
        state.closed = true;
        state.subscribers.clear();
        drop(state);
        self.notify.notify_one();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn wait_for_output_returns_for_frames_emitted_before_the_wait() {
        let hub = OutputHub::default();
        assert!(hub.poll_frames().1.is_empty());
        hub.emit(Bytes::from_static(b"frame"));
        tokio::time::timeout(Duration::from_millis(200), hub.wait_for_output())
            .await
            .expect("a frame emitted before the wait must not be missed");
    }

    #[tokio::test]
    async fn wait_for_output_wakes_for_frames_emitted_while_waiting() {
        let hub = std::sync::Arc::new(OutputHub::default());
        let emitter = hub.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            emitter.emit(Bytes::from_static(b"frame"));
        });
        tokio::time::timeout(Duration::from_secs(2), hub.wait_for_output())
            .await
            .expect("an emit while waiting must wake the waiter");
    }

    #[tokio::test]
    async fn wait_for_output_returns_when_closed() {
        let hub = OutputHub::default();
        hub.close();
        tokio::time::timeout(Duration::from_millis(200), hub.wait_for_output())
            .await
            .expect("a closed hub must not block the waiter");
    }

    #[test]
    fn poll_lease_is_exclusive_until_the_stream_ends() {
        let hub = Arc::new(OutputHub::default());
        let lease = hub.try_begin_poll().expect("first poll must be admitted");
        assert!(
            hub.try_begin_poll().is_none(),
            "second poll must be rejected"
        );
        drop(lease);
        assert!(hub.try_begin_poll().is_some(), "lease must be released");
    }

    #[test]
    fn poll_frames_reports_closed_under_the_same_lock() {
        let hub = OutputHub::default();
        hub.emit(Bytes::from_static(b"frame"));
        hub.close();
        let (start, frames, closed) = hub.poll_frames();
        assert_eq!(start, 0);
        assert_eq!(frames.len(), 1);
        assert!(
            closed,
            "closed must be observed together with the final frames"
        );
    }

    #[test]
    fn poll_cursor_continues_after_delivered_frames() {
        let hub = OutputHub::default();
        hub.emit(Bytes::from_static(b"a"));
        hub.emit(Bytes::from_static(b"b"));
        let (start, frames, _) = hub.poll_frames();
        assert_eq!(start, 0);
        assert_eq!(frames.len(), 2);

        hub.advance_poll_cursor(2);
        hub.emit(Bytes::from_static(b"c"));

        let (start, frames, _) = hub.poll_frames();
        assert_eq!(start, 2, "a new poll continues where the cursor stopped");
        assert_eq!(frames, vec![Bytes::from_static(b"c")]);
    }
}
