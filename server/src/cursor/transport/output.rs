//! Buffers, replays, broadcasts, and atomically closes downstream output.

use bytes::Bytes;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;

#[derive(Default)]
pub struct OutputHub {
    state: parking_lot::Mutex<OutputState>,
    /// Index of the next frame a BidiPoll client has not received yet.
    poll_cursor: AtomicUsize,
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
    pub fn poll_frames(&self) -> (usize, Vec<Bytes>, bool) {
        let state = self.state.lock();
        let start = self.poll_cursor.load(Ordering::SeqCst);
        let available = state.history.len();
        let frames = if start < available {
            state.history[start..].to_vec()
        } else {
            Vec::new()
        };
        drop(state);
        (start, frames, self.is_closed())
    }

    pub fn advance_poll_cursor(&self, by: usize) {
        self.poll_cursor.fetch_add(by, Ordering::SeqCst);
    }

    pub fn is_closed(&self) -> bool {
        self.state.lock().closed
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
}
