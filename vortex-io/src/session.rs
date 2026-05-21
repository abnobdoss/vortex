// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;

use vortex_error::VortexExpect;
use vortex_session::SessionExt;
use vortex_session::SessionVar;

use crate::runtime::Handle;

/// Session state for Vortex async runtimes.
pub struct RuntimeSession {
    handle: Option<Handle>,
}

impl SessionVar for RuntimeSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Default for RuntimeSession {
    fn default() -> Self {
        Self {
            handle: Handle::find(),
        }
    }
}

impl Debug for RuntimeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSession").finish_non_exhaustive()
    }
}

/// Extension trait for accessing runtime session data.
pub trait RuntimeSessionExt: SessionExt {
    /// Returns a handle for this session's runtime.
    fn handle(&self) -> Handle {
        self.get::<RuntimeSession>().handle
                .as_ref()
                .vortex_expect("Runtime handle not configured in Vortex session. Please setup a `CurrentThreadRuntime`, or configure the session for `with_tokio`.")
                .clone()
    }

    /// Configure the runtime session to use the application's Tokio runtime.
    ///
    /// For example, if the application is launched using `#[tokio::main]`.
    #[cfg(feature = "tokio")]
    fn with_tokio(self) -> Self {
        use crate::runtime::tokio::TokioRuntime;
        self.get_mut::<RuntimeSession>().handle = Some(TokioRuntime::current());
        self
    }

    /// Configure the runtime session to use a specific Vortex runtime handle.
    fn with_handle(self, handle: Handle) -> Self {
        self.get_mut::<RuntimeSession>().handle = Some(handle);
        self
    }
}
impl<S: SessionExt> RuntimeSessionExt for S {}

#[cfg(test)]
mod tests {
    use vortex_session::SessionExt;
    use vortex_session::VortexSession;

    use super::RuntimeSession;
    use super::RuntimeSessionExt;
    use crate::runtime::single::block_on;

    /// Regression test for ABA-19.
    ///
    /// `with_handle` writes the supplied `Handle` into the shared `Arc<SessionVars>` DashMap via
    /// `get_mut`. Because `VortexSession::clone` only bumps the `Arc` reference count (see
    /// `vortex-session/src/lib.rs:28`), every "clone" shares the same underlying map. Calling
    /// `with_handle` on any clone therefore mutates the original session and every other clone.
    ///
    /// Expected-correct behaviour: configuring a clone must not affect the original. The original's
    /// `RuntimeSession.handle` must remain `None` after the clone is configured.
    ///
    /// Today the assertion fails — the original is silently poisoned with the handle written by
    /// the clone — so this test is marked `#[ignore]` until the fix lands.
    #[test]
    #[ignore = "demonstrates ABA-19; see https://linear.app/abanoubdoss/issue/ABA-19"]
    fn issue_aba19_session_with_handle_does_not_mutate_clone() {
        // Create the original session *outside* any runtime so that `Handle::find()` returns
        // `None` and `RuntimeSession` is not yet materialised in the DashMap. This gives us a
        // clean baseline: the original has no handle.
        let original = VortexSession::empty();

        // Inside a `block_on` scope, take an Arc-bump clone and configure it with the runtime's
        // handle. If `with_handle` branched a copy-on-write session this would be a no-op for
        // `original`; today it stamps `h` straight into the shared DashMap.
        //
        // Capture a separate clone to hand into the async closure; the post-block_on assertion
        // uses `original` directly.
        let original_for_clone = original.clone();
        block_on(|h| async move {
            let _configured_clone = original_for_clone.clone().with_handle(h);
            // Drop the configured clone before the runtime ends (mirrors the H17 pattern).
            drop(_configured_clone);
        });
        // The runtime (and thus every Handle it minted) is now dropped.

        // Assert the original session's RuntimeSession slot is still absent / handle-free.
        // On a correct implementation `with_handle` on a clone would not touch `original` at all,
        // so `get_mut_opt` returns `None` (the key was never inserted into `original`'s own map).
        // Under the bug, the key exists and `handle` is `Some(dead_handle)`.
        let handle_state = original
            .get_mut_opt::<RuntimeSession>()
            .map(|rs| rs.handle.is_some());
        assert!(
            handle_state != Some(true),
            "ABA-19 reproduced: calling `with_handle` on a clone of `original` mutated \
             `original`'s shared `Arc<SessionVars>`. The original session now carries the \
             handle that was written by the clone. \
             Fix: `with_handle` must deep-clone (or fork) the DashMap before mutating so that \
             configuring a clone never leaks into the original session."
        );
    }
}
