use napi::{Env, JsUnknown};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Helper for visitor API of Graph traversal
pub struct DFSActions {
  /// If JS flips this on a parent we will skip its children
  skipped: Rc<AtomicBool>,
  /// If JS flips this on a parent we will return immediately
  stopped: Rc<AtomicBool>,
}

impl DFSActions {
  pub fn new() -> Self {
    Self {
      skipped: Rc::new(AtomicBool::new(false)),
      stopped: Rc::new(AtomicBool::new(false)),
    }
  }

  /// Skip children of the current node
  pub fn skip_children(&self) {
    self.skipped.store(true, Ordering::Relaxed);
  }

  /// Stop the traversal
  pub fn stop(&self) {
    self.stopped.store(true, Ordering::Relaxed);
  }

  /// Reset state on each node visit
  pub fn reset(&self) {
    self.skipped.store(false, Ordering::Relaxed);
  }

  /// Return true if visitors have called the skip hook.
  ///
  /// The DFS should skip a subtree on this case. This will be reset by set_skipped.
  pub fn is_skipped(&self) -> bool {
    self.skipped.load(Ordering::Relaxed)
  }

  /// Return true if visitors have called the stop hook.
  ///
  /// The DFS should early terminate on this case.
  pub fn is_stopped(&self) -> bool {
    self.stopped.load(Ordering::Relaxed)
  }

  /// Build a shared object for the DFSActions struct. The JavaScript object will reference the
  /// atomic values of the struct.
  /// Avoid running this on a loop and prefer sharing the result across visitor invocations.
  ///
  /// Note this struct is not thread safe as it is. If multiple threads try to use DFSActions
  /// we might have use-after-free issues. It should be impossible for that to happen on neither
  /// Rust or Node.js sides.
  pub fn to_js(&self, env: &Env) -> napi::Result<JsUnknown> {
    let mut js_actions = env.create_object()?;
    js_actions.set(
      "skipChildren",
      env.create_function_from_closure("skipChildren", {
        let skipped = self.skipped.clone();
        move |_| {
          skipped.store(true, Ordering::Relaxed);
          Ok(())
        }
      })?,
    )?;
    js_actions.set(
      "stop",
      env.create_function_from_closure("stop", {
        let stopped = self.stopped.clone();
        move |_| {
          stopped.store(true, Ordering::Relaxed);
          Ok(())
        }
      })?,
    )?;
    Ok(js_actions.into_unknown())
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_dfs_actions() {
    let actions = DFSActions::new();
    assert_eq!(actions.is_skipped(), false);
    assert_eq!(actions.is_stopped(), false);
  }
}
