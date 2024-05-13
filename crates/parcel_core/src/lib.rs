//! Core re-implementation in Rust

use napi::bindgen_prelude::Function;
use napi::threadsafe_function::ThreadsafeFunction;

pub mod hash;
/// New-type for paths relative to a project-root
pub mod project_path;
/// Request types and run functions
pub mod requests;

struct JavaScriptController {
    callback: ThreadsafeFunction<...>
}

#[napi]
fn register_worker(callback: Function<...>) {/**/}

