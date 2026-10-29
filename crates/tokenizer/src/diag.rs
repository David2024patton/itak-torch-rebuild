//! Diagnostic seam (David's debugging law): every refusal and load event
//! in this crate can emit a structured line when the `debug-log` feature
//! is enabled. With the feature off the macro expands to nothing and the
//! format arguments are never evaluated.

macro_rules! dbglog {
    ($($arg:tt)*) => {
        #[cfg(feature = "debug-log")]
        {
            eprintln!("[tokenizer] {}", format!($($arg)*));
        }
    };
}

pub(crate) use dbglog;
