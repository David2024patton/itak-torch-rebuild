//! itak-loader: the bridge between the `gguf` crate and the
//! architecture modules (ARCHITECTURE.md module grid, TODO T-2.1).
//!
//! The arch modules (here `muse-glimmer`) never parse files. They read
//! metadata through their [`Metadata`](muse_glimmer::Metadata) trait
//! and weights through their
//! [`TensorSource`](muse_glimmer::TensorSource) trait. This crate is
//! the implementor of those traits over a parsed
//! [`gguf::GgufFile`], plus the inventory check that verifies a file
//! against the arch module's published expectation before any weight
//! bytes move:
//!
//! - [`load_config`] parses a [`GlimmerConfig`] from a real file's
//!   metadata, end to end, with the arch module's fail-loud key naming
//!   intact.
//! - [`coverage`] compares the file's tensor names against
//!   [`muse_glimmer::expected_tensor_names`]; [`CoverageReport::
//!   require_complete`](CoverageReport::require_complete) fails by
//!   name on any miss, all misses at once.
//! - [`load_weights`] runs the coverage gate first, then loads the
//!   full weight set through the [`TensorSource`] adapter. A partially
//!   loaded model is never returned.
//!
//! Diagnostics are built in from birth (David's rule): structured
//! `key=value` debug lines flow through [`diag`] whenever
//! `ITAK_LOADER_DEBUG` is set or [`diag::set_debug_enabled`] is
//! called.
//!
//! Quick start:
//!
//! ```no_run
//! use std::path::Path;
//! let file = gguf::GgufFile::open(Path::new("model.gguf")).unwrap();
//! let cfg = itak_loader::load_config(&file).unwrap();
//! let report = itak_loader::coverage(&file, &cfg);
//! report.require_complete().unwrap();
//! let weights = itak_loader::load_weights(&file, &cfg).unwrap();
//! println!("{} layers loaded", weights.layers.len());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adapter;
pub mod coverage;
pub mod diag;
pub mod error;

pub use adapter::{GgufMetadata, GgufTensorSource};
pub use coverage::{coverage, CoverageReport};
pub use error::{LoaderError, Result};

use gguf::GgufFile;
use muse_glimmer::{GlimmerConfig, Weights};

/// Parse the Muse Glimmer config from a parsed GGUF file's metadata.
///
/// # Errors
/// The arch module's own errors, unwrapped: a missing key is
/// [`GlimmerError::MissingMetadata`](muse_glimmer::GlimmerError)
/// naming the key, a wrong `general.architecture` is a refusal naming
/// what the file declares.
pub fn load_config(file: &GgufFile) -> muse_glimmer::Result<GlimmerConfig> {
    let cfg = GlimmerConfig::from_metadata(&GgufMetadata::new(file))?;
    diag::debug(
        "config_loaded",
        &format!(
            "arch={} layers={} hidden={} vocab={} pattern_source={:?}",
            muse_glimmer::ARCHITECTURE,
            cfg.block_count,
            cfg.hidden,
            cfg.vocab_size,
            cfg.pattern_source
        ),
    );
    Ok(cfg)
}

/// Load the full Glimmer weight set from a parsed GGUF file, gated on
/// the inventory check: coverage runs first and fails by name on any
/// miss, so the error lists every absent tensor instead of whichever
/// one the load order happened to reach first.
///
/// # Errors
/// [`LoaderError::MissingTensors`] when the inventory is incomplete;
/// the arch module's errors (as [`LoaderError::Glimmer`]) when a
/// present tensor cannot be delivered as stored.
pub fn load_weights(file: &GgufFile, cfg: &GlimmerConfig) -> Result<Weights> {
    coverage(file, cfg).require_complete()?;
    let weights = Weights::load(&GgufTensorSource::new(file), cfg)?;
    diag::debug(
        "weights_loaded",
        &format!("layers={}", weights.layers.len()),
    );
    Ok(weights)
}
