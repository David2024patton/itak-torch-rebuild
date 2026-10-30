//! Operational limits for the kernels, held as configuration.
//!
//! House rule (David, standing): every operational limit is config
//! with a default, never a compiled-in constant buried in a kernel.
//! The block geometry of a quant format is NOT an operational limit
//! (it is the file format itself, and it comes from the `gguf`
//! crate's type definitions); what is config is how much work a
//! single call may be asked to do before it refuses by name instead
//! of trusting a hostile or corrupt shape.

/// Configuration for the kernel entry points in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    /// Maximum element count a single dequantize or matvec call may
    /// cover (input elements for dequantize, `rows * cols` for
    /// matvec). A call whose shapes imply more elements refuses with
    /// [`crate::KernelError::LimitExceeded`] before any arithmetic.
    pub max_elements: usize,
}

impl KernelConfig {
    /// The default element ceiling: 2^31 elements. At 4 bytes per
    /// dequantized element that bounds one call's output at 8 GiB,
    /// far above any single tensor in the models this engine serves,
    /// while still catching absurd shapes from corrupt inputs.
    pub const DEFAULT_MAX_ELEMENTS: usize = 1 << 31;

    /// A config with the given element ceiling.
    pub fn new(max_elements: usize) -> Self {
        Self { max_elements }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_ELEMENTS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_documented_ceiling() {
        assert_eq!(
            KernelConfig::default().max_elements,
            KernelConfig::DEFAULT_MAX_ELEMENTS
        );
        assert_eq!(KernelConfig::DEFAULT_MAX_ELEMENTS, 1 << 31);
    }
}
