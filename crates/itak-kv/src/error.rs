//! Named errors for the itak-kv crate. Every fallible operation
//! returns one of these; nothing in this crate panics on a corrupt
//! file, a mismatched geometry, or a forged entry supplied by a
//! caller or read back from disk.

use std::fmt;

use attention::AttentionError;

/// Everything that can go wrong in the snapshot codec or the
/// prefix store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// The attention crate refused an operation while capturing or
    /// restoring a snapshot.
    Attention(AttentionError),
    /// A token prefix of length zero was offered to the store; an
    /// empty prefix matches everything and stores nothing, so it is
    /// refused rather than indexed.
    EmptyPrefix,
    /// A snapshot's token count disagrees with the token list it was
    /// offered under.
    TokenCountMismatch {
        /// Token count inside the snapshot.
        snapshot: usize,
        /// Token count in the offered token list.
        tokens: usize,
    },
    /// The token ids stored in an entry do not hash to the entry's
    /// key: the entry is forged or corrupt and is never applied.
    PrefixHashMismatch {
        /// The key the entry is filed under.
        stored: u64,
        /// The hash of the token ids the entry actually carries.
        computed: u64,
    },
    /// An entry's preamble token count disagrees with the token
    /// count inside its snapshot payload.
    SnapshotTokenMismatch {
        /// Token count in the entry preamble.
        entry: usize,
        /// Token count inside the snapshot.
        snapshot: usize,
    },
    /// A byte stream does not start with the expected magic.
    BadMagic {
        /// Which format was being read ("snapshot" or "entry").
        what: &'static str,
    },
    /// A byte stream carries a format version this build does not
    /// read.
    UnsupportedVersion {
        /// Which format was being read ("snapshot" or "entry").
        what: &'static str,
        /// The version found in the stream.
        version: u32,
    },
    /// A byte stream ends before its own header says it should.
    Truncated {
        /// Which format was being read.
        what: &'static str,
        /// Bytes the header requires.
        expected: usize,
        /// Bytes actually present.
        actual: usize,
    },
    /// A snapshot's trailing checksum does not match its contents.
    ChecksumMismatch {
        /// The checksum stored in the stream.
        stored: u64,
        /// The checksum computed over the stream.
        computed: u64,
    },
    /// A snapshot's per-layer rows do not add up against its own
    /// header (token count, windows, row widths).
    InconsistentSnapshot {
        /// What does not add up.
        detail: String,
    },
    /// A snapshot's geometry differs from the geometry the caller
    /// requires (layer pattern, KV heads, or head dimension).
    GeometryMismatch {
        /// What differs.
        detail: String,
    },
    /// A snapshot's model fingerprint differs from the fingerprint
    /// the caller requires.
    FingerprintMismatch {
        /// The fingerprint the caller requires.
        expected: String,
        /// The fingerprint the snapshot carries.
        actual: String,
    },
    /// A live cache cannot be snapshotted because its layers do not
    /// agree on one position (a cache is snapshotted as one
    /// sequence, so every layer must have appended the same number
    /// of positions).
    IncoherentCache {
        /// What disagrees.
        detail: String,
    },
    /// One entry is larger than the store's whole byte cap; it is
    /// refused at `put` time instead of evicting everything else
    /// and still not fitting.
    EntryTooLarge {
        /// The entry's size in bytes.
        bytes: u64,
        /// The store's byte cap.
        cap: u64,
    },
    /// A configuration value (env override or caller argument) is
    /// unusable as given.
    InvalidConfig {
        /// What is wrong with it.
        detail: String,
    },
    /// A filesystem operation failed.
    Io {
        /// What the code was doing when it failed.
        context: String,
        /// The operating system's message.
        message: String,
    },
}

impl KvError {
    /// Wrap an `io::Error` with the operation that produced it.
    pub(crate) fn io(context: impl Into<String>, error: std::io::Error) -> Self {
        KvError::Io {
            context: context.into(),
            message: error.to_string(),
        }
    }
}

impl From<AttentionError> for KvError {
    fn from(error: AttentionError) -> Self {
        KvError::Attention(error)
    }
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvError::Attention(e) => write!(f, "attention: {e}"),
            KvError::EmptyPrefix => write!(f, "empty token prefix refused"),
            KvError::TokenCountMismatch { snapshot, tokens } => write!(
                f,
                "snapshot covers {snapshot} tokens but was offered under {tokens} token ids"
            ),
            KvError::PrefixHashMismatch { stored, computed } => write!(
                f,
                "stored token ids hash to {computed:#018x}, not the entry key {stored:#018x}; entry refused"
            ),
            KvError::SnapshotTokenMismatch { entry, snapshot } => write!(
                f,
                "entry preamble covers {entry} tokens but its snapshot covers {snapshot}"
            ),
            KvError::BadMagic { what } => write!(f, "{what}: bad magic"),
            KvError::UnsupportedVersion { what, version } => {
                write!(f, "{what}: unsupported format version {version}")
            }
            KvError::Truncated {
                what,
                expected,
                actual,
            } => write!(f, "{what}: truncated, needs {expected} bytes, has {actual}"),
            KvError::ChecksumMismatch { stored, computed } => write!(
                f,
                "snapshot checksum mismatch: stored {stored:#018x}, computed {computed:#018x}"
            ),
            KvError::InconsistentSnapshot { detail } => {
                write!(f, "inconsistent snapshot: {detail}")
            }
            KvError::GeometryMismatch { detail } => write!(f, "geometry mismatch: {detail}"),
            KvError::FingerprintMismatch { expected, actual } => write!(
                f,
                "model fingerprint mismatch: need {expected}, snapshot carries {actual}"
            ),
            KvError::IncoherentCache { detail } => write!(f, "incoherent cache: {detail}"),
            KvError::EntryTooLarge { bytes, cap } => {
                write!(f, "entry of {bytes} bytes exceeds the store cap of {cap} bytes")
            }
            KvError::InvalidConfig { detail } => write!(f, "invalid config: {detail}"),
            KvError::Io { context, message } => write!(f, "{context}: {message}"),
        }
    }
}

impl std::error::Error for KvError {}
