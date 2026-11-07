//! Prefix keying: FNV-1a 64 over the token ids, the same hash the
//! diag capture fingerprints use elsewhere in this workspace.
//!
//! The hash is an INDEX, never a proof. Two different token
//! sequences may share a key (a collision), so every entry also
//! stores its full token-id list, and a lookup only ever hits after
//! comparing the stored ids against the query's prefix token for
//! token. A collision therefore costs a wasted candidate read; it
//! can never produce a wrong hit. A stored id list whose hash does
//! not reproduce the entry's key is treated as forged or corrupt
//! and refused by name ([`crate::KvError::PrefixHashMismatch`]).

/// FNV-1a 64 offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// An alternate offset basis for the discriminator hash, so an
/// entry filename depends on two independent hashes of the same
/// token list.
const ALT_OFFSET: u64 = 0x8422_2325_cbf2_9ce4;

/// FNV-1a 64 over raw bytes, from an explicit offset basis.
fn fnv1a(bytes: &[u8], basis: u64) -> u64 {
    let mut hash = basis;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Feed one token id into a running FNV-1a state (little-endian
/// bytes, matching the on-disk token encoding).
fn fnv1a_token(mut hash: u64, token: u32) -> u64 {
    for byte in token.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The prefix key for a token-id sequence.
#[must_use]
pub fn prefix_hash(tokens: &[u32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |h, &t| fnv1a_token(h, t))
}

/// A second, independent hash of the same token list, used as the
/// filename discriminator so two colliding prefixes do not fight
/// over one filename.
#[must_use]
pub fn prefix_discriminator(tokens: &[u32]) -> u64 {
    tokens.iter().fold(ALT_OFFSET, |h, &t| fnv1a_token(h, t))
}

/// FNV-1a 64 over a byte slice (the snapshot payload checksum).
#[must_use]
pub fn checksum(bytes: &[u8]) -> u64 {
    fnv1a(bytes, FNV_OFFSET)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_order_and_value_sensitive() {
        assert_ne!(prefix_hash(&[1, 2, 3]), prefix_hash(&[3, 2, 1]));
        assert_ne!(prefix_hash(&[1, 2, 3]), prefix_hash(&[1, 2, 4]));
        assert_ne!(prefix_hash(&[1, 2]), prefix_hash(&[1, 2, 0]));
        assert_eq!(prefix_hash(&[]), FNV_OFFSET);
    }

    #[test]
    fn discriminator_is_a_distinct_hash() {
        let tokens = [7_u32, 8, 9, 10];
        assert_ne!(prefix_hash(&tokens), prefix_discriminator(&tokens));
        // And it is deterministic: the same list names the same file.
        assert_eq!(prefix_discriminator(&tokens), prefix_discriminator(&tokens));
    }

    #[test]
    fn checksum_matches_fnv1a_of_empty_and_nonempty() {
        assert_eq!(checksum(&[]), FNV_OFFSET);
        assert_ne!(checksum(b"itkv"), checksum(b"itkw"));
    }
}
