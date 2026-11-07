//! Store-level integration tests: LRU eviction under the byte
//! cap, the forged-entry trust guard, geometry and fingerprint
//! refusals at lookup, torn files, and persistence across reopen.

use std::fs;
use std::path::PathBuf;

use attention::{KvCache, LayerKind};
use itak_kv::{KvGeometry, KvSnapshot, PrefixStore, StoreConfig};

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("itak-kv-it-{}-{}", std::process::id(), tag));
    let _ = fs::remove_dir_all(&root);
    root
}

fn geometry() -> KvGeometry {
    KvGeometry {
        layers: vec![LayerKind::Windowed { window: 4 }, LayerKind::Global],
        num_kv_heads: 2,
        head_dim: 3,
    }
}

fn snapshot_for(tokens: &[u32], fingerprint: &str) -> KvSnapshot {
    let geometry = geometry();
    let mut cache =
        KvCache::new(&geometry.layers, geometry.num_kv_heads, geometry.head_dim).expect("builds");
    let row = geometry.row_floats();
    for (position, &token) in tokens.iter().enumerate() {
        for layer in 0..geometry.layers.len() {
            let k: Vec<f32> = (0..row)
                .map(|i| (token as f32) + (position * 10 + layer + i) as f32 / 11.0)
                .collect();
            let v: Vec<f32> = (0..row)
                .map(|i| -((token as f32) + (position + layer * 3 + i) as f32 / 5.0))
                .collect();
            cache.append(layer, &k, &v, position).expect("append");
        }
    }
    KvSnapshot::capture(&cache, fingerprint).expect("capture")
}

fn open_store(root: &std::path::Path, cap: u64) -> PrefixStore {
    PrefixStore::open(&StoreConfig {
        root: root.to_path_buf(),
        max_bytes: cap,
    })
    .expect("store opens")
}

#[test]
fn eviction_respects_the_cap_and_lru_order() {
    let root = temp_root("evict");
    let a = vec![1_u32, 1, 1, 1];
    let b = vec![2_u32, 2, 2, 2];
    let c = vec![3_u32, 3, 3, 3];
    // Measure one entry's size with an uncapped store, then set
    // the cap to hold two entries plus slack, never three.
    let mut probe = open_store(&root, u64::MAX);
    probe.put(&a, &snapshot_for(&a, "m"), 1).expect("put a");
    let entry_bytes = probe.stats().total_bytes;
    drop(probe);
    let _ = fs::remove_dir_all(&root);

    let cap = entry_bytes * 2 + entry_bytes / 2;
    let mut store = open_store(&root, cap);
    store.put(&a, &snapshot_for(&a, "m"), 1).expect("put a");
    store.put(&b, &snapshot_for(&b, "m"), 2).expect("put b");
    // Touch A so B becomes the least recently used.
    assert!(store
        .lookup(&[1, 1, 1, 1, 9], None, None, 3)
        .expect("lookup")
        .is_some());
    // Putting C pushes the store over the cap; B must go.
    store.put(&c, &snapshot_for(&c, "m"), 4).expect("put c");
    let stats = store.stats();
    assert!(stats.total_bytes <= cap, "over cap: {stats:?}");
    assert_eq!(stats.entries, 2);
    assert!(
        store.lookup(&b, None, None, 5).expect("lookup").is_none(),
        "the LRU entry (b) must be evicted"
    );
    assert!(store.lookup(&a, None, None, 6).expect("lookup").is_some());
    assert!(store.lookup(&c, None, None, 7).expect("lookup").is_some());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn forged_entry_is_refused_and_shorter_honest_prefix_still_serves() {
    let root = temp_root("forged");
    let mut store = open_store(&root, 1 << 30);
    let short = vec![5_u32, 6];
    let long = vec![5_u32, 6, 7, 8];
    store
        .put(&short, &snapshot_for(&short, "m"), 1)
        .expect("put short");
    store
        .put(&long, &snapshot_for(&long, "m"), 2)
        .expect("put long");
    // Forge the long entry: rewrite its stored token ids in place
    // (same length) so they no longer hash to the entry's key and
    // no longer match the query either.
    let long_meta = store
        .entries()
        .into_iter()
        .find(|e| e.token_count == 4)
        .expect("long entry indexed");
    let path = root.join(&long_meta.file_name);
    let mut bytes = fs::read(&path).expect("read entry");
    // Token region starts after the fixed 72-byte preamble.
    bytes[72] = 0xef;
    bytes[73] = 0xbe;
    bytes[74] = 0xad;
    bytes[75] = 0xde;
    fs::write(&path, &bytes).expect("forge entry");
    // Reopen so the index is rebuilt from disk, then query through
    // the forged entry's prefix: the long entry must be refused
    // and the honest short entry must serve.
    let mut store = open_store(&root, 1 << 30);
    let hit = store
        .lookup(&[5, 6, 7, 8, 9], None, None, 3)
        .expect("lookup")
        .expect("the honest short prefix serves");
    assert_eq!(hit.token_count, 2);
    assert_eq!(hit.tokens, short);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn geometry_and_fingerprint_requirements_refuse_at_lookup() {
    let root = temp_root("geom");
    let mut store = open_store(&root, 1 << 30);
    let tokens = vec![3_u32, 1, 4];
    store
        .put(&tokens, &snapshot_for(&tokens, "model-a"), 1)
        .expect("put");
    let mut foreign = geometry();
    foreign.head_dim = 12;
    assert!(store
        .lookup(&tokens, Some(&foreign), None, 2)
        .expect("lookup")
        .is_none());
    assert!(store
        .lookup(&tokens, None, Some("model-b"), 3)
        .expect("lookup")
        .is_none());
    let hit = store
        .lookup(&tokens, Some(&geometry()), Some("model-a"), 4)
        .expect("lookup")
        .expect("matching requirements hit");
    assert_eq!(hit.token_count, 3);
    // And the hit's snapshot restores under the checked path.
    hit.snapshot
        .restore_checked(&geometry(), "model-a")
        .expect("restore checked");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn torn_temp_and_torn_final_files_are_ignored() {
    let root = temp_root("torn");
    fs::create_dir_all(&root).expect("mkdir");
    // A torn temp file from a crashed writer: never indexed.
    fs::write(root.join(".tmp-deadbeef-123.kv"), b"half an entry").expect("write temp");
    // A torn final file: indexed never, refused by name at scan.
    fs::write(root.join("0123456789abcdef-4-0.kv"), b"ITKV").expect("write torn");
    let mut store = open_store(&root, 1 << 30);
    assert_eq!(store.stats().entries, 0);
    // The store still works normally afterwards.
    let tokens = vec![8_u32, 8];
    store
        .put(&tokens, &snapshot_for(&tokens, "m"), 1)
        .expect("put");
    assert!(store
        .lookup(&[8, 8, 8], None, None, 2)
        .expect("lookup")
        .is_some());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn entries_survive_reopen() {
    let root = temp_root("persist");
    let tokens = vec![42_u32, 43, 44, 45, 46, 47];
    {
        let mut store = open_store(&root, 1 << 30);
        store
            .put(&tokens, &snapshot_for(&tokens, "m"), 10)
            .expect("put");
    }
    let mut store = open_store(&root, 1 << 30);
    let hit = store
        .lookup(&[42, 43, 44, 45, 46, 47, 48], None, None, 20)
        .expect("lookup")
        .expect("hit after reopen");
    assert_eq!(hit.snapshot.token_count, 6);
    let restored = hit.snapshot.restore().expect("restore");
    assert_eq!(restored.next_position(0).expect("position"), 6);
    let _ = fs::remove_dir_all(&root);
}
