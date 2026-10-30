//! The architecture registry and dispatch: the only load path
//! (TODO T-2.8, ARCHITECTURE.md Section 5.3, PLAN S5).
//!
//! Routing is by the file's own `general.architecture` string,
//! looked up in a registry of versioned entries. There is no other
//! way in: [`ArchRegistry::dispatch`] reads the string from the
//! parsed file, finds the entry, and runs the handler the entry
//! names. An unrecognized string is a named refusal carrying the
//! complete supported list
//! ([`LoaderError::UnknownArchitecture`](crate::LoaderError)), so
//! the error itself states what this engine loads.
//!
//! Two module kinds live in the registry. A full module
//! ([`ModuleKind::Full`]) parses its real config here (muse-glimmer
//! through the crate's existing loader bridge, llama and qwen2
//! through `arch-llama`). A registered stub
//! ([`ModuleKind::RegisteredStub`]) proves routing and reads the
//! family's own keys through [`crate::stub`], and is replaced by
//! the family's real module under the same string when it lands.
//!
//! Engine selection for the serve path is a separate, capability-
//! based step ([`select_engine`]): a candidate serves a load only
//! when it is available and lists forward support for the load's
//! architecture, and the load's module must provide the FORWARD
//! capability. File size is not an input to that decision anywhere:
//! a 4 MB file and a 40 GB file with the same architecture string
//! select the same engine, and a tiny file whose architecture has
//! no forward is refused exactly like a huge one.
//!
//! Diagnostics are built in (David's rule): dispatch decisions and
//! refusals emit structured lines through [`crate::diag`].

use std::fmt;
use std::ops::BitOr;

use arch_llama::LlamaConfig;
use gguf::GgufFile;
use muse_glimmer::GlimmerConfig;

use crate::diag;
use crate::error::{LoaderError, Result};
use crate::stub::{self, StubConfig};

/// Version of the registry format itself. Entries carry their own
/// module versions beside this; bump when the entry shape changes.
pub const REGISTRY_VERSION: u32 = 1;

/// What a registered module can do with a file, stated as a set so
/// serve-path decisions are capability checks, not guesses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Capabilities {
    bits: u8,
}

impl Capabilities {
    /// The module parses the family's config from file metadata.
    pub const CONFIG: Self = Self { bits: 1 };
    /// The module publishes an expected tensor inventory, so a
    /// coverage check can run against the file.
    pub const COVERAGE: Self = Self { bits: 2 };
    /// The module runs a forward pass (it can generate).
    pub const FORWARD: Self = Self { bits: 4 };
    /// No capabilities.
    pub const EMPTY: Self = Self { bits: 0 };

    /// The union of two capability sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Whether this set contains every capability in `other`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }

    /// The capability names in this set, in a stable order, for
    /// load results and error text.
    #[must_use]
    pub fn names(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.contains(Self::CONFIG) {
            out.push("config");
        }
        if self.contains(Self::COVERAGE) {
            out.push("coverage");
        }
        if self.contains(Self::FORWARD) {
            out.push("forward");
        }
        out
    }
}

impl BitOr for Capabilities {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = self.names();
        if names.is_empty() {
            write!(f, "none")
        } else {
            write!(f, "{}", names.join("+"))
        }
    }
}

/// How complete the module behind a registry entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleKind {
    /// A real architecture module: config, tensor map, forward.
    Full,
    /// A registered routing stub: proves dispatch and reads the
    /// family's keys, but has no forward yet.
    RegisteredStub,
}

/// Which loader a registry entry dispatches to. The registry is
/// data: dispatch matches on the entry's handler, never on the
/// architecture string itself, so a family can move handlers (stub
/// to full) by changing one entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handler {
    /// The muse-glimmer module, through this crate's config bridge.
    Glimmer,
    /// The arch-llama module (llama and qwen2 families).
    LlamaFamily,
    /// A stub in [`crate::stub`], keyed by the entry's string.
    RegisteredStub,
}

/// One versioned registry entry: an architecture string, the module
/// behind it, and what that module can do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchEntry {
    architecture: &'static str,
    module_version: u32,
    kind: ModuleKind,
    capabilities: Capabilities,
    handler: Handler,
}

impl ArchEntry {
    /// An entry for a full architecture module.
    #[must_use]
    pub fn full(architecture: &'static str, module_version: u32, handler: Handler) -> Self {
        Self {
            architecture,
            module_version,
            kind: ModuleKind::Full,
            capabilities: Capabilities::CONFIG | Capabilities::COVERAGE | Capabilities::FORWARD,
            handler,
        }
    }

    /// An entry for a registered stub module: config and coverage
    /// shape today, no forward.
    #[must_use]
    pub fn stub(architecture: &'static str, module_version: u32) -> Self {
        Self {
            architecture,
            module_version,
            kind: ModuleKind::RegisteredStub,
            capabilities: Capabilities::CONFIG | Capabilities::COVERAGE,
            handler: Handler::RegisteredStub,
        }
    }

    /// The architecture string this entry routes, exactly as a file
    /// declares it in `general.architecture`.
    #[must_use]
    pub fn architecture(&self) -> &'static str {
        self.architecture
    }

    /// The module's version. Bumped when the module behind the
    /// string changes shape (a stub replaced by the full module
    /// bumps it, so load results can state what served a file).
    #[must_use]
    pub fn module_version(&self) -> u32 {
        self.module_version
    }

    /// Whether the module is full or a registered stub.
    #[must_use]
    pub fn kind(&self) -> ModuleKind {
        self.kind
    }

    /// What the module can do with a file.
    #[must_use]
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    /// The loader this entry dispatches to.
    #[must_use]
    pub fn handler(&self) -> Handler {
        self.handler
    }
}

/// What dispatch produced: the loaded configuration, in the shape
/// of the module that served the file.
#[derive(Debug)]
pub enum Dispatched {
    /// muse-glimmer, parsed by the full Glimmer module.
    Glimmer(Box<GlimmerConfig>),
    /// llama or qwen2, parsed by the full arch-llama module.
    LlamaFamily(Box<LlamaConfig>),
    /// A family whose registered stub proved the routing.
    RegisteredStub(Box<StubConfig>),
}

impl Dispatched {
    /// The architecture string of the module that produced this.
    #[must_use]
    pub fn architecture(&self) -> &str {
        match self {
            Dispatched::Glimmer(_) => muse_glimmer::ARCHITECTURE,
            Dispatched::LlamaFamily(cfg) => &cfg.architecture,
            Dispatched::RegisteredStub(cfg) => &cfg.architecture,
        }
    }
}

/// The result of a dispatch: the registry entry that routed the
/// file (what served it, at what module version, with what
/// capabilities) beside the loaded configuration. ARCHITECTURE.md
/// 5.3 asks for arch support and backend availability to be stated
/// in the load result; the entry half of that statement lives here,
/// the backend half in [`EngineSelection`].
#[derive(Debug)]
pub struct DispatchResult {
    /// The registry entry that routed the file.
    pub entry: ArchEntry,
    /// The loaded configuration.
    pub load: Dispatched,
}

/// The architecture registry. Construct the standard one with
/// [`ArchRegistry::standard`], or an empty one with
/// [`ArchRegistry::new`] and register entries; dispatch only ever
/// consults the registry it is called on.
#[derive(Clone, Debug, Default)]
pub struct ArchRegistry {
    entries: Vec<ArchEntry>,
}

impl ArchRegistry {
    /// An empty registry. Nothing dispatches through it until
    /// entries are registered.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// The registry this engine ships: the full modules that exist
    /// (muse-glimmer, llama, qwen2) and registered stubs for the
    /// families next in the conformance order (gemma2, olmo2,
    /// deepseek2). Every entry is module version 1.
    #[must_use]
    pub fn standard() -> Self {
        let mut registry = Self::new();
        let entries = [
            ArchEntry::full("muse-glimmer", 1, Handler::Glimmer),
            ArchEntry::full("llama", 1, Handler::LlamaFamily),
            ArchEntry::full("qwen2", 1, Handler::LlamaFamily),
            ArchEntry::stub("gemma2", 1),
            ArchEntry::stub("olmo2", 1),
            ArchEntry::stub("deepseek2", 1),
        ];
        for entry in entries {
            // The shipped table cannot contain a duplicate; a panic
            // here is a build-time bug in this function, not input.
            registry
                .register(entry)
                .expect("standard registry entries are unique");
        }
        registry
    }

    /// Register one entry. An architecture string routes to exactly
    /// one module: registering a string that already has an entry
    /// is refused, by name.
    ///
    /// # Errors
    /// [`LoaderError::DuplicateArchitecture`] when the string is
    /// already registered.
    pub fn register(&mut self, entry: ArchEntry) -> Result<()> {
        if self.lookup(entry.architecture()).is_some() {
            diag::debug(
                "register_refused",
                &format!("arch={} reason=duplicate", entry.architecture()),
            );
            return Err(LoaderError::DuplicateArchitecture {
                architecture: entry.architecture().to_string(),
            });
        }
        diag::debug(
            "registered",
            &format!(
                "arch={} module_version={} kind={:?} capabilities={}",
                entry.architecture(),
                entry.module_version(),
                entry.kind(),
                entry.capabilities()
            ),
        );
        self.entries.push(entry);
        Ok(())
    }

    /// The entry registered for an architecture string, if any.
    #[must_use]
    pub fn lookup(&self, architecture: &str) -> Option<&ArchEntry> {
        self.entries
            .iter()
            .find(|e| e.architecture() == architecture)
    }

    /// Every architecture string this registry routes, sorted: the
    /// list an unknown-architecture refusal carries.
    #[must_use]
    pub fn supported(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .iter()
            .map(|e| e.architecture().to_string())
            .collect();
        names.sort();
        names
    }

    /// Route a parsed GGUF file to the module registered for the
    /// file's own `general.architecture` string and load its
    /// configuration. This is the only load path: callers do not
    /// pick a module, the file does, through this registry.
    ///
    /// # Errors
    /// - [`LoaderError::Gguf`] when the file carries no
    ///   `general.architecture` key (the gguf error names the key).
    /// - [`LoaderError::UnknownArchitecture`] when the string names
    ///   no registered module; the error carries the supported list.
    /// - The serving module's own errors, unwrapped in kind
    ///   ([`LoaderError::Glimmer`], [`LoaderError::Llama`],
    ///   [`LoaderError::Gguf`]), when its config read fails.
    pub fn dispatch(&self, file: &GgufFile) -> Result<DispatchResult> {
        let arch = file.architecture()?;
        let entry = match self.lookup(arch) {
            Some(entry) => entry.clone(),
            None => {
                let supported = self.supported();
                diag::debug(
                    "dispatch_refused",
                    &format!(
                        "arch={arch} reason=unknown supported={}",
                        supported.join(",")
                    ),
                );
                return Err(LoaderError::UnknownArchitecture {
                    found: arch.to_string(),
                    supported,
                });
            }
        };
        let load = match entry.handler() {
            Handler::Glimmer => Dispatched::Glimmer(Box::new(crate::load_config(file)?)),
            Handler::LlamaFamily => Dispatched::LlamaFamily(Box::new(
                LlamaConfig::from_file(file).map_err(LoaderError::from)?,
            )),
            Handler::RegisteredStub => {
                Dispatched::RegisteredStub(Box::new(stub::load(file, entry.architecture())?))
            }
        };
        diag::debug(
            "dispatched",
            &format!(
                "arch={} module_version={} kind={:?} capabilities={}",
                entry.architecture(),
                entry.module_version(),
                entry.kind(),
                entry.capabilities()
            ),
        );
        Ok(DispatchResult { entry, load })
    }
}

/// One engine the serve path could run a load on, described by
/// capability: whether the backend is available on this machine and
/// which architectures it can run a forward for. There is
/// deliberately no size field: selection never routes by file size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineCandidate {
    /// Engine name as it should appear in the load result.
    pub name: &'static str,
    /// Whether the backend is available right now (a Vulkan engine
    /// on a machine whose GPU probe failed is not a candidate).
    pub available: bool,
    /// Architectures this engine can run a forward pass for.
    pub forward_architectures: &'static [&'static str],
}

impl EngineCandidate {
    /// Whether this candidate can serve `architecture`: available,
    /// and the architecture is in its forward list.
    #[must_use]
    pub fn serves(&self, architecture: &str) -> bool {
        self.available && self.forward_architectures.contains(&architecture)
    }
}

/// The outcome of engine selection, stated in full so a load result
/// can carry it: which engine, for which architecture, backed by
/// which module capabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineSelection {
    /// The chosen engine's name.
    pub engine: String,
    /// The architecture being served.
    pub architecture: String,
    /// The capabilities of the module behind the architecture.
    pub module_capabilities: Capabilities,
}

/// Select the engine for a dispatched load, by capability only.
///
/// The module must provide FORWARD (a registered stub is refused
/// here no matter what the candidates claim, because the refusal is
/// about the module, not the machines). Among the candidates, the
/// first in caller priority order that is available and lists the
/// load's architecture in its forward set wins.
///
/// # Errors
/// [`LoaderError::NoCapableEngine`] naming the architecture and the
/// capability check that failed: the module has no forward, or no
/// available candidate lists the architecture.
pub fn select_engine(
    result: &DispatchResult,
    candidates: &[EngineCandidate],
) -> Result<EngineSelection> {
    let arch = result.load.architecture().to_string();
    let caps = result.entry.capabilities();
    if !caps.contains(Capabilities::FORWARD) {
        diag::debug(
            "engine_refused",
            &format!("arch={arch} reason=module_has_no_forward capabilities={caps}"),
        );
        return Err(LoaderError::NoCapableEngine {
            architecture: arch,
            detail: format!(
                "the registered module is a stub (capabilities {caps}); \
                 it routes and reads config but cannot generate yet"
            ),
        });
    }
    for candidate in candidates {
        if candidate.serves(&arch) {
            diag::debug(
                "engine_selected",
                &format!("arch={arch} engine={}", candidate.name),
            );
            return Ok(EngineSelection {
                engine: candidate.name.to_string(),
                architecture: arch,
                module_capabilities: caps,
            });
        }
    }
    let considered: Vec<&str> = candidates.iter().map(|c| c.name).collect();
    diag::debug(
        "engine_refused",
        &format!(
            "arch={arch} reason=no_candidate candidates={}",
            considered.join(",")
        ),
    );
    Err(LoaderError::NoCapableEngine {
        architecture: arch,
        detail: format!(
            "no available engine lists this architecture for forward; considered: {}",
            if considered.is_empty() {
                "none".to_string()
            } else {
                considered.join(", ")
            }
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub_result(arch: &'static str) -> DispatchResult {
        let entry = ArchEntry::stub(arch, 1);
        DispatchResult {
            entry,
            load: Dispatched::RegisteredStub(Box::new(StubConfig {
                architecture: arch.to_string(),
                block_count: 2,
                hidden: 32,
                context_length: 128,
                tensor_count: 1,
            })),
        }
    }

    fn llama_result(arch: &'static str) -> DispatchResult {
        let entry = ArchEntry::full(arch, 1, Handler::LlamaFamily);
        DispatchResult {
            entry,
            load: Dispatched::LlamaFamily(Box::new(LlamaConfig {
                architecture: arch.to_string(),
                name: None,
                block_count: 2,
                context_length: 128,
                hidden: 32,
                ffn_hidden: 48,
                q_heads: 4,
                kv_heads: 2,
                head_dim: 8,
                rope_freq_base: 10_000.0,
                rope_dim: 8,
                rms_eps: 1e-6,
                vocab_size: 64,
            })),
        }
    }

    #[test]
    fn standard_registry_supported_list_is_complete_and_sorted() {
        let registry = ArchRegistry::standard();
        assert_eq!(
            registry.supported(),
            [
                "deepseek2",
                "gemma2",
                "llama",
                "muse-glimmer",
                "olmo2",
                "qwen2"
            ]
        );
        for name in registry.supported() {
            let entry = registry.lookup(&name).expect("listed means registered");
            assert_eq!(entry.module_version(), 1);
        }
        assert_eq!(REGISTRY_VERSION, 1);
    }

    #[test]
    fn standard_registry_kinds_and_capabilities() {
        let registry = ArchRegistry::standard();
        let glimmer = registry.lookup("muse-glimmer").expect("glimmer");
        assert_eq!(glimmer.kind(), ModuleKind::Full);
        assert_eq!(glimmer.handler(), Handler::Glimmer);
        assert!(glimmer.capabilities().contains(Capabilities::FORWARD));
        let llama = registry.lookup("llama").expect("llama");
        assert_eq!(llama.handler(), Handler::LlamaFamily);
        let gemma = registry.lookup("gemma2").expect("gemma2");
        assert_eq!(gemma.kind(), ModuleKind::RegisteredStub);
        assert!(!gemma.capabilities().contains(Capabilities::FORWARD));
        assert!(gemma.capabilities().contains(Capabilities::CONFIG));
        assert!(registry.lookup("phi3").is_none());
    }

    #[test]
    fn duplicate_registration_is_refused_by_name() {
        let mut registry = ArchRegistry::new();
        registry
            .register(ArchEntry::stub("phi3", 1))
            .expect("first");
        let err = registry
            .register(ArchEntry::stub("phi3", 2))
            .expect_err("second registration of one string must fail");
        match err {
            LoaderError::DuplicateArchitecture { architecture } => {
                assert_eq!(architecture, "phi3")
            }
            other => panic!("expected DuplicateArchitecture, got {other:?}"),
        }
        // The first entry survives untouched.
        assert_eq!(registry.lookup("phi3").expect("phi3").module_version(), 1);
    }

    #[test]
    fn capabilities_compose_and_name() {
        let caps = Capabilities::CONFIG | Capabilities::FORWARD;
        assert!(caps.contains(Capabilities::CONFIG));
        assert!(!caps.contains(Capabilities::COVERAGE));
        assert_eq!(caps.to_string(), "config+forward");
        assert_eq!(Capabilities::EMPTY.to_string(), "none");
    }

    #[test]
    fn engine_selection_refuses_stub_modules_by_capability() {
        // Even a candidate that claims the architecture cannot make
        // a stub serve: the module itself has no forward.
        let result = stub_result("gemma2");
        let candidates = [EngineCandidate {
            name: "cpu-reference",
            available: true,
            forward_architectures: &["gemma2"],
        }];
        let err = select_engine(&result, &candidates).expect_err("stub cannot serve");
        match err {
            LoaderError::NoCapableEngine {
                architecture,
                detail,
            } => {
                assert_eq!(architecture, "gemma2");
                assert!(detail.contains("stub"), "detail: {detail}");
            }
            other => panic!("expected NoCapableEngine, got {other:?}"),
        }
    }

    #[test]
    fn engine_selection_skips_unavailable_candidates() {
        let result = llama_result("llama");
        let candidates = [
            EngineCandidate {
                name: "vulkan-7900xtx",
                available: false,
                forward_architectures: &["llama"],
            },
            EngineCandidate {
                name: "cpu-reference",
                available: true,
                forward_architectures: &["llama", "qwen2"],
            },
        ];
        let selection = select_engine(&result, &candidates).expect("cpu serves");
        assert_eq!(selection.engine, "cpu-reference");
        assert_eq!(selection.architecture, "llama");
        assert!(selection
            .module_capabilities
            .contains(Capabilities::FORWARD));
    }

    #[test]
    fn engine_selection_names_the_architecture_when_nobody_serves_it() {
        let result = llama_result("qwen2");
        let err = select_engine(&result, &[]).expect_err("no candidates, no engine");
        assert!(err.to_string().contains("qwen2"), "error: {err}");
    }
}
