//! The coopmat device bridge: how a device that can run
//! cooperative-matrix shaders is opened inside a backend
//! whose framework cannot create one.
//!
//! wgpu 27 exposes no cooperative matrix support at any
//! layer (no feature bit, no WGSL syntax, no Naga IR
//! type; bench/coopmat-feasibility-2026-11-06.md pins
//! the greps), and its own Vulkan device creation can
//! never enable VK_KHR_cooperative_matrix: the device
//! extension list is computed from the requested wgpu
//! features alone. A passthrough SPIR-V module declaring
//! the coopmat capability against such a device is a
//! spec violation, not a shortcut.
//!
//! The bridge is wgpu-hal's own sanctioned interop, used
//! end to end: the hal adapter's `open_with_callback`
//! creates the device exactly as the stock path does
//! (same extension computation, same feature selection,
//! same queue family), and a [`CreateDeviceCallback`]
//! adds the one extension wgpu cannot name plus its
//! feature struct to the create chain; the resulting
//! hal device is then wrapped by the safe
//! `Adapter::create_device_from_hal`, requesting
//! EXPERIMENTAL_PASSTHROUGH_SHADERS so coopmat SPIR-V
//! can enter through `create_shader_module_passthrough`.
//! The callback's own contract is the design's: "You
//! must not remove anything from this list, but you may
//! add to it."
//!
//! Gating: the bridge is entered only when the
//! environment names it ([`COOPMAT_BRIDGE_ENV`], exact
//! value `1`). With the variable unset or any other
//! value, `GpuContext::open` runs the shipped
//! `request_device` path untouched. The bridge is
//! fail-closed in the other direction too: every probe
//! below (extension exposed, feature bit true, the
//! s8 x s8 to s32 16x16x16 subgroup property row
//! present) must pass before any device is created, and
//! any refusal is returned as a named string; the
//! caller logs it on the diag channel and opens the
//! stock device instead, so a machine without coopmat
//! serves exactly as it did before this module existed,
//! with the refusal on record. The coopmat kernel
//! module reads [`GpuContext::coopmat`] and exists only
//! when this bridge actually delivered a device: a
//! routing request against a stock device refuses by
//! name there, never silently.
//!
//! The required property row is the one the llama
//! parity spec pins for `mul_mmq_cm1` on this card
//! family (bench/llama-parity-specs-2026-11-06.md):
//! subgroup scope, 16x16x16, signed 8-bit A and B,
//! signed 32-bit accumulate, the shape ggml's
//! `coopmat_int_support` requires. The step-0 probe on
//! the RX 7900 XTX found that row among its 14 rows;
//! see bench/coopmat-build-2026-11-06.md.
//!
//! [`CreateDeviceCallback`]: wgpu_hal::vulkan::CreateDeviceCallback
//! [`GpuContext::coopmat`]: crate::device::GpuContext::coopmat

use crate::diag::dbg_log;

/// Environment variable naming the coopmat device
/// bridge switch: the exact value `1` opens the GPU
/// device through the bridge in this module (ash
/// device creation with VK_KHR_cooperative_matrix
/// enabled, handed to wgpu through its hal interop).
/// Operational config, not a compiled constant, and
/// deliberately fail-closed: an unset variable, or any
/// other value, keeps the shipped `request_device`
/// path byte-for-byte, so every existing proof ran on
/// the device it thinks it ran on. The server proof
/// that can move this switch's default is sequenced in
/// bench/coopmat-build-2026-11-06.md.
pub const COOPMAT_BRIDGE_ENV: &str = "WGPU_COMPUTE_COOPMAT_BRIDGE";

/// Resolve the bridge switch from the raw environment
/// value: only the exact value `1` (after trimming)
/// enables it; everything else is off.
pub fn resolve_coopmat_bridge(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1"))
}

/// Whether this process asked for the bridged device.
pub fn coopmat_bridge_requested() -> bool {
    resolve_coopmat_bridge(std::env::var(COOPMAT_BRIDGE_ENV).ok().as_deref())
}

/// What the bridge probe confirmed about the device it
/// opened. Recorded on [`GpuContext`](crate::device::GpuContext)
/// at construction; the coopmat kernel module gates its
/// own existence on this being present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopmatInfo {
    /// Cooperative matrix property rows the driver
    /// reported for the physical device (14 on the
    /// server's RADV gfx1100 at the step-0 probe).
    pub property_rows: u32,
    /// The physical device's subgroup size, read from
    /// VkPhysicalDeviceSubgroupProperties in the same
    /// probe (64 on the server's RADV). The coopmat
    /// shader's workgroup geometry is compiled for a
    /// fixed subgroup size, so the kernel constructor
    /// checks this value and refuses by name on a
    /// mismatch.
    pub subgroup_size: u32,
}

/// Open a device on `adapter` through the bridge,
/// requesting the features in `desc` (which the caller
/// has already extended with
/// EXPERIMENTAL_PASSTHROUGH_SHADERS). On success the
/// returned context info certifies that the probes
/// passed and the device really carries the extension.
///
/// # Safety contract
///
/// This function is the crate's one `#[allow(unsafe_code)]`
/// site, and every unsafe call in it is one of the
/// following, each with its contract:
///
/// - `Adapter::as_hal` and the hal adapter's accessors:
///   the adapter came from this process's own Vulkan
///   instance enumeration (detect enumerates VULKAN
///   only), so the Vulkan hal adapter exists; the
///   returned guard borrows the adapter and is held
///   until the hal device is wrapped.
/// - The ash probes (`enumerate_device_extension_
///   properties`, `get_physical_device_features2`, the
///   KHR property query): read-only queries against the
///   hal adapter's own physical device handle, through
///   the hal instance's own ash function tables
///   (`shared_instance().raw_instance()`). The one
///   object built from this crate's own ash `Entry` is
///   the KHR extension's function table; an ash `Entry`
///   is only a loader handle for `vkGetInstanceProcAddr`
///   against the same libvulkan, never a second
///   instance.
/// - The create callback's chain edit: the callback
///   adds VK_KHR_cooperative_matrix to the extension
///   list (add-only, as the callback contract requires)
///   and chains a `VkPhysicalDeviceCooperativeMatrix
///   FeaturesKHR` with cooperativeMatrix = true into
///   the create info. The feature struct is a local of
///   this function; the callback runs synchronously
///   inside `open_with_callback`, and the device is
///   created inside that same call, so the struct
///   outlives every use of the chain. The extensions
///   and features wgpu-hal computed are not modified
///   in any other way, so the device the bridge returns
///   differs from the stock device by exactly the
///   coopmat extension and feature.
/// - `Adapter::create_device_from_hal`: the hal device
///   came from this same hal adapter's
///   `open_with_callback`, and `desc` requests exactly
///   the feature set the hal device was opened with,
///   so the subset clause of its contract holds.
///
/// # Errors
///
/// A named refusal string for every probe failure,
/// extension absence, or hal failure; the caller logs
/// it and falls back to the stock device.
#[allow(unsafe_code)]
pub(crate) fn open_bridged(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue, CoopmatInfo), String> {
    use ash::vk;
    use std::ffi::CStr;

    // SAFETY: per the module contract above; each call
    // is annotated at its site.
    unsafe {
        let Some(hal_guard) = adapter.as_hal::<wgpu_hal::vulkan::Api>() else {
            return Err("the selected adapter has no Vulkan hal adapter".to_string());
        };
        let hal = &*hal_guard;
        let ash_instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();

        // Probe 1: the extension must be exposed.
        let extensions = ash_instance
            .enumerate_device_extension_properties(physical)
            .map_err(|e| format!("device extension enumeration failed: {e:?}"))?;
        let cm_ext_name = ash::khr::cooperative_matrix::NAME;
        let exposed = extensions.iter().any(|e| {
            // Extension names are NUL-terminated per the
            // Vulkan spec; the array is 256 bytes.
            CStr::from_ptr(e.extension_name.as_ptr()) == cm_ext_name
        });
        if !exposed {
            return Err(
                "VK_KHR_cooperative_matrix is not exposed by this physical device".to_string(),
            );
        }

        // Probe 2: the feature bit, and the subgroup
        // size. The subgroup properties chain into the
        // properties query, not the features query; the
        // two structs live in different pNext families.
        let mut cm_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut cm_features);
        ash_instance.get_physical_device_features2(physical, &mut features2);
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
        ash_instance.get_physical_device_properties2(physical, &mut props2);
        if cm_features.cooperative_matrix == 0 {
            return Err(
                "VK_KHR_cooperative_matrix is exposed but its cooperativeMatrix \
                 feature is false on this physical device"
                    .to_string(),
            );
        }
        let subgroup_size = subgroup_props.subgroup_size;

        // Probe 3: the property rows must carry the one
        // shape the kernel is built for: s8 inputs, s32
        // accumulate, 16x16x16, subgroup scope.
        let entry = ash::Entry::load()
            .map_err(|e| format!("loading the Vulkan loader for the probe failed: {e}"))?;
        let cm_fn = ash::khr::cooperative_matrix::Instance::new(&entry, ash_instance);
        let rows = cm_fn
            .get_physical_device_cooperative_matrix_properties(physical)
            .map_err(|e| format!("cooperative matrix property query failed: {e:?}"))?;
        let required_row = rows.iter().any(|r| {
            r.m_size == 16
                && r.n_size == 16
                && r.k_size == 16
                && r.a_type == vk::ComponentTypeKHR::SINT8
                && r.b_type == vk::ComponentTypeKHR::SINT8
                && r.c_type == vk::ComponentTypeKHR::SINT32
                && r.result_type == vk::ComponentTypeKHR::SINT32
                && r.scope == vk::ScopeKHR::SUBGROUP
        });
        if !required_row {
            return Err(format!(
                "no s8 x s8 to s32 16x16x16 subgroup cooperative matrix row \
                 among the {} rows this device reports",
                rows.len()
            ));
        }
        dbg_log!(
            "coopmat bridge probe: extension exposed, feature on, {} property rows, \
             s8 x s8 to s32 16x16x16 subgroup row present, subgroup size {subgroup_size}",
            rows.len()
        );

        // Device creation: wgpu-hal's own open, with the
        // callback adding exactly the coopmat extension
        // and its feature struct. `cm_enable` outlives
        // the call: the callback runs and the device is
        // created synchronously inside open_with_callback.
        let mut cm_enable =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let cm_enable_ref = &mut cm_enable;
        let callback: Box<wgpu_hal::vulkan::CreateDeviceCallback> = Box::new(
            move |args: wgpu_hal::vulkan::CreateDeviceCallbackArgs<'_, '_, '_>| {
                args.extensions.push(ash::khr::cooperative_matrix::NAME);
                let info = std::mem::take(args.create_info);
                *args.create_info = info.push_next(&mut *cm_enable_ref);
            },
        );
        let open_device = hal
            .open_with_callback(desc.required_features, &desc.memory_hints, Some(callback))
            .map_err(|e| format!("hal device open through the bridge failed: {e:?}"))?;
        let (device, queue) = adapter
            .create_device_from_hal::<wgpu_hal::vulkan::Api>(open_device, desc)
            .map_err(|e| format!("wrapping the bridged hal device failed: {e}"))?;
        Ok((
            device,
            queue,
            CoopmatInfo {
                property_rows: rows.len() as u32,
                subgroup_size,
            },
        ))
    }
}

/// Create a shader module from precompiled SPIR-V words
/// through wgpu's passthrough path, on a device this
/// bridge opened. This is the second and last unsafe
/// site in the crate, and it lives here so the unsafe
/// surface stays in one module.
///
/// # Safety contract
///
/// `create_shader_module_passthrough` is unsafe because
/// wgpu validates nothing: the words are fed to the
/// driver as-is. The contract this caller upholds:
///
/// - The words come from a checked-in artifact compiled
///   from the in-repo GLSL source by the recorded
///   rebuild script and validated with spirv-val at
///   build time (kernels/coopmat/rebuild.sh); they are
///   never assembled, patched, or generated at runtime.
/// - The device came from [`open_bridged`]: its probes
///   confirmed VK_KHR_cooperative_matrix is enabled, so
///   the cooperative matrix capability the module
///   declares is legal on it. Calling this on a stock
///   device would be a spec violation; the kernel
///   module gates its construction on
///   [`GpuContext::coopmat`](crate::device::GpuContext::coopmat)
///   being present, which only a bridged device sets.
#[allow(unsafe_code)]
pub(crate) fn passthrough_shader_module(
    device: &wgpu::Device,
    label: &str,
    spirv_words: &[u32],
) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptorPassthrough {
        label: Some(label),
        spirv: Some(std::borrow::Cow::Borrowed(spirv_words)),
        ..Default::default()
    };
    // SAFETY: per the contract above.
    unsafe { device.create_shader_module_passthrough(desc) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_switch_is_fail_closed() {
        assert!(resolve_coopmat_bridge(Some("1")));
        assert!(resolve_coopmat_bridge(Some(" 1 ")));
        assert!(!resolve_coopmat_bridge(None));
        assert!(!resolve_coopmat_bridge(Some("0")));
        assert!(!resolve_coopmat_bridge(Some("")));
        assert!(!resolve_coopmat_bridge(Some("true")));
        assert!(!resolve_coopmat_bridge(Some("11")));
    }
}
