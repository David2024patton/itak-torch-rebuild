//! Adapter feature readout: the diag-capture checklist's open
//! evaluation (2026-11-04 visibility research, carried into
//! bench/diag-capture-2026-11-06.md) asks whether
//! `wgpu::Features::PIPELINE_STATISTICS_QUERY` is offered on
//! gfx1100 under RADV, and, if it is, whether a device can be
//! opened with it and a pipeline-statistics query set created
//! without disturbing anything else.
//!
//! This example answers both from the adapter at runtime: it
//! enumerates every adapter wgpu can see, prints the facts the
//! backend's own detect path records plus the two feature bits
//! the session cares about (timestamp queries, which the diag
//! recorder already uses, and pipeline statistics), and, when
//! pipeline statistics are offered, tries to open a device
//! with the feature and create a query set for two stages.
//! Every refusal is printed by name; the exit code is 0 when
//! the readout itself completed, whatever the answers are
//! (the answers are data, not pass/fail).
//!
//! Usage: `adapter_features`

use std::process::ExitCode;

fn main() -> ExitCode {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    if adapters.is_empty() {
        println!("adapter_features: no Vulkan adapters enumerated");
        return ExitCode::SUCCESS;
    }
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        let features = adapter.features();
        let psq = features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY);
        let tsq = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        println!(
            "adapter[{i}] name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        println!("adapter[{i}] features: timestamp_query={tsq} pipeline_statistics_query={psq}");
        if psq {
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("adapter_features probe"),
                required_features: wgpu::Features::PIPELINE_STATISTICS_QUERY,
                required_limits: adapter.limits(),
                ..Default::default()
            })) {
                Ok((device, _queue)) => {
                    let _query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("adapter_features pipeline statistics"),
                        ty: wgpu::QueryType::PipelineStatistics(
                            wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
                                | wgpu::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
                        ),
                        count: 1,
                    });
                    println!(
                        "adapter[{i}] probe device opened with PIPELINE_STATISTICS_QUERY; \
                         pipeline-statistics query set created (count 1)"
                    );
                }
                Err(e) => {
                    println!("adapter[{i}] probe device refused: {e}");
                }
            }
        }
    }
    ExitCode::SUCCESS
}
