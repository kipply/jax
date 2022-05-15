#include "third_party/py/jax/experimental/australis/client-private.h"
#include "third_party/tensorflow/compiler/xla/pjrt/gpu_device.h"

namespace aux {
namespace internal {

BackendFactoryRegister _register_gpu(
    "gpu", +[]() -> absl::StatusOr<std::shared_ptr<xla::PjRtClient>> {
      return ToAbslStatusOr(xla::GetGpuClient(/*asynchronous=*/true, xla::GpuAllocatorConfig(),
                          /*distributed_client=*/nullptr, /*node_id=*/0)
    });

}  // namespace internal
}  // namespace aux
