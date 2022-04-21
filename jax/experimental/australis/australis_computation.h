#ifndef JAX_EXPERIMENTAL_AUSTRALIS_AUSTRALIS_COMPUTATION_H_
#define JAX_EXPERIMENTAL_AUSTRALIS_AUSTRALIS_COMPUTATION_H_

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "jax/experimental/australis/client.h"
#include "jax/experimental/australis/petri.h"

namespace aux {
namespace internal {

// Superclass for australis code-generated computations.
//
// DO NOT SUBCLASS ME IN HUMAN-EDITED SOURCE CODE!
class AustralisComputation {
 protected:
  static absl::StatusOr<AustralisComputation> LoadHlo(
      aux::Client client, absl::string_view hlo_binary_text,
      absl::string_view executable_spec, int64_t version);

  absl::StatusOr<PTree> ExecuteInternal(std::vector<const DeviceArray*> inputs);

 private:
  explicit AustralisComputation(std::unique_ptr<Executable> e, PTree out_shape)
      : executable_(std::move(e)), out_shape_(std::move(out_shape)) {}
  std::unique_ptr<aux::Executable> executable_;
  PTree out_shape_;
};

}  // namespace internal
}  // namespace aux

#endif  // JAX_EXPERIMENTAL_AUSTRALIS_AUSTRALIS_COMPUTATION_H_
