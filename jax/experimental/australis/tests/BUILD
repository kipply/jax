# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tests for australis.

load("//third_party/py/jax/experimental/australis:australis.bzl", "australis")

licenses(["notice"])

australis(
    name = "jit_test_fns",
    cc_namespace = "australis::test",
)

australis(
    name = "simple_pjit",
    cc_namespace = "australis::test",
)

australis(
    name = "tuple_pjit",
    cc_namespace = "australis::test",
)

australis(
    name = "higher_arity_pjit",
    cc_namespace = "australis::test",
)

australis(
    name = "multi_return_pjit",
    cc_namespace = "australis::test",
)

australis(
    name = "donate_arg_pjit",
    cc_namespace = "australis::test",
)

australis(
    name = "flax_jit",
    cc_namespace = "australis::test",
    py_deps = [
        "//third_party/py/flax",
        "//third_party/py/flax:optim_deprecated",
    ],
)

cc_test(
    name = "pjit_test",
    srcs = ["pjit_test.cc"],
    deps = [
        ":donate_arg_pjit_cc",
        ":higher_arity_pjit_cc",
        ":multi_return_pjit_cc",
        ":simple_pjit_cc",
        ":tuple_pjit_cc",
        "//base",
        "//file/localfile",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status",
        "//third_party/py/jax/experimental/australis",
        "//third_party/py/jax/experimental/australis:cpu_support",
        "//third_party/py/jax/experimental/australis:petri",
    ],
)

cc_test(
    name = "jit_test",
    srcs = ["jit_test.cc"],
    deps = [
        ":jit_test_fns_cc",
        "//base",
        "//file/localfile",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status",
        "//third_party/py/jax/experimental/australis",
        "//third_party/py/jax/experimental/australis:cloud_tpu_support",
        "//third_party/py/jax/experimental/australis:cpu_support",
        "//third_party/py/jax/experimental/australis:petri",
    ],
)

cc_test(
    name = "jit_test_gpu",
    srcs = ["jit_test.cc"],
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":jit_test_fns_cc",
        "//base",
        "//file/localfile",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status",
        "//third_party/py/jax/experimental/australis",
        "//third_party/py/jax/experimental/australis:gpu_support",
        "//third_party/py/jax/experimental/australis:petri",
    ],
)

cc_test(
    name = "flax_test",
    srcs = ["flax_test.cc"],
    linkstatic = 1,
    deps = [
        ":flax_jit_cc",
        "//testing/base/public:gunit_main",
        "//third_party/absl/status",
        "//third_party/py/jax/experimental/australis",
        "//third_party/py/jax/experimental/australis:cpu_support",
        "//third_party/py/jax/experimental/australis:petri",
    ],
)
