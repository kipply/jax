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

"""BUILD file macros to make it easy to use Australis."""

def australis(
        name,
        py_file = None,
        py_deps = [],
        args = [],
        cc_namespace = "jax::experimental",
        canonicalize_function_name = True):
    """Generates a C++ library by staging out functions defined in `name`.py.

    TODO(saeta): Include an example usage here? (Or point to alternate documentation.)

    Args:
      name: The name for this build target.
      py_file: (Optional.) The name of the Python file that defines the functions
        to be exported. If not specified `$(name).py` is assumed.
      py_deps: The list of py_libraries py_file depends upon.
      hardware_platform: The hardware platform to build the target for (one of: 'cpu',
        'gpu', 'tpu1', 'tpu4', 'tpu'). Default: 'cpu'. Unsupported (coming soon):
        'gpu'. :-) 'tpu1' means 1 chip (2 physical cores), 'tpu4' ('tpu' is a synonym)
        means 4 chips (8 physical cores).
      cc_namespace: The namespace for the C++ generated code.
      canonicalize_function_name: Modify the name of the function to follow typical
        C++ naming conventions.
    """

    if py_file == None:
        py_file = name + ".py"
    exporter_target_name = name + "_exporter"
    genrule_target_name = name + "_genrule"
    library_target_name = name + "_cc"

    header_name = name + ".h"
    impl_name = name + ".cc"
    cc_embed_impl_name = name + "_cc_embed.inc"

    additional_exporter_deps = []
    exporter_flags = " ".join(args)

    native.py_binary(
        name = exporter_target_name,
        srcs = [py_file],
        main = py_file,
        deps = py_deps + additional_exporter_deps,
    )

    native.genrule(
        name = genrule_target_name,
        srcs = [],
        outs = [
            header_name,
            impl_name,
            cc_embed_impl_name,
        ],
        cmd = ("./$(location " + exporter_target_name + ") --name=" + name +
               " --header_name=$(location " + header_name +
               ") --cc_embed_impl_name=$(location " + cc_embed_impl_name +
               ") --impl_name=$(location " + impl_name + ") " +
               exporter_flags + " --cc_namespace='" + cc_namespace +
               "' --canonicalize_function_name=" + str(canonicalize_function_name)),
        tools = [":" + exporter_target_name],
    )

    native.cc_library(
        name = library_target_name,
        hdrs = [header_name],
        srcs = [impl_name, cc_embed_impl_name],
        data = [],
        deps = [
            Label("//jax/experimental/australis:australis_computation"),
        ],
    )
