#include "libttl.h"
#include "manifest_json.h"

#include <ATen/ATen.h>
#if defined(TTL_LIBTORCH_CUDA)
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#elif defined(TTL_LIBTORCH_MPS)
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#else
#error "select a TTL LibTorch backend"
#endif

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ttl_device {
    explicit ttl_device(c10::Device value) : value(std::move(value)) {}
    c10::Device value;
};

struct ttl_tensor {
    ttl_tensor(at::Tensor storage, ttl_dtype_t dtype,
               std::vector<int64_t> shape, std::vector<int64_t> stride)
        : storage(std::move(storage)), dtype(dtype), shape(std::move(shape)),
          stride(std::move(stride)) {}
    at::Tensor storage;
    ttl_dtype_t dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
};

struct ttl_stream {
#if defined(TTL_LIBTORCH_CUDA)
    explicit ttl_stream(c10::cuda::CUDAStream value) : value(value) {}
    c10::cuda::CUDAStream value;
#else
    explicit ttl_stream(at::mps::MPSStream *value) : value(value) {}
    at::mps::MPSStream *value;
#endif
};

struct ttl_timer {
#if defined(TTL_LIBTORCH_CUDA)
    explicit ttl_timer(c10::DeviceIndex device_index)
        : device_index(device_index), begin_event(cudaEventDefault),
          end_event(cudaEventDefault) {}
    c10::DeviceIndex device_index;
    c10::cuda::CUDAEvent begin_event;
    c10::cuda::CUDAEvent end_event;
#else
    explicit ttl_timer(c10::DeviceIndex device_index)
        : device_index(device_index) {}
    c10::DeviceIndex device_index;
    std::chrono::steady_clock::time_point begin_time;
#endif
    bool active = false;
};

enum class tensor_access { read, write, read_write };

struct tensor_spec {
    std::string name;
    ttl_dtype_t dtype;
    tensor_access access;
    std::vector<int64_t> shape;
    bool contiguous;
};

struct scalar_spec {
    std::string name;
    ttl_dtype_t dtype;
};

enum class argument_kind { tensor, scalar_value, scalar_device, literal };

struct argument_spec {
    argument_kind kind;
    size_t index = 0;
    ttl_dtype_t dtype = TTL_DTYPE_INT32;
    int32_t int32_value = 0;
    float float32_value = 0.0f;
};

struct module_spec {
    std::string name;
    std::string kernel_name;
    std::array<unsigned int, 3> grid{};
    std::array<unsigned int, 3> block{};
    unsigned int shared_bytes = 0;
    std::string artifact_file;
    std::vector<tensor_spec> tensors;
    std::vector<scalar_spec> scalars;
    std::vector<argument_spec> arguments;
};

struct program_tensor_spec {
    std::string name;
    ttl_dtype_t dtype;
    std::vector<int64_t> shape;
};

struct program_call_spec {
    std::string function;
    std::vector<std::string> arguments;
    std::vector<std::string> results;
};

struct program_spec {
    std::string name;
    std::vector<program_tensor_spec> inputs;
    std::vector<program_tensor_spec> outputs;
    std::vector<program_call_spec> calls;
};

ttl_capability_set_t backend_capabilities() {
#if defined(TTL_LIBTORCH_CUDA)
    return TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY |
        TTL_CAPABILITY_INDEPENDENT_STREAMS |
        TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION;
#else
    return 0;
#endif
}

ttl_capability_set_t required_capabilities(const module_spec &spec) {
    return spec.shared_bytes == 0
        ? 0 : TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY;
}

std::string capability_names(ttl_capability_set_t capabilities) {
    std::vector<std::string> names;
    if ((capabilities & TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY) != 0) {
        names.emplace_back("dynamic_workgroup_memory");
    }
    if ((capabilities & TTL_CAPABILITY_INDEPENDENT_STREAMS) != 0) {
        names.emplace_back("independent_streams");
    }
    if ((capabilities & TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION) != 0) {
        names.emplace_back("native_graph_execution");
    }
    std::string result;
    for (const auto &name : names) {
        if (!result.empty()) result += ", ";
        result += name;
    }
    return result.empty() ? "none" : result;
}

struct ttl_module {
    ttl_module(std::filesystem::path directory, module_spec spec)
        : directory(std::move(directory)), spec(std::move(spec)) {}
    std::filesystem::path directory;
    module_spec spec;
};

struct ttl_kernel {
#if defined(TTL_LIBTORCH_CUDA)
    ttl_kernel(module_spec spec, CUmodule module, CUfunction function,
               c10::DeviceIndex device_index,
               std::vector<at::Tensor> scalar_storage)
        : spec(std::move(spec)), module(module), function(function),
          device_index(device_index),
          scalar_storage(std::move(scalar_storage)) {}
    ~ttl_kernel() {
        if (module != nullptr) cuModuleUnload(module);
    }
    module_spec spec;
    CUmodule module;
    CUfunction function;
    c10::DeviceIndex device_index;
    std::vector<at::Tensor> scalar_storage;
    c10::cuda::CUDAEvent scalar_staging_event{cudaEventDisableTiming};
    bool scalar_staging_active = false;
    std::mutex scalar_staging_mutex;
#else
    ttl_kernel(module_spec spec,
               std::unique_ptr<at::native::mps::PrecompiledMetalShaderLibrary> library,
               std::shared_ptr<at::native::mps::MetalKernelFunction> function,
               c10::DeviceIndex device_index,
               std::vector<at::Tensor> scalar_storage)
        : spec(std::move(spec)), library(std::move(library)),
          function(std::move(function)), device_index(device_index),
          scalar_storage(std::move(scalar_storage)) {}
    module_spec spec;
    std::unique_ptr<at::native::mps::PrecompiledMetalShaderLibrary> library;
    std::shared_ptr<at::native::mps::MetalKernelFunction> function;
    c10::DeviceIndex device_index;
    std::vector<at::Tensor> scalar_storage;
#endif
};

struct ttl_program {
    ttl_program(std::filesystem::path directory, program_spec spec)
        : directory(std::move(directory)), spec(std::move(spec)) {}
    std::filesystem::path directory;
    program_spec spec;
};

struct program_call_execution {
    ttl_kernel_t *kernel = nullptr;
    std::vector<ttl_tensor_t *> tensors;
    std::vector<size_t> dependencies;
#if defined(TTL_LIBTORCH_CUDA)
    CUgraphNode native_node = nullptr;
    std::vector<std::array<unsigned char, 8>> argument_values;
    std::vector<void *> native_arguments;
#endif
};

struct ttl_program_execution {
    ~ttl_program_execution() {
#if defined(TTL_LIBTORCH_CUDA)
        if (native_execution != nullptr) cuGraphExecDestroy(native_execution);
        if (native_graph != nullptr) cuGraphDestroy(native_graph);
#endif
        for (auto *tensor : intermediates) delete tensor;
        for (auto *kernel : kernels) delete kernel;
        for (auto *module : modules) delete module;
    }
    std::vector<ttl_module_t *> modules;
    std::vector<ttl_kernel_t *> kernels;
    std::vector<ttl_tensor_t *> intermediates;
    std::vector<program_call_execution> calls;
#if defined(TTL_LIBTORCH_CUDA)
    CUgraph native_graph = nullptr;
    CUgraphExec native_execution = nullptr;
    c10::DeviceIndex device_index = -1;
#endif
};

namespace {

thread_local std::string last_error;

ttl_status_t fail(ttl_status_t status, std::string message) {
    last_error = std::move(message);
    return status;
}

template <typename Function>
ttl_status_t protect(Function &&function) {
    try {
        last_error.clear();
        function();
        return TTL_STATUS_OK;
    } catch (const c10::Error &error) {
        return fail(TTL_STATUS_BACKEND_ERROR, error.what_without_backtrace());
    } catch (const std::invalid_argument &error) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::out_of_range &error) {
        return fail(TTL_STATUS_OUT_OF_RANGE, error.what());
    } catch (const std::exception &error) {
        return fail(TTL_STATUS_BACKEND_ERROR, error.what());
    } catch (...) {
        return fail(TTL_STATUS_BACKEND_ERROR, "unknown backend failure");
    }
}

size_t dtype_bit_width(ttl_dtype_t dtype) {
    switch (dtype) {
        case TTL_DTYPE_FLOAT4_E2M1: return 4;
        case TTL_DTYPE_BOOL:
        case TTL_DTYPE_INT8:
        case TTL_DTYPE_FLOAT8_E4M3:
        case TTL_DTYPE_FLOAT8_E8M0: return 8;
        case TTL_DTYPE_FLOAT16: return 16;
        case TTL_DTYPE_INT32:
        case TTL_DTYPE_FLOAT32: return 32;
    }
    throw std::invalid_argument("unsupported TTL dtype");
}

size_t storage_bytes(ttl_dtype_t dtype, size_t element_count) {
    const size_t bits = dtype_bit_width(dtype);
    if (bits == 4) {
        return element_count / 2 + element_count % 2;
    }
    const size_t bytes_per_element = bits / 8;
    if (element_count > std::numeric_limits<size_t>::max() / bytes_per_element) {
        throw std::overflow_error("tensor byte size does not fit size_t");
    }
    return element_count * bytes_per_element;
}

std::vector<int64_t> checked_shape(size_t rank, const int64_t *shape) {
    if (rank == 0) return {};
    if (shape == nullptr) {
        throw std::invalid_argument("shape is NULL for a non-scalar tensor");
    }
    std::vector<int64_t> result(shape, shape + rank);
    for (int64_t extent : result) {
        if (extent < 0) {
            throw std::invalid_argument("tensor extents must be non-negative");
        }
    }
    return result;
}

size_t element_count(const std::vector<int64_t> &shape) {
    size_t count = 1;
    for (int64_t extent : shape) {
        const size_t value = static_cast<size_t>(extent);
        if (value != 0 && count > std::numeric_limits<size_t>::max() / value) {
            throw std::overflow_error("tensor element count does not fit size_t");
        }
        count *= value;
    }
    return count;
}

std::vector<int64_t> contiguous_stride(const std::vector<int64_t> &shape) {
    std::vector<int64_t> stride(shape.size(), 1);
    int64_t running = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        stride[i] = running;
        if (shape[i] != 0 &&
            running > std::numeric_limits<int64_t>::max() / shape[i]) {
            throw std::overflow_error("tensor stride does not fit int64_t");
        }
        running *= shape[i];
    }
    return stride;
}

ttl_tensor_t *allocate_tensor(
    const c10::Device &device, ttl_dtype_t dtype,
    std::vector<int64_t> shape) {
    const size_t bytes = storage_bytes(dtype, element_count(shape));
    if (bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("tensor allocation exceeds backend size limit");
    }
    auto storage = at::empty(
        {static_cast<int64_t>(bytes)},
        at::TensorOptions().dtype(at::kByte).device(device));
    return new ttl_tensor(
        std::move(storage), dtype, shape, contiguous_stride(shape));
}

template <typename Value>
Value load_value(const uint8_t *data, size_t byte_offset) {
    Value value;
    std::memcpy(&value, data + byte_offset, sizeof(value));
    return value;
}

double float8_e4m3_as_double(uint8_t value) {
    const bool negative = (value & 0x80u) != 0;
    const uint8_t exponent = (value >> 3) & 0x0fu;
    const uint8_t mantissa = value & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double magnitude = exponent == 0
        ? std::ldexp(static_cast<double>(mantissa), -9)
        : std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                     static_cast<int>(exponent) - 7);
    return negative ? -magnitude : magnitude;
}

double element_as_double(
    const ttl_tensor_t &tensor, const uint8_t *data, size_t index) {
    static constexpr double fp4_values[16] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    };
    switch (tensor.dtype) {
        case TTL_DTYPE_BOOL: return data[index] != 0 ? 1.0 : 0.0;
        case TTL_DTYPE_INT8: return static_cast<int8_t>(data[index]);
        case TTL_DTYPE_INT32:
            return load_value<int32_t>(data, index * sizeof(int32_t));
        case TTL_DTYPE_FLOAT16:
            return static_cast<double>(
                load_value<at::Half>(data, index * sizeof(at::Half)));
        case TTL_DTYPE_FLOAT32:
            return load_value<float>(data, index * sizeof(float));
        case TTL_DTYPE_FLOAT4_E2M1: {
            const uint8_t packed = data[index / 2];
            const uint8_t encoding = index % 2 == 0
                ? packed & 0x0fu : packed >> 4;
            return fp4_values[encoding];
        }
        case TTL_DTYPE_FLOAT8_E4M3:
            return float8_e4m3_as_double(data[index]);
        case TTL_DTYPE_FLOAT8_E8M0:
            return data[index] == 0xffu
                ? std::numeric_limits<double>::quiet_NaN()
                : std::ldexp(1.0, static_cast<int>(data[index]) - 127);
    }
    throw std::invalid_argument("tensor dtype is outside the TTL profile");
}

using jsoncons::json;

ttl_dtype_t manifest_dtype(const std::string &name) {
    if (name == "bool") return TTL_DTYPE_BOOL;
    if (name == "int8") return TTL_DTYPE_INT8;
    if (name == "int32") return TTL_DTYPE_INT32;
    if (name == "float16") return TTL_DTYPE_FLOAT16;
    if (name == "float32") return TTL_DTYPE_FLOAT32;
    if (name == "float4_e2m1") return TTL_DTYPE_FLOAT4_E2M1;
    if (name == "float8_e4m3") return TTL_DTYPE_FLOAT8_E4M3;
    if (name == "float8_e8m0") return TTL_DTYPE_FLOAT8_E8M0;
    throw std::invalid_argument("unknown TTL module dtype: " + name);
}

std::array<unsigned int, 3> parse_geometry(const json &value) {
    std::array<unsigned int, 3> result{1, 1, 1};
    for (size_t i = 0; i < value.size(); ++i) {
        result[i] = value[i].as<unsigned int>();
    }
    return result;
}

module_spec parse_manifest(const std::filesystem::path &path) {
    const json root = ttl_internal::parse_and_validate_manifest(path);
    const json launch = ttl_internal::parse_and_validate_launch(
        path.parent_path() / "launch.json");
    module_spec result;
    result.name = root.at("name").as<std::string>();

    const json &kernel = root.at("kernel");
    result.kernel_name = kernel.at("name").as<std::string>();
    const json &native_kernel = launch.at("kernel");
    if (native_kernel.at("name").as<std::string>() != result.kernel_name) {
        throw std::invalid_argument(
            "private launch kernel does not match module manifest");
    }
    result.grid = parse_geometry(native_kernel.at("grid"));
    result.block = parse_geometry(native_kernel.at("block"));
    result.shared_bytes =
        native_kernel.at("shared_bytes").as<unsigned int>();

    const json &artifact = launch.at("artifact");
#if defined(TTL_LIBTORCH_CUDA)
    if (launch.at("backend").as<std::string>() != "libtorch_cuda" ||
        artifact.at("format").as<std::string>() != "cubin") {
        throw std::invalid_argument(
            "LibTorch CUDA backend requires a cubin launch description");
    }
#else
    if (launch.at("backend").as<std::string>() != "libtorch_mps" ||
        artifact.at("format").as<std::string>() != "metallib") {
        throw std::invalid_argument(
            "LibTorch MPS backend requires a metallib launch description");
    }
#endif
    result.artifact_file = artifact.at("file").as<std::string>();

    std::map<std::string, size_t> tensor_indices;
    for (const auto &tensor : root.at("tensors").array_range()) {
        tensor_spec spec;
        spec.name = tensor.at("name").as<std::string>();
        tensor_indices.emplace(spec.name, result.tensors.size());
        spec.dtype = manifest_dtype(tensor.at("dtype").as<std::string>());
        const std::string access = tensor.at("access").as<std::string>();
        if (access == "read") spec.access = tensor_access::read;
        else if (access == "write") spec.access = tensor_access::write;
        else spec.access = tensor_access::read_write;
        for (const auto &extent : tensor.at("shape").array_range()) {
            spec.shape.push_back(extent.as<int64_t>());
        }
        spec.contiguous = !tensor.contains("contiguous") ||
            tensor.at("contiguous").as<bool>();
        result.tensors.push_back(std::move(spec));
    }

    std::map<std::string, size_t> scalar_indices;
    for (const auto &scalar : root.at("scalars").array_range()) {
        const std::string name = scalar.at("name").as<std::string>();
        scalar_indices.emplace(name, result.scalars.size());
        result.scalars.push_back({
            name, manifest_dtype(scalar.at("dtype").as<std::string>())});
    }

    for (const auto &argument : root.at("arguments").array_range()) {
        const std::string kind = argument.at("kind").as<std::string>();
        argument_spec spec;
        if (kind == "tensor") spec.kind = argument_kind::tensor;
        else if (kind == "scalar_value") {
            spec.kind = argument_kind::scalar_value;
        } else if (kind == "scalar_device") {
            spec.kind = argument_kind::scalar_device;
        } else {
            spec.kind = argument_kind::literal;
        }

        if (spec.kind == argument_kind::tensor ||
            spec.kind == argument_kind::scalar_value ||
            spec.kind == argument_kind::scalar_device) {
            const std::string name = argument.at("name").as<std::string>();
            spec.index = spec.kind == argument_kind::tensor
                ? tensor_indices.at(name) : scalar_indices.at(name);
        } else {
            spec.dtype = manifest_dtype(
                argument.at("dtype").as<std::string>());
            if (spec.dtype == TTL_DTYPE_INT32) {
                spec.int32_value =
                    argument.at("value").as<int32_t>();
            } else {
                spec.float32_value =
                    argument.at("value").as<float>();
            }
        }
        result.arguments.push_back(spec);
    }
    return result;
}

program_tensor_spec parse_program_tensor(const json &tensor) {
    program_tensor_spec result;
    result.name = tensor.at("name").as<std::string>();
    result.dtype = manifest_dtype(tensor.at("dtype").as<std::string>());
    for (const auto &extent : tensor.at("shape").array_range()) {
        result.shape.push_back(extent.as<int64_t>());
    }
    return result;
}

program_spec parse_program(const std::filesystem::path &path) {
    const json root = ttl_internal::parse_and_validate_program(path);
    program_spec result;
    result.name = root.at("name").as<std::string>();

    std::map<std::string, bool> defined;
    for (const auto &tensor : root.at("inputs").array_range()) {
        auto spec = parse_program_tensor(tensor);
        if (!defined.emplace(spec.name, true).second) {
            throw std::invalid_argument("program input names must be unique");
        }
        result.inputs.push_back(std::move(spec));
    }

    std::map<std::string, bool> output_names;
    for (const auto &tensor : root.at("outputs").array_range()) {
        auto spec = parse_program_tensor(tensor);
        if (defined.count(spec.name) != 0 ||
            !output_names.emplace(spec.name, true).second) {
            throw std::invalid_argument(
                "program output names must be unique and distinct from inputs");
        }
        result.outputs.push_back(std::move(spec));
    }

    for (const auto &call : root.at("calls").array_range()) {
        program_call_spec spec;
        spec.function = call.at("function").as<std::string>();
        for (const auto &argument : call.at("arguments").array_range()) {
            const std::string name = argument.as<std::string>();
            if (defined.count(name) == 0) {
                throw std::invalid_argument(
                    "program calls must use values defined earlier: " + name);
            }
            spec.arguments.push_back(name);
        }
        for (const auto &value : call.at("results").array_range()) {
            const std::string name = value.as<std::string>();
            if (!defined.emplace(name, true).second) {
                throw std::invalid_argument(
                    "program call results must be fresh values: " + name);
            }
            spec.results.push_back(name);
        }
        result.calls.push_back(std::move(spec));
    }
    for (const auto &output : result.outputs) {
        if (defined.count(output.name) == 0) {
            throw std::invalid_argument(
                "program output is not produced: " + output.name);
        }
    }
    return result;
}

#if defined(TTL_LIBTORCH_CUDA)
void check_cuda(CUresult status, const char *operation) {
    if (status == CUDA_SUCCESS) return;
    const char *name = nullptr;
    const char *description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    throw std::runtime_error(
        std::string(operation) + " failed: " + (name == nullptr ? "CUDA_ERROR" : name) +
        " (" + (description == nullptr ? "no description" : description) + ")");
}
#endif

}  // namespace

extern "C" {

uint32_t ttl_api_version(void) { return TTL_API_VERSION; }

const char *ttl_last_error(void) { return last_error.c_str(); }

const char *ttl_dtype_name(ttl_dtype_t dtype) {
    switch (dtype) {
        case TTL_DTYPE_BOOL: return "bool";
        case TTL_DTYPE_INT8: return "int8";
        case TTL_DTYPE_INT32: return "int32";
        case TTL_DTYPE_FLOAT16: return "float16";
        case TTL_DTYPE_FLOAT32: return "float32";
        case TTL_DTYPE_FLOAT4_E2M1: return "float4_e2m1";
        case TTL_DTYPE_FLOAT8_E4M3: return "float8_e4m3";
        case TTL_DTYPE_FLOAT8_E8M0: return "float8_e8m0";
    }
    return "invalid";
}

size_t ttl_dtype_bit_width(ttl_dtype_t dtype) {
    try {
        return dtype_bit_width(dtype);
    } catch (...) {
        return 0;
    }
}

ttl_status_t ttl_dtype_storage_bytes(
    ttl_dtype_t dtype, size_t element_count, size_t *bytes) {
    if (bytes == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "bytes output is NULL");
    }
    return protect([&] { *bytes = storage_bytes(dtype, element_count); });
}

ttl_status_t ttl_device_count(size_t *count) {
    if (count == nullptr) return fail(TTL_STATUS_INVALID_ARGUMENT, "count is NULL");
#if defined(TTL_LIBTORCH_CUDA)
    return protect([&] { *count = static_cast<size_t>(c10::cuda::device_count()); });
#else
    return protect([&] { *count = at::mps::is_available() ? 1u : 0u; });
#endif
}

ttl_status_t ttl_device_open(size_t position, ttl_device_t **device) {
    if (device == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "device output is NULL");
    }
    *device = nullptr;
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        if (position >= static_cast<size_t>(c10::cuda::device_count())) {
            throw std::out_of_range("accelerator position is out of range");
        }
        *device = new ttl_device(c10::Device(
            c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(position)));
#else
        if (!at::mps::is_available() || position != 0) {
            throw std::out_of_range("accelerator position is out of range");
        }
        *device = new ttl_device(c10::Device(c10::DeviceType::MPS, 0));
#endif
    });
}

ttl_status_t ttl_device_capabilities(
    const ttl_device_t *device, ttl_capability_set_t *capabilities) {
    if (device == nullptr || capabilities == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "device and capabilities output are required");
    }
    return protect([&] { *capabilities = backend_capabilities(); });
}

void ttl_device_destroy(ttl_device_t *device) { delete device; }

ttl_status_t ttl_tensor_empty_host(
    ttl_dtype_t dtype, size_t rank, const int64_t *shape,
    ttl_tensor_t **tensor) {
    if (tensor == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor output is NULL");
    }
    *tensor = nullptr;
    return protect([&] {
        *tensor = allocate_tensor(
            c10::Device(c10::DeviceType::CPU), dtype,
            checked_shape(rank, shape));
    });
}

ttl_status_t ttl_tensor_empty(
    const ttl_device_t *device, ttl_dtype_t dtype, size_t rank,
    const int64_t *shape, ttl_tensor_t **tensor) {
    if (device == nullptr || tensor == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "device and tensor output are required");
    }
    *tensor = nullptr;
    return protect([&] {
        *tensor = allocate_tensor(
            device->value, dtype, checked_shape(rank, shape));
    });
}

void ttl_tensor_destroy(ttl_tensor_t *tensor) { delete tensor; }

ttl_status_t ttl_tensor_write(
    ttl_tensor_t *tensor, const void *source, size_t bytes) {
    if (tensor == nullptr || source == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and source are required");
    }
    return protect([&] {
        if (bytes != static_cast<size_t>(tensor->storage.numel())) {
            throw std::invalid_argument("write byte count does not match tensor size");
        }
        if (tensor->dtype == TTL_DTYPE_FLOAT4_E2M1 &&
            element_count(tensor->shape) % 2 != 0 &&
            (static_cast<const uint8_t *>(source)[bytes - 1] & 0xf0u) != 0) {
            throw std::invalid_argument(
                "unused float4 storage nibble must be zero");
        }
        auto source_tensor = at::from_blob(
            const_cast<void *>(source), {static_cast<int64_t>(bytes)},
            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        tensor->storage.copy_(source_tensor, false);
    });
}

ttl_status_t ttl_tensor_read(
    const ttl_tensor_t *tensor, void *destination, size_t bytes) {
    if (tensor == nullptr || destination == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "tensor and destination are required");
    }
    return protect([&] {
        if (bytes != static_cast<size_t>(tensor->storage.numel())) {
            throw std::invalid_argument("read byte count does not match tensor size");
        }
        at::Tensor host = tensor->storage.is_cpu()
            ? tensor->storage : tensor->storage.to(at::kCPU, false, true);
        std::memcpy(destination, host.const_data_ptr(), bytes);
    });
}

ttl_status_t ttl_tensor_copy(
    ttl_tensor_t *destination, const ttl_tensor_t *source,
    ttl_stream_t *stream) {
    if (destination == nullptr || source == nullptr || stream == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "destination, source, and stream are required");
    }
    return protect([&] {
        if (destination->shape != source->shape ||
            destination->dtype != source->dtype) {
            throw std::invalid_argument("copy tensors must have equal shape and dtype");
        }
#if defined(TTL_LIBTORCH_CUDA)
        c10::cuda::CUDAStreamGuard guard(stream->value);
#else
        if (stream->value != at::mps::getCurrentMPSStream()) {
            throw std::invalid_argument("stream is not the LibTorch MPS stream");
        }
#endif
        destination->storage.copy_(source->storage, true);
    });
}

ttl_status_t ttl_tensor_rank(const ttl_tensor_t *tensor, size_t *rank) {
    if (tensor == nullptr || rank == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and rank are required");
    }
    return protect([&] { *rank = tensor->shape.size(); });
}

ttl_status_t ttl_tensor_extent(
    const ttl_tensor_t *tensor, size_t dimension, int64_t *extent) {
    if (tensor == nullptr || extent == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and extent are required");
    }
    if (dimension >= tensor->shape.size()) {
        return fail(TTL_STATUS_OUT_OF_RANGE, "tensor dimension is out of range");
    }
    return protect([&] { *extent = tensor->shape[dimension]; });
}

ttl_status_t ttl_tensor_stride(
    const ttl_tensor_t *tensor, size_t dimension, int64_t *stride) {
    if (tensor == nullptr || stride == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and stride are required");
    }
    if (dimension >= tensor->stride.size()) {
        return fail(TTL_STATUS_OUT_OF_RANGE, "tensor dimension is out of range");
    }
    return protect([&] { *stride = tensor->stride[dimension]; });
}

ttl_status_t ttl_tensor_dtype(const ttl_tensor_t *tensor, ttl_dtype_t *dtype) {
    if (tensor == nullptr || dtype == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and dtype are required");
    }
    return protect([&] { *dtype = tensor->dtype; });
}

ttl_status_t ttl_tensor_byte_size(const ttl_tensor_t *tensor, size_t *bytes) {
    if (tensor == nullptr || bytes == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and bytes are required");
    }
    return protect([&] { *bytes = static_cast<size_t>(tensor->storage.numel()); });
}

ttl_status_t ttl_tensor_is_host(const ttl_tensor_t *tensor, bool *is_host) {
    if (tensor == nullptr || is_host == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and result are required");
    }
    return protect([&] { *is_host = tensor->storage.is_cpu(); });
}

ttl_status_t ttl_tensor_is_contiguous(
    const ttl_tensor_t *tensor, bool *is_contiguous) {
    if (tensor == nullptr || is_contiguous == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "tensor and result are required");
    }
    return protect([&] { *is_contiguous = true; });
}

ttl_status_t ttl_close(
    const ttl_tensor_t *actual,
    const ttl_tensor_t *reference,
    double relative_tolerance,
    double absolute_tolerance,
    ttl_close_result_t *result) {
    if (actual == nullptr || reference == nullptr || result == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "actual, reference, and result are required");
    }
    if (relative_tolerance < 0.0 || absolute_tolerance < 0.0) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "comparison tolerances must be non-negative");
    }
    return protect([&] {
        if (actual->shape != reference->shape) {
            throw std::invalid_argument("comparison tensors must have equal shapes");
        }
        (void)dtype_bit_width(actual->dtype);
        (void)dtype_bit_width(reference->dtype);
        const at::Tensor actual_host = actual->storage.to(at::kCPU).contiguous();
        const at::Tensor reference_host = reference->storage.to(at::kCPU).contiguous();
        const auto *actual_data = actual_host.const_data_ptr<uint8_t>();
        const auto *reference_data = reference_host.const_data_ptr<uint8_t>();
        const size_t count = element_count(actual->shape);

        *result = {true, 0, SIZE_MAX, 0.0, 0.0};
        for (size_t i = 0; i < count; ++i) {
            const double a = element_as_double(*actual, actual_data, i);
            const double r = element_as_double(*reference, reference_data, i);
            double absolute_error = 0.0;
            double relative_error = 0.0;
            if (a != r) {
                if (std::isnan(a) || std::isnan(r) ||
                    std::isinf(a) || std::isinf(r)) {
                    absolute_error = std::numeric_limits<double>::infinity();
                    relative_error = absolute_error;
                } else {
                    absolute_error = std::abs(a - r);
                    relative_error = r == 0.0
                        ? (absolute_error == 0.0 ? 0.0
                                                 : std::numeric_limits<double>::infinity())
                        : absolute_error / std::abs(r);
                }
            }
            if (absolute_error > result->max_absolute_error) {
                result->max_absolute_error = absolute_error;
                result->worst_linear_index = i;
            }
            if (relative_error > result->max_relative_error) {
                result->max_relative_error = relative_error;
            }
            const bool close = a == r || absolute_error <=
                absolute_tolerance + relative_tolerance * std::abs(r);
            if (!close) {
                result->is_close = false;
                ++result->mismatch_count;
                if (result->worst_linear_index == SIZE_MAX) {
                    result->worst_linear_index = static_cast<size_t>(i);
                }
            }
        }
    });
}

ttl_status_t ttl_stream_create(
    const ttl_device_t *device, ttl_stream_t **stream) {
    if (device == nullptr || stream == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "device and stream output are required");
    }
    *stream = nullptr;
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        *stream = new ttl_stream(
            c10::cuda::getStreamFromPool(false, device->value.index()));
#else
        if (!device->value.is_mps()) {
            throw std::invalid_argument("device is not an MPS accelerator");
        }
        // LibTorch currently exposes one serial MPS stream. TTL keeps the
        // handle explicit even though multiple handles share that queue.
        *stream = new ttl_stream(at::mps::getCurrentMPSStream());
#endif
    });
}

ttl_status_t ttl_stream_wait(
    ttl_stream_t *waiting, const ttl_stream_t *dependency) {
    if (waiting == nullptr || dependency == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "both streams are required");
    }
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        c10::cuda::CUDAEvent event;
        event.record(dependency->value);
        event.block(waiting->value);
#else
        if (waiting->value != dependency->value) {
            throw std::invalid_argument("MPS stream handles do not share a queue");
        }
#endif
    });
}

ttl_status_t ttl_stream_synchronize(ttl_stream_t *stream) {
    if (stream == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "stream is NULL");
    }
#if defined(TTL_LIBTORCH_CUDA)
    return protect([&] { stream->value.synchronize(); });
#else
    return protect([&] {
        stream->value->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    });
#endif
}

void ttl_stream_destroy(ttl_stream_t *stream) { delete stream; }

ttl_status_t ttl_timer_create(
    const ttl_device_t *device, ttl_timer_t **timer) {
    if (device == nullptr || timer == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "device and timer output are required");
    }
    *timer = nullptr;
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        if (!device->value.is_cuda()) {
            throw std::invalid_argument("timer device is not a CUDA accelerator");
        }
#else
        if (!device->value.is_mps()) {
            throw std::invalid_argument("timer device is not an MPS accelerator");
        }
#endif
        *timer = new ttl_timer(device->value.index());
    });
}

ttl_status_t ttl_timer_begin(ttl_timer_t *timer, ttl_stream_t *stream) {
    if (timer == nullptr || stream == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "timer and stream are required");
    }
    if (timer->active) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "timer interval is already active");
    }
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        if (timer->device_index != stream->value.device_index()) {
            throw std::invalid_argument("timer and stream devices do not match");
        }
        timer->begin_event.record(stream->value);
#else
        if (timer->device_index != 0 ||
            stream->value != at::mps::getCurrentMPSStream()) {
            throw std::invalid_argument("timer and stream devices do not match");
        }
        stream->value->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        timer->begin_time = std::chrono::steady_clock::now();
#endif
        timer->active = true;
    });
}

ttl_status_t ttl_timer_end(
    ttl_timer_t *timer, ttl_stream_t *stream, double *milliseconds) {
    if (timer == nullptr || stream == nullptr || milliseconds == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "timer, stream, and milliseconds output are required");
    }
    *milliseconds = 0.0;
    if (!timer->active) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "timer interval is not active");
    }
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        if (timer->device_index != stream->value.device_index()) {
            throw std::invalid_argument("timer and stream devices do not match");
        }
        timer->end_event.record(stream->value);
        timer->end_event.synchronize();
        *milliseconds = static_cast<double>(
            timer->begin_event.elapsed_time(timer->end_event));
#else
        if (timer->device_index != 0 ||
            stream->value != at::mps::getCurrentMPSStream()) {
            throw std::invalid_argument("timer and stream devices do not match");
        }
        stream->value->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        *milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timer->begin_time).count();
#endif
        timer->active = false;
    });
}

void ttl_timer_destroy(ttl_timer_t *timer) { delete timer; }

ttl_status_t ttl_module_load(const char *path, ttl_module_t **module) {
    if (path == nullptr || module == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "path and module output are required");
    }
    *module = nullptr;
    return protect([&] {
        const std::filesystem::path directory(path);
        if (!std::filesystem::is_directory(directory)) {
            throw std::invalid_argument(
                "TTL module path must name a module directory");
        }
        module_spec spec = parse_manifest(directory / "module.json");
        const std::filesystem::path artifact(spec.artifact_file);
        if (artifact.is_absolute() || artifact.has_parent_path()) {
            throw std::invalid_argument(
                "module artifact file must be a simple relative filename");
        }
        if (!std::filesystem::is_regular_file(directory / artifact)) {
            throw std::invalid_argument("module artifact is missing");
        }
        *module = new ttl_module(directory, std::move(spec));
    });
}

void ttl_module_destroy(ttl_module_t *module) { delete module; }

ttl_status_t ttl_module_required_capabilities(
    const ttl_module_t *module, ttl_capability_set_t *capabilities) {
    if (module == nullptr || capabilities == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "module and capabilities output are required");
    }
    return protect([&] {
        *capabilities = required_capabilities(module->spec);
    });
}

ttl_status_t ttl_kernel_prepare(
    const ttl_module_t *module,
    const ttl_device_t *device,
    ttl_kernel_t **kernel) {
    if (module == nullptr || device == nullptr || kernel == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "module, device, and kernel output are required");
    }
    *kernel = nullptr;
    const ttl_capability_set_t missing =
        required_capabilities(module->spec) & ~backend_capabilities();
    if (missing != 0) {
        return fail(
            TTL_STATUS_UNSUPPORTED,
            "module requires unavailable device capabilities: " +
                capability_names(missing));
    }
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        c10::cuda::CUDAGuard guard(device->value);
        check_cuda(cuInit(0), "cuInit");
        CUmodule native_module = nullptr;
        const std::string artifact =
            (module->directory / module->spec.artifact_file).string();
        check_cuda(cuModuleLoad(&native_module, artifact.c_str()), "cuModuleLoad");
        try {
            CUfunction function = nullptr;
            check_cuda(
                cuModuleGetFunction(
                    &function, native_module, module->spec.kernel_name.c_str()),
                "cuModuleGetFunction");
            std::vector<bool> needs_device_scalar(module->spec.scalars.size(), false);
            for (const auto &argument : module->spec.arguments) {
                if (argument.kind == argument_kind::scalar_device) {
                    needs_device_scalar[argument.index] = true;
                }
            }
            std::vector<at::Tensor> scalar_storage(module->spec.scalars.size());
            for (size_t i = 0; i < needs_device_scalar.size(); ++i) {
                if (!needs_device_scalar[i]) continue;
                scalar_storage[i] = at::empty(
                    {4}, at::TensorOptions().dtype(at::kByte).device(device->value));
            }
            *kernel = new ttl_kernel(
                module->spec, native_module, function, device->value.index(),
                std::move(scalar_storage));
            native_module = nullptr;
        } catch (...) {
            if (native_module != nullptr) cuModuleUnload(native_module);
            throw;
        }
#else
        if (!device->value.is_mps() || device->value.index() != 0) {
            throw std::invalid_argument("prepared device is not the MPS accelerator");
        }
        const std::string artifact =
            (module->directory / module->spec.artifact_file).string();
        auto library = std::make_unique<
            at::native::mps::PrecompiledMetalShaderLibrary>(artifact);
        auto function = library->getKernelFunction(module->spec.kernel_name);
        std::vector<bool> needs_device_scalar(
            module->spec.scalars.size(), false);
        for (const auto &argument : module->spec.arguments) {
            if (argument.kind == argument_kind::scalar_device) {
                needs_device_scalar[argument.index] = true;
            }
        }
        std::vector<at::Tensor> scalar_storage(module->spec.scalars.size());
        for (size_t i = 0; i < needs_device_scalar.size(); ++i) {
            if (needs_device_scalar[i]) {
                scalar_storage[i] = at::empty(
                    {4}, at::TensorOptions().dtype(at::kByte).device(device->value));
            }
        }
        *kernel = new ttl_kernel(
            module->spec, std::move(library), std::move(function), 0,
            std::move(scalar_storage));
#endif
    });
}

void ttl_kernel_destroy(ttl_kernel_t *kernel) { delete kernel; }

ttl_status_t ttl_kernel_launch(
    ttl_kernel_t *kernel,
    ttl_tensor_t *const *tensors,
    size_t tensor_count,
    const ttl_scalar_t *scalars,
    size_t scalar_count,
    ttl_stream_t *stream) {
    if (kernel == nullptr || stream == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "kernel and stream are required");
    }
    const auto &specification = kernel->spec;
    if (tensor_count != specification.tensors.size() ||
        scalar_count != specification.scalars.size()) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "launch argument count does not match module manifest");
    }
    if ((tensor_count != 0 && tensors == nullptr) ||
        (scalar_count != 0 && scalars == nullptr)) {
        return fail(TTL_STATUS_INVALID_ARGUMENT, "launch argument arrays are NULL");
    }
    return protect([&] {
#if defined(TTL_LIBTORCH_CUDA)
        if (stream->value.device_index() != kernel->device_index) {
            throw std::invalid_argument("stream and prepared kernel use different devices");
        }
        c10::cuda::CUDAGuard guard(
            c10::Device(c10::DeviceType::CUDA, kernel->device_index));
        std::vector<void *> tensor_data(tensor_count);
#else
        if (stream->value != at::mps::getCurrentMPSStream() ||
            kernel->device_index != 0) {
            throw std::invalid_argument(
                "stream and prepared kernel use different devices");
        }
#endif
        for (size_t i = 0; i < tensor_count; ++i) {
            if (tensors[i] == nullptr) {
                throw std::invalid_argument("launch tensor is NULL");
            }
            const auto &tensor = *tensors[i];
            const auto &spec = specification.tensors[i];
#if defined(TTL_LIBTORCH_CUDA)
            if (!tensor.storage.is_cuda() ||
                tensor.storage.get_device() != kernel->device_index) {
#else
            if (!tensor.storage.is_mps() || tensor.storage.get_device() != 0) {
#endif
                throw std::invalid_argument(
                    "launch tensors must use the prepared accelerator");
            }
            if (tensor.dtype != spec.dtype || tensor.shape.size() != spec.shape.size()) {
                throw std::invalid_argument(
                    "launch tensor dtype or rank does not match module manifest");
            }
            for (size_t dimension = 0; dimension < spec.shape.size(); ++dimension) {
                if (spec.shape[dimension] != -1 &&
                    tensor.shape[dimension] != spec.shape[dimension]) {
                    throw std::invalid_argument(
                        "launch tensor shape does not match module manifest");
                }
            }
            if (spec.contiguous && !tensor.storage.is_contiguous()) {
                throw std::invalid_argument(
                    "launch tensor must be contiguous");
            }
#if defined(TTL_LIBTORCH_CUDA)
            tensor_data[i] = tensor.storage.mutable_data_ptr();
#endif
        }
        for (size_t i = 0; i < scalar_count; ++i) {
            if (scalars[i].dtype != specification.scalars[i].dtype) {
                throw std::invalid_argument(
                    "launch scalar dtype does not match module manifest");
            }
        }

#if defined(TTL_LIBTORCH_CUDA)
        const CUstream native_stream =
            reinterpret_cast<CUstream>(stream->value.stream());
        const bool has_device_scalar = std::any_of(
            specification.arguments.begin(), specification.arguments.end(),
            [](const argument_spec &argument) {
                return argument.kind == argument_kind::scalar_device;
            });
        std::unique_lock<std::mutex> staging_lock(
            kernel->scalar_staging_mutex, std::defer_lock);
        if (has_device_scalar) {
            // Device-scalar storage belongs to the prepared kernel. Order its
            // reuse after the preceding launch, even when callers alternate
            // streams. A device fill also avoids an asynchronous host-buffer
            // lifetime.
            staging_lock.lock();
            if (kernel->scalar_staging_active) {
                kernel->scalar_staging_event.block(stream->value);
            }
        }
        for (const auto &argument : specification.arguments) {
            if (argument.kind != argument_kind::scalar_device) continue;
            const size_t index = argument.index;
            unsigned int value = 0;
            if (scalars[index].dtype == TTL_DTYPE_INT32) {
                std::memcpy(&value, &scalars[index].value.int32, sizeof(value));
            } else {
                std::memcpy(&value, &scalars[index].value.float32, sizeof(value));
            }
            check_cuda(
                cuMemsetD32Async(
                    reinterpret_cast<CUdeviceptr>(
                        kernel->scalar_storage[index].mutable_data_ptr()),
                    value, 1, native_stream),
                "cuMemsetD32Async");
        }

        std::vector<std::array<unsigned char, 8>> values(
            specification.arguments.size());
        std::vector<void *> parameters(specification.arguments.size());
        for (size_t i = 0; i < specification.arguments.size(); ++i) {
            const auto &argument = specification.arguments[i];
            void *pointer = nullptr;
            if (argument.kind == argument_kind::tensor) {
                pointer = tensor_data[argument.index];
                std::memcpy(values[i].data(), &pointer, sizeof(pointer));
            } else if (argument.kind == argument_kind::scalar_device) {
                pointer = kernel->scalar_storage[argument.index].mutable_data_ptr();
                std::memcpy(values[i].data(), &pointer, sizeof(pointer));
            } else if (argument.kind == argument_kind::scalar_value) {
                const auto &scalar = scalars[argument.index];
                if (scalar.dtype == TTL_DTYPE_INT32) {
                    std::memcpy(values[i].data(), &scalar.value.int32, sizeof(int32_t));
                } else {
                    std::memcpy(values[i].data(), &scalar.value.float32, sizeof(float));
                }
            } else if (argument.dtype == TTL_DTYPE_INT32) {
                std::memcpy(values[i].data(), &argument.int32_value, sizeof(int32_t));
            } else {
                std::memcpy(values[i].data(), &argument.float32_value, sizeof(float));
            }
            parameters[i] = values[i].data();
        }
        check_cuda(
            cuLaunchKernel(
                kernel->function,
                specification.grid[0], specification.grid[1], specification.grid[2],
                specification.block[0], specification.block[1], specification.block[2],
                specification.shared_bytes, native_stream,
                parameters.data(), nullptr),
            "cuLaunchKernel");
        if (has_device_scalar) {
            kernel->scalar_staging_event.record(stream->value);
            kernel->scalar_staging_active = true;
        }
#else
        for (const auto &argument : specification.arguments) {
            if (argument.kind != argument_kind::scalar_device) continue;
            const size_t index = argument.index;
            std::array<unsigned char, 4> value{};
            if (scalars[index].dtype == TTL_DTYPE_INT32) {
                std::memcpy(value.data(), &scalars[index].value.int32, 4);
            } else {
                std::memcpy(value.data(), &scalars[index].value.float32, 4);
            }
            auto host = at::from_blob(
                value.data(), {4},
                at::TensorOptions().dtype(at::kByte).device(at::kCPU));
            kernel->scalar_storage[index].copy_(host, false);
        }

        kernel->function->runCommandBlock([&] {
            kernel->function->startEncoding();
            for (size_t i = 0; i < specification.arguments.size(); ++i) {
                const auto &argument = specification.arguments[i];
                if (argument.kind == argument_kind::tensor) {
                    kernel->function->setArg(
                        static_cast<unsigned int>(i),
                        tensors[argument.index]->storage);
                } else if (argument.kind == argument_kind::scalar_device) {
                    kernel->function->setArg(
                        static_cast<unsigned int>(i),
                        kernel->scalar_storage[argument.index]);
                } else if (argument.kind == argument_kind::scalar_value) {
                    const auto &scalar = scalars[argument.index];
                    if (scalar.dtype == TTL_DTYPE_INT32) {
                        kernel->function->setArg(
                            static_cast<unsigned int>(i), scalar.value.int32);
                    } else {
                        kernel->function->setArg(
                            static_cast<unsigned int>(i), scalar.value.float32);
                    }
                } else if (argument.dtype == TTL_DTYPE_INT32) {
                    kernel->function->setArg(
                        static_cast<unsigned int>(i), argument.int32_value);
                } else {
                    kernel->function->setArg(
                        static_cast<unsigned int>(i), argument.float32_value);
                }
            }
            std::array<uint64_t, 3> threads{};
            std::array<uint64_t, 3> group{};
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                threads[dimension] =
                    static_cast<uint64_t>(specification.grid[dimension]) *
                    specification.block[dimension];
                group[dimension] = specification.block[dimension];
            }
            kernel->function->dispatch(
                c10::ArrayRef<uint64_t>(threads),
                c10::OptionalArrayRef<uint64_t>(group));
        });
#endif
    });
}

ttl_status_t ttl_program_load(const char *path, ttl_program_t **program) {
    if (path == nullptr || program == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "path and program output are required");
    }
    *program = nullptr;
    return protect([&] {
        const std::filesystem::path directory(path);
        if (!std::filesystem::is_directory(directory)) {
            throw std::invalid_argument(
                "TTL program path must name a program directory");
        }
        program_spec spec = parse_program(directory / "program.json");
        *program = new ttl_program(directory, std::move(spec));
    });
}

void ttl_program_destroy(ttl_program_t *program) { delete program; }

ttl_status_t ttl_program_prepare(
    const ttl_program_t *program,
    const ttl_device_t *device,
    const ttl_program_binding_t *bindings,
    size_t binding_count,
    ttl_program_execution_t **execution) {
    if (program == nullptr || device == nullptr || execution == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "program, device, and execution output are required");
    }
    const size_t expected_bindings =
        program->spec.inputs.size() + program->spec.outputs.size();
    if (binding_count != expected_bindings ||
        (binding_count != 0 && bindings == nullptr)) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "program binding count does not match inputs and outputs");
    }
    *execution = nullptr;
    return protect([&] {
        auto prepared = std::make_unique<ttl_program_execution>();
        std::map<std::string, ttl_tensor_t *> bound;
        for (size_t i = 0; i < binding_count; ++i) {
            if (bindings[i].name == nullptr || bindings[i].tensor == nullptr) {
                throw std::invalid_argument(
                    "program binding names and tensors are required");
            }
            if (!bound.emplace(bindings[i].name, bindings[i].tensor).second) {
                throw std::invalid_argument(
                    "program binding names must be unique");
            }
        }

        auto validate_tensor = [&](const ttl_tensor_t *tensor,
                                   ttl_dtype_t dtype,
                                   const std::vector<int64_t> &shape,
                                   const std::string &name) {
            if (tensor->dtype != dtype || tensor->shape != shape) {
                throw std::invalid_argument(
                    "program tensor type does not match for " + name);
            }
#if defined(TTL_LIBTORCH_CUDA)
            if (!tensor->storage.is_cuda() ||
                tensor->storage.get_device() != device->value.index()) {
#else
            if (!tensor->storage.is_mps() ||
                tensor->storage.get_device() != device->value.index()) {
#endif
                throw std::invalid_argument(
                    "program tensors must use the prepared device");
            }
        };

        std::map<std::string, const program_tensor_spec *> external_specs;
        for (const auto &spec : program->spec.inputs) {
            external_specs.emplace(spec.name, &spec);
        }
        for (const auto &spec : program->spec.outputs) {
            external_specs.emplace(spec.name, &spec);
        }
        for (const auto &[name, tensor] : bound) {
            const auto found = external_specs.find(name);
            if (found == external_specs.end()) {
                throw std::invalid_argument(
                    "unknown program tensor binding: " + name);
            }
            validate_tensor(
                tensor, found->second->dtype, found->second->shape, name);
        }
        for (const auto &[name, spec] : external_specs) {
            (void)spec;
            if (bound.count(name) == 0) {
                throw std::invalid_argument(
                    "missing program tensor binding: " + name);
            }
        }

        std::map<std::string, ttl_tensor_t *> values;
        std::map<std::string, size_t> producers;
        for (const auto &input : program->spec.inputs) {
            values.emplace(input.name, bound.at(input.name));
        }
        std::map<std::string, ttl_tensor_t *> program_outputs;
        for (const auto &output : program->spec.outputs) {
            program_outputs.emplace(output.name, bound.at(output.name));
        }

        struct prepared_function {
            ttl_module_t *module;
            ttl_kernel_t *kernel;
        };
        std::map<std::string, prepared_function> functions;

        for (const auto &call : program->spec.calls) {
            auto function = functions.find(call.function);
            if (function == functions.end()) {
                const std::filesystem::path path =
                    program->directory / "functions" / call.function;
                ttl_module_t *module = nullptr;
                ttl_status_t status = ttl_module_load(
                    path.string().c_str(), &module);
                if (status != TTL_STATUS_OK) {
                    throw std::invalid_argument(
                        "cannot load program function " + call.function +
                        ": " + last_error);
                }
                prepared->modules.push_back(module);
                ttl_kernel_t *kernel = nullptr;
                status = ttl_kernel_prepare(module, device, &kernel);
                if (status != TTL_STATUS_OK) {
                    throw std::invalid_argument(
                        "cannot prepare program function " + call.function +
                        ": " + last_error);
                }
                prepared->kernels.push_back(kernel);
                function = functions.emplace(
                    call.function, prepared_function{module, kernel}).first;
            }

            const module_spec &module = function->second.module->spec;
            if (!module.scalars.empty()) {
                throw std::invalid_argument(
                    "program functions cannot expose scalar arguments");
            }
            size_t input_index = 0;
            size_t output_index = 0;
            program_call_execution prepared_call;
            prepared_call.kernel = function->second.kernel;
            prepared_call.tensors.reserve(module.tensors.size());
            for (const auto &name : call.arguments) {
                const auto producer = producers.find(name);
                if (producer != producers.end() &&
                    std::find(
                        prepared_call.dependencies.begin(),
                        prepared_call.dependencies.end(), producer->second) ==
                        prepared_call.dependencies.end()) {
                    prepared_call.dependencies.push_back(producer->second);
                }
            }
            for (const auto &tensor_specification : module.tensors) {
                if (tensor_specification.access == tensor_access::read_write) {
                    throw std::invalid_argument(
                        "program functions must be purely read or write");
                }
                ttl_tensor_t *tensor = nullptr;
                if (tensor_specification.access == tensor_access::read) {
                    if (input_index == call.arguments.size()) {
                        throw std::invalid_argument(
                            "program call has too few arguments");
                    }
                    const std::string &name = call.arguments[input_index++];
                    const auto value = values.find(name);
                    if (value == values.end()) {
                        throw std::invalid_argument(
                            "program call argument is undefined: " + name);
                    }
                    tensor = value->second;
                    validate_tensor(
                        tensor, tensor_specification.dtype,
                        tensor_specification.shape, name);
                } else {
                    if (output_index == call.results.size()) {
                        throw std::invalid_argument(
                            "program call has too few results");
                    }
                    const std::string &name = call.results[output_index++];
                    const auto external = program_outputs.find(name);
                    if (external != program_outputs.end()) {
                        tensor = external->second;
                    } else {
                        for (int64_t extent : tensor_specification.shape) {
                            if (extent < 1) {
                                throw std::invalid_argument(
                                    "program functions require static shapes");
                            }
                        }
                        tensor = allocate_tensor(
                            device->value, tensor_specification.dtype,
                            tensor_specification.shape);
                        prepared->intermediates.push_back(tensor);
                    }
                    validate_tensor(
                        tensor, tensor_specification.dtype,
                        tensor_specification.shape, name);
                    if (!values.emplace(name, tensor).second) {
                        throw std::invalid_argument(
                            "program result is not a fresh value: " + name);
                    }
                    producers.emplace(name, prepared->calls.size());
                }
                prepared_call.tensors.push_back(tensor);
            }
            if (input_index != call.arguments.size() ||
                output_index != call.results.size()) {
                throw std::invalid_argument(
                    "program call arity does not match its function");
            }
            prepared->calls.push_back(std::move(prepared_call));
        }
        for (const auto &output : program->spec.outputs) {
            if (values.count(output.name) == 0) {
                throw std::invalid_argument(
                    "program output is not produced: " + output.name);
            }
        }
#if defined(TTL_LIBTORCH_CUDA)
        c10::cuda::CUDAGuard guard(device->value);
        check_cuda(cuInit(0), "cuInit");
        prepared->device_index = device->value.index();
        check_cuda(cuGraphCreate(&prepared->native_graph, 0),
                   "cuGraphCreate");
        for (auto &call : prepared->calls) {
            std::vector<CUgraphNode> dependencies;
            dependencies.reserve(call.dependencies.size());
            for (size_t dependency : call.dependencies) {
                dependencies.push_back(
                    prepared->calls.at(dependency).native_node);
            }
            const module_spec &specification = call.kernel->spec;
            call.argument_values.resize(specification.arguments.size());
            call.native_arguments.resize(specification.arguments.size());
            for (size_t i = 0; i < specification.arguments.size(); ++i) {
                const auto &argument = specification.arguments[i];
                if (argument.kind == argument_kind::tensor) {
                    void *pointer = call.tensors.at(argument.index)
                                        ->storage.mutable_data_ptr();
                    std::memcpy(
                        call.argument_values[i].data(), &pointer,
                        sizeof(pointer));
                } else if (argument.kind == argument_kind::literal &&
                           argument.dtype == TTL_DTYPE_INT32) {
                    std::memcpy(
                        call.argument_values[i].data(),
                        &argument.int32_value, sizeof(argument.int32_value));
                } else if (argument.kind == argument_kind::literal) {
                    std::memcpy(
                        call.argument_values[i].data(),
                        &argument.float32_value,
                        sizeof(argument.float32_value));
                } else {
                    throw std::invalid_argument(
                        "program functions cannot expose runtime scalars");
                }
                call.native_arguments[i] = call.argument_values[i].data();
            }
            CUDA_KERNEL_NODE_PARAMS parameters{};
            parameters.func = call.kernel->function;
            parameters.gridDimX = specification.grid[0];
            parameters.gridDimY = specification.grid[1];
            parameters.gridDimZ = specification.grid[2];
            parameters.blockDimX = specification.block[0];
            parameters.blockDimY = specification.block[1];
            parameters.blockDimZ = specification.block[2];
            parameters.sharedMemBytes = specification.shared_bytes;
            parameters.kernelParams = call.native_arguments.data();
            check_cuda(
                cuGraphAddKernelNode(
                    &call.native_node, prepared->native_graph,
                    dependencies.data(), dependencies.size(), &parameters),
                "cuGraphAddKernelNode");
        }
        check_cuda(
            cuGraphInstantiate(
                &prepared->native_execution, prepared->native_graph, 0),
            "cuGraphInstantiate");
#endif
        *execution = prepared.release();
    });
}

void ttl_program_execution_destroy(ttl_program_execution_t *execution) {
    delete execution;
}

ttl_status_t ttl_program_launch(
    ttl_program_execution_t *execution, ttl_stream_t *stream) {
    if (execution == nullptr || stream == nullptr) {
        return fail(TTL_STATUS_INVALID_ARGUMENT,
                    "program execution and stream are required");
    }
#if defined(TTL_LIBTORCH_CUDA)
    return protect([&] {
        if (stream->value.device_index() != execution->device_index) {
            throw std::invalid_argument(
                "stream and prepared program use different devices");
        }
        c10::cuda::CUDAGuard guard(
            c10::Device(c10::DeviceType::CUDA, execution->device_index));
        const CUstream native_stream =
            reinterpret_cast<CUstream>(stream->value.stream());
        check_cuda(
            cuGraphLaunch(execution->native_execution, native_stream),
            "cuGraphLaunch");
    });
#else
    for (auto &call : execution->calls) {
        const ttl_status_t status = ttl_kernel_launch(
            call.kernel, call.tensors.data(), call.tensors.size(),
            nullptr, 0, stream);
        if (status != TTL_STATUS_OK) return status;
    }
    return TTL_STATUS_OK;
#endif
}

}  // extern "C"
