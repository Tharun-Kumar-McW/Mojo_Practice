# Mojo Custom Operations Guide: GPU/CPU with MAX Engine

A comprehensive guide to creating and executing custom operations on GPU/CPU using Mojo and the MAX Engine framework.

---

## Table of Contents

1. [Creating and Calling Functions on GPU/CPU](#creating-and-calling-functions-on-gpucpu)
2. [Basic Skeleton of Custom Ops](#basic-skeleton-of-custom-ops)
3. [Basic Implementation of Custom Ops](#basic-implementation-of-custom-ops)
4. [Dispatcher Flow for Custom Ops](#dispatcher-flow-for-custom-ops)
5. [Built-in vs Custom Kernel Implementation](#built-in-vs-custom-kernel-implementation)

---

## Creating and Calling Functions on GPU/CPU

### Overview

The MAX Engine provides a flexible framework to create custom operations that automatically dispatch to GPU or CPU based on device availability. This is achieved through:

1. **Graph Definition** - Define computation as a graph using `ops.custom()`
2. **Custom Extensions** - Load Mojo kernel implementations from external files
3. **Device-Aware Dispatch** - Automatic selection of CPU or GPU execution
4. **Buffer Management** - Handle data transfer between host and device

### Device Selection Pattern

```python
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Device selection
device = CPU() if accelerator_count() == 0 else Accelerator()

# Create graph on selected device
graph = Graph(
    "my_operation",
    forward=lambda x, y: ops.custom(
        name="my_custom_kernel",
        device=DeviceRef.from_device(device),
        values=[x, y],
        out_types=[TensorType(..., device=DeviceRef.from_device(device))],
    )[0].tensor,
    input_types=[...],
    custom_extensions=[Path("./kernels")],
)

# Load and execute
session = InferenceSession(devices=[device])
model = session.load(graph)
result = model.execute(input_buffer)[0]
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `DeviceRef.from_device()` | Convert device object to device reference |
| `ops.custom()` | Declare custom operation in computation graph |
| `custom_extensions` | Path to Mojo kernel implementations |
| `InferenceSession` | Manages execution on specified devices |
| `Buffer` management | Transfer data between numpy and device |

---

## Basic Skeleton of Custom Ops

### Kernel Structure

A custom operation requires:

1. **Kernel Function Definition** - Implements the actual computation
2. **Kernel Registration** - Registers with `@compiler.register()`
3. **Device Dispatcher** - Routes to CPU/GPU implementation

### Minimal Example

```mojo
# File: kernels/__init__.mojo
import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

# Import device-specific implementations
from .cpu.my_kernel_cpu import _my_kernel_cpu
from .gpu.my_kernel_gpu import _my_kernel_gpu

@compiler.register("my-custom-kernel")
struct MyCustomKernel:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        input: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "cpu":
            _my_kernel_cpu(output, input)
        elif target == "gpu":
            _my_kernel_gpu(output, input, ctx)
        else:
            raise Error("Unknown target: ", target)
```

### Kernel Organization

```
kernels/
├── __init__.mojo                 # Dispatcher & registration
├── cpu/
│   ├── __init__.mojo
│   └── my_kernel_cpu.mojo        # CPU implementation
└── gpu/
    ├── __init__.mojo
    └── my_kernel_gpu.mojo        # GPU implementation
```

---

## Basic Implementation of Custom Ops

### CPU Implementation Pattern

CPU kernels are optimized for serial or threaded execution with focus on cache efficiency:

```mojo
# File: kernels/cpu/my_kernel_cpu.mojo
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

def _my_kernel_cpu(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    input: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    """
    CPU implementation optimizes for cache locality and memory access patterns.
    """
    var rows = input.dim_size(0)
    var cols = input.dim_size(1)
    
    # Outer loop: iterate over output rows
    for i in range(rows):
        # Inner loop: process columns sequentially (cache-friendly)
        for j in range(cols):
            var val = input.load[2](IndexList[2](i, j))[0]
            # Apply computation
            output.store[2](IndexList[2](i, j), val * 2.0)
```

#### Optimization Techniques for CPU

1. **Loop Reordering** - Arrange loops for spatial locality
   ```mojo
   # ❌ Poor: Repeated loads of outer dimension
   for k in outer_dim:
       for j in seq_len:
           val += data[j][k]
   
   # ✅ Good: Sequential access pattern
   for j in seq_len:
       score = scores[j]
       for k in outer_dim:
           val += score * data[j][k]
   ```

2. **Tiling Strategy** - Process data in cache-friendly blocks
   ```mojo
   let TILE_SIZE = 32
   for i in range(rows, step=TILE_SIZE):
       for j in range(cols, step=TILE_SIZE):
           # Process TILE_SIZE × TILE_SIZE block
           process_tile(input, output, i, j, TILE_SIZE)
   ```

3. **Data Type Management** - Use FP32 for computation, store in FP16
   ```mojo
   # Accumulate in higher precision, store in lower
   var acc: Float32 = 0.0
   for k in range(dim):
       var inp = Float32(input.load[2](idx)[0])
       acc += inp
   output.store[2](idx, Float16(acc))
   ```

### GPU Implementation Pattern

GPU kernels distribute work across blocks and threads:

```mojo
# File: kernels/gpu/my_kernel_gpu.mojo
from std.math import exp, sqrt
from std.gpu import block_dim, block_idx, thread_idx
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

def _my_kernel_gpu(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    input: InputTensor[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
) raises:
    """
    GPU implementation distributes computation across thread blocks.
    """
    comptime BLOCK_SIZE = 64  # Threads per block
    
    var rows = input.dim_size(0)
    var cols = input.dim_size(1)
    var gpu_ctx = ctx.get_device_context()
    
    @parameter
    fn _kernel(rows_: Int, cols_: Int) capturing -> None:
        # Each thread handles one element
        var i = Int(block_idx.x)      # Row index (block)
        var j = Int(thread_idx.x)     # Column index (thread)
        
        if i >= rows_ or j >= cols_:
            return
        
        # Perform computation
        var val = input.load[2](IndexList[2](i, j))[0]
        output.store[2](IndexList[2](i, j), val * 2.0)
    
    # Launch kernel: rows blocks, BLOCK_SIZE threads per block
    gpu_ctx.launch_kernel[_kernel, rows, BLOCK_SIZE](rows, cols)
```

#### GPU-Specific Considerations

| Aspect | Pattern |
|--------|---------|
| **Work Distribution** | `block_idx` for coarse granularity, `thread_idx` for fine granularity |
| **Block Size** | Typically 64-256 threads (power of 2) |
| **Global Barriers** | Use `gpu_ctx.launch_kernel()` to synchronize work |
| **Shared Memory** | Store frequently accessed data in block-local cache |
| **Coalescing** | Access global memory sequentially from consecutive threads |

---

## Dispatcher Flow for Custom Ops

### Compilation Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Python Code: Define Graph with ops.custom()                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  ops.custom(name, device, values, out_types)                │
│  • Specifies operation name: "my-custom-kernel"              │
│  • Specifies target device: CPU/GPU (DeviceRef)              │
│  • Lists input/output tensors                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MAX Engine Resolution Phase                                 │
│  • Searches custom_extensions paths                          │
│  • Finds @compiler.register("my-custom-kernel") struct      │
│  • Registers kernel with runtime                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Execution Phase: model.execute(input_buffer)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
     ╔═══════════════╩════════════════╗
     │                                │
     ▼                                ▼
┌──────────────────────┐    ┌────────────────────┐
│  Target == "cpu"     │    │  Target == "gpu"   │
│                      │    │                    │
│  execute[cpu](...)   │    │  execute[gpu](...) │
│  calls:              │    │  calls:            │
│  _my_kernel_cpu()    │    │  _my_kernel_gpu()  │
└──────────────────────┘    └────────────────────┘
     │                                │
     ▼                                ▼
┌──────────────────────┐    ┌────────────────────┐
│  Serial Computation  │    │  GPU Kernel Launch │
│  Loop over rows      │    │  Blocks × Threads  │
│  Process elements    │    │  Parallel Exec     │
└──────────────────────┘    └────────────────────┘
```

### Dispatcher Implementation Details

The dispatcher operates through **compile-time conditionals** (comptime if):

```mojo
@compiler.register("sda-custom-ops")
struct SdaCustomOps:
    @staticmethod
    def execute[target: StaticString, flag: Int](
        output: OutputTensor[dtype=DType.float16, rank=2, ...],
        Q: InputTensor[dtype=output.dtype, rank=2, ...],
        K: InputTensor[dtype=output.dtype, rank=2, ...],
        V: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        # Compile-time dispatch: Only target code is compiled
        comptime if target == "cpu":
            @parameter
            if flag == 0:
                _sda_cpu_naive(output, Q, K, V)
            elif flag == 1:
                _sda_cpu_loop_reordered(output, Q, K, V)
            elif flag == 2:
                _sda_cpu_loop_tiling(output, Q, K, V)
        elif target == "gpu":
            _sda_gpu_naive(output, Q, K, V, ctx)
        else:
            raise Error("Unknown target: ", target)
```

### Dispatching by Parameters

In addition to device target, kernels can dispatch based on parameters:

```python
# Python: Pass parameter flag to kernel
graph = Graph(
    "attention",
    forward=lambda Q, K, V: ops.custom(
        name="sda-custom-ops",
        device=DeviceRef.from_device(device),
        values=[Q, K, V],
        out_types=[TensorType(...)],
        parameters={"flag": 1},  # ← Select implementation variant
    )[0].tensor,
    ...
)
```

The `@parameter` decorator enables compile-time branching on this flag value, selecting the appropriate kernel variant without runtime overhead.

---

## Built-in vs Custom Kernel Implementation

### Comparison Matrix

| Aspect | Built-in Kernels | Custom Kernels |
|--------|------------------|-----------------|
| **Definition** | Defined in MAX Engine core | User-defined in .mojo files |
| **Use Case** | Standard ops (matrix multiply, softmax) | Domain-specific optimizations |
| **Performance** | Highly optimized, vendor-tuned | Customizable, can exceed built-ins |
| **Compilation** | Pre-compiled, binary | Compiled on-the-fly with model |
| **Flexibility** | Fixed behavior | Full control over computation |

### Implementation Differences

#### Built-in Kernel (e.g., Matrix Multiply)

```python
# Using built-in op
graph = Graph(
    "matmul",
    forward=lambda a, b: ops.matmul(a, b),
    input_types=[...],
    # No custom_extensions needed
)
```

**Characteristics:**
- ✅ No implementation code required
- ✅ Automatically optimized for all devices
- ✅ No compilation overhead
- ❌ Cannot customize computation
- ❌ Device selection handled internally

#### Custom Kernel (e.g., Specialized Attention)

```python
# Using custom op
graph = Graph(
    "attention",
    forward=lambda Q, K, V: ops.custom(
        name="sda-custom-ops",
        device=DeviceRef.from_device(device),
        values=[Q, K, V],
        out_types=[...],
        parameters={"flag": 1},
    )[0].tensor,
    input_types=[...],
    custom_extensions=[Path("./kernels/SDPA")],  # ← Implementation provided
)
```

**Characteristics:**
- ✅ Full control over computation
- ✅ Can leverage algorithm-specific optimizations
- ✅ Parameters for variant selection
- ❌ Must provide both CPU and GPU implementations
- ❌ Responsible for correctness and performance

### File Structure Comparison

#### Built-in Operation Path
```
MAX Engine Core
└── Built-in kernels (pre-compiled)
```

#### Custom Operation Path
```
Project Directory
├── kernels/
│   ├── __init__.mojo              ← Dispatcher
│   ├── cpu/
│   │   ├── __init__.mojo
│   │   └── impl.mojo              ← CPU kernels
│   └── gpu/
│       ├── __init__.mojo
│       └── impl.mojo              ← GPU kernels
└── Python script
    └── custom_extensions=[Path(...)]  ← Referenced here
```

### Example: From Built-in to Custom

#### Start with Built-in (Baseline)
```python
# Standard softmax - works but not optimized
graph = Graph(
    "softmax_baseline",
    forward=lambda x: ops.softmax(x, axis=1),
    input_types=[TensorType(DType.float32, shape=[seq_len], device=device)],
)
```

#### Convert to Custom (Optimized)
```python
# Custom softmax with numerical stability tricks
graph = Graph(
    "softmax_optimized",
    forward=lambda x: ops.custom(
        name="softmax-stable",
        device=DeviceRef.from_device(device),
        values=[x],
        out_types=[TensorType(DType.float32, shape=[seq_len], device=device)],
        parameters={"use_log": 1},
    )[0].tensor,
    input_types=[TensorType(DType.float32, shape=[seq_len], device=device)],
    custom_extensions=[Path("./kernels")],
)
```

Then implement:
- `kernels/cpu/softmax_cpu.mojo` - CPU version with loop reordering
- `kernels/gpu/softmax_gpu.mojo` - GPU version with shared memory tiling

---

## Practical Example: Scaled Dot Product Attention (SDPA)

### Complete Implementation Overview

This example implements SDPA with:
- ✅ CPU baseline and optimizations
- ✅ GPU implementation with thread blocks
- ✅ Dispatcher with multiple variants
- ✅ Parameter-based kernel selection

### File Organization

```
kernels/SDPA/
├── __init__.mojo           # Dispatcher
├── sda.mojo                # Kernel registration
├── cpu/
│   ├── __init__.mojo
│   └── sda_cpu.mojo        # CPU: naive, loop-reordered, tiling
└── gpu/
    ├── __init__.mojo
    └── sda_gpu.mojo        # GPU: naive, optimized
```

### Key SDPA Computation

Attention = softmax(Q·K^T / √d_k) · V

### CPU Optimizations

**Naive (flag=0):**
```
for i in seq:
    for j in seq:
        for k in d_k:
            scores[j] += Q[i][k] * K[j][k]
    softmax(scores)
    for k in d_k:
        for j in seq:
            out[k] += scores[j] * V[j][k]  # ❌ Poor cache
```

**Loop-Reordered (flag=1):**
```
for i in seq:
    for j in seq:
        for k in d_k:
            scores[j] += Q[i][k] * K[j][k]
    softmax(scores)
    for j in seq:
        score = scores[j]
        for k in d_k:
            out[k] += score * V[j][k]  # ✅ Sequential access
```

**Result:** ~1.24x speedup through better cache locality

### GPU Implementation

- **Blocks:** One per query token (seq_len blocks)
- **Threads:** One per dimension (d_k threads per block)
- **Computation:** Each thread accumulates weighted values

```mojo
var i = block_idx.x      # Query token (0 to seq_len-1)
var k = thread_idx.x     # Output dimension (0 to d_k-1)
# Thread (i, k) computes attention output for position i, dimension k
```

---

## Best Practices

### 1. Correctness First
```mojo
# Always verify against reference implementation
result = model.execute(input_buffer)[0]
expected = reference_implementation(input_numpy)
assert np.allclose(result, expected, atol=1e-4)
```

### 2. Performance Profiling
```python
import time

# Warmup
_ = model.execute(buffer)

# Measure
start = time.perf_counter()
for _ in range(100):
    result = model.execute(buffer)
elapsed = (time.perf_counter() - start) / 100
print(f"Time per iteration: {elapsed*1e3:.2f}ms")
```

### 3. Device Abstraction
```python
# Support both CPU and GPU without code duplication
device = CPU() if accelerator_count() == 0 else Accelerator()
# Same graph definition works for both devices
```

### 4. Numerical Stability
```mojo
# Use FP32 for accumulation, store results in FP16
var acc: Float32 = 0.0
for i in range(n):
    acc += Float32(input[i])  # Higher precision computation
output[j] = Float16(acc)      # Lower precision storage
```

### 5. Memory Efficiency
```mojo
# Pre-compute and cache values
var scale: Float32 = 1.0 / sqrt(Float32(d_k))  # Compute once
for i in range(seq_len):
    for j in range(seq_len):
        scores[j] = dot_product(Q[i], K[j]) * scale  # Use cached scale
```

---

## Troubleshooting

### Issue: Kernel Not Found
**Symptom:** `Unknown kernel name`
**Solution:** 
- Check `@compiler.register()` name matches `ops.custom(name=...)`
- Verify `custom_extensions` path includes the .mojo file
- Ensure `__init__.mojo` files exist in directory structure

### Issue: Type Mismatch
**Symptom:** `Cannot convert DType.float16 to float32`
**Solution:**
- Explicitly cast: `Float32(input.load[2](idx)[0])`
- Keep tensor dtype consistent with computation dtype
- Match `TensorType(dtype=...)` with kernel implementation

### Issue: Wrong Device Execution
**Symptom:** Operation runs on CPU instead of GPU
**Solution:**
- Check `accelerator_count() > 0` returns true
- Verify `DeviceRef.from_device(device)` uses correct device
- Confirm `.mojo` files implement both `target=="cpu"` and `target=="gpu"`

---

## Summary

Custom operations in Mojo with MAX Engine enable:

1. **GPU/CPU Flexibility** - Write once, run on both devices
2. **Algorithmic Control** - Optimize for specific computation patterns
3. **Device Dispatch** - Automatic selection via compile-time conditionals
4. **Performance** - Achieve up to 1.24x+ speedup through optimization

Key takeaways:
- Use `@compiler.register()` to define kernel entry point
- Implement separate CPU and GPU kernel functions
- Use `ops.custom()` in Python graph to invoke kernel
- Dispatcher handles device selection automatically
- Parameters enable multiple kernel variants
- Always verify correctness against reference implementations

---

## References

- **Scaled Dot Product Attention:** `Mojo_custom_ops/`
- **Matrix Operations:** `Python_to_Mojo/matmul*.mojo`
- **Parallel Reduction:** `Paralle_reduction/basic.mojo`

