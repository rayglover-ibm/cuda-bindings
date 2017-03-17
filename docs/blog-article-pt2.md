# <em>CUDA language bindings</em> – Part 2: Microlibrary

- [`Add` Kernel](#Part2-1)
- [Running Kernels](#Part2-2)
- [Error Handling](#Part2-3)
- [Unit Tests](#Part2-4)
- [Build Options](#Part2-5)

---

The `src/kernels` folder will contain our algorithm. The term _kernel_ is used to describe a low-level building block which facilitates some higher-level algorithm. A kernel may be invoked directly by a consumer of your library, but more typically kernels are encapsulated via some higher-level public API defined by the library. There can be several kernel implementations for the same logical operation; it's the job of the library to select one based on the inputs to the operation and the capabilities of the underlying hardware.

For `mwe` our single kernel will be called `add`, which unsurprisingly just adds integers together. We'll have 2 kernel implementations, distinguished by their _compute mode_. The _compute mode_ determines where the kernel is executed, i.e. `CPU` or `CUDA`, but you could imagine others, for instance `OpenCL`, `FPGA`, `OpenMP` and so on.

<br>

## <a name="Part2-1"></a> `Add` Kernel

Other than adding numbers, we'd also like our kernel to allow us to:

- permit building `mwe` with or without a GPU present.
- permit running `mwe` with or without a GPU, and be able to control this behavior at runtime.

These two requirements are particularly useful for real-world cross-platform applications, although it's also a missing piece in a typical GPGPU tutorial. I'll spend some time describing how to accomplish this with a small utility I wrote called `kernel.h`.

<br>

### Declare the Kernel

![img](./fig-1.PNG)

With `kernel.h` we declare our kernel with a simple macro, `KERNEL_DECL` in `src/kernels/add.h`:

```c++
KERNEL_DECL(add,
    compute_mode::CPU, compute_mode::CUDA)
{};
```

Here we've declared a kernel named `add` which supports `CPU` and `CUDA` `compute_mode`'s. Currently there are no _operations_ on this kernel, so we can't invoke it yet.

An _operation_ must be a static member template named `op`. We can overload `op` on each kernel with varying arguments and return types, but for now we'll declare one operation which takes two `int`'s and returns another `int`:

```c++
KERNEL_DECL(add,
    compute_mode::CPU, compute_mode::CUDA)
{
    template <compute_mode> static int op(
        int a, int b);
};
```

Once the operation is declared, we need to implement it. Since our kernel supports `CPU` and `CUDA`, the kernel runner will expect to be able to find the respective implementation if it's enabled during compilation. For example, if we enable `CUDA` at compile time, the runner will expect to find a specialization of `add::op` for `compute_mode::CUDA`.

<br>

### Implement the kernel operation(s)

To keep things well structured, the implementations (i.e. definitions) of each operation are grouped by `compute_mode` in to separate files. In our case the definitions reside in `src/kernels/add.cpp` and `src/kernels/add.cu` for `CPU` and `CUDA` respectively.

- ### CPU

    ```c++
    template <> int add::op<compute_mode::CPU>(
        int a, int b)
    {
        return a + b;
    }
    ```

    The CPU implementation (above) is trivially simple. Note we're using the `template <>` syntax to denote an [explicit specialization](http://en.cppreference.com/w/cpp/language/template_specialization) of our `add::op` template, in this case for `compute_mode::CPU`.

- ### CUDA

    The CUDA implementation is more substantial. For such a trivial operation like `int add(int, int)`, it's highly unlikely that a GPU implementation would be even half as fast as the CPU implementation because of various unavoidable overheads. None the less we can introduce the basic (but fundamental) parts of the CUDA workflow here: memory management and kernel launching:
    
    ```c++
    namespace
    {
        __global__ void add(
            int a, int b, int* result)
        {
            *result = a + b;                       (3)
        }
    }
    
    template <> int add::op<compute_mode::CUDA>(
        int a, int b)
    {
        device_ptr<int> dev_c;                     (1)
    
        ::add<<< 1, 1 >>>(a, b, dev_c.get());      (2)
    
        int c;
        dev_c.copy_to({ &c, 1 });                  (4)
    
        return c;                                  (5)
    }
    ```

    1. Allocate a single `int` on the GPU (a.k.a. the _device_), where the result of will be written to.
    2. Launch the CUDA kernel, supplying the 2 input integers and a pointer to the output.
    3. Our device code, to run on the GPU.
    4. Copy the output from the device back to main memory.
    5. Return the result.
    
    One initial point of interest in this implementation at (1) is the `device_ptr<T>` type. Similar to `std::unique_ptr<T>`, this is a smart pointer that owns and manages an object of type `T` on the GPU, and disposes of that object when it goes out of scope. We can copy memory to and from the device with `copy_to` and `copy_from` member functions. `device_ptr` also makes it easy to allocate a chunk of memory capable of holding N elements. For example, to allocate 256 `float`'s contiguously in device memory:
    
    ```c++
    device_ptr<float> device_elements(256);
    ```
    
    Once `device_elements` goes out of scope, the device memory is freed.
    
    This isn't intended to be a comprehensive tutorial on CUDA itself. If you're interested in knowing more about say, the odd looking `::add<<< G, B >>>` syntax at (2), or what `__global__` means, you can acquaint yourself with the core concepts by reading the introductory tutorial on the NVIDIA Developer blog [here](https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/).
    
    Depending on your particular problem, it's likely you'll be able to leverage preexisting CUDA libraries like cuBLAS or _NVIDIA Performance Primitives_ (NPP). There are many open source 3rd party libraries; an incomplete list can be found [here](https://developer.nvidia.com/gpu-accelerated-libraries).

<br>

## <a name="Part2-2"></a> Running kernels

![img](./fig-2.PNG)

Now that we've fully implemented the operations on our `add` kernel for both CPU and GPU, we can wrap it in a public API to be exposed through `include/mwe.h`. Here is the declaration for `add`:

```c++
maybe<int> add(int a, int b);
```

Note the introduction of the `maybe<T>` type, which is how we will propagate potential errors back to the caller. We'll discuss error handling in more detail in a moment.

In `src/mwe.cpp`, the implementation looks like:

```c++
#include "kernels/add.h"            (1)
#include "kernel_invoke.h"          (2)

maybe<int> add(int a, int b) {
    return kernel::run<add>(a, b);  (3)
}
```

1. Include our `add` kernel
2. Include functionality to invoke kernels
3. Invoke the `add` kernel operation with two `int`s, and return the result.

`kernel::run<K>(...)` is a blocking function that internally creates a `kernel::runner`, and subsequently invokes the operation on `add` which matches the given arguments. There is only one operation on the kernel, and so there is only one way to invoke it.

Implicitly, a special `compute_mode` called `AUTO` is selected, which at runtime invokes an algorithm to determine which `compute_mode` to invoke. For our kernel (that supports both CUDA and CPU) the process like this:

![img](./fig-6.PNG)

Essentially, the CUDA implementation is used when `compute_mode::CUDA` is _enabled_ at compile time _and_ a valid CUDA context is available at runtime. If the kernel fails during execution for some reason (i.e. it returns an error) then we fall back to the CPU implementation. For kernels that have different combinations of `compute_mode` support, the control-flow is altered to account for this.

It's possible to override this behavior by explicitly specifying which `compute_mode` to use. For example, if we knew that the inputs to an operation were insufficiently large to benefit from the GPU, we can easily alter the behavior to include such a condition:

```c++
maybe<int> add(int a, int b)
{
    return (<some condition>) ?
          run<kernels::add, compute_mode::CPU>(a, b)
        : run<kernels::add>(a, b);
}
```

Here, if `<some condition>` evaluated to true, we'd _force_ the kernel runner to use the CPU, or otherwise proceed as normal.

<br>

### custom `kernel::runner`'s

Lastly, it's also possible to supply a non-default or custom `kernel::runner`. `kernel.h` comes with the `log_runner<K>` runner which when supplied to `run_with` will log various things during the kernel invocation. For example:

```c++
log_runner<kernels::add> log(&std::cout);
run_with<kernels::add>(log, a, b);
```

#### Sample Output:

    [add] mode=CPU
    [add] status=Success

You could probably imagine other runners which could help you write tests or benchmark your kernels.

<br>

## <a name="Part2-3"></a> Error handling

So far we've only touched on handling the return values of operations, and as part of this how we deal with errors.

Firstly, `kernel.h` doesn't throw C++ exceptions, and doesn't catch any; if your operation can throw one, then it's your users responsibility to catch it. Instead `kernel.h` internally uses the `kernel::error_code` enum to communicate various possible error states.

A kernel operation (specifically, an `op` method) is just a normal method, and so it can return any value, or `void`. However, `kernel.h` also recognizes several special return types, designed to make error handling more ergonomic.

So, lets take our `add` operation. We really want to be able to return an `int` _or_ an error. Errors can occur in CUDA, for instance, when we allocate memory. To accommodate these two possible outcomes, we can use the `variant` type, which is being [introduced in C++17](http://en.cppreference.com/w/cpp/utility/variant). `variant` is capable of holding a value that can be one of a number of possible types, and do so in a type safe way. If you want to become more familiar with variant, I suggest watching [this](https://www.youtube.com/watch?v=k3O4EKX4z1c) presentation by D. Sankel.

With `variant`, our operation changes from `int op(int, int)` to `variant<int, error_code> op(int, int)`.

The `variant<T, error_code>` return type pattern is recognized by the kernel runner. Upon encountering an error, the runner will convert this error to our library's more generic error type, `error`. Altogether, we can tabulate the return type patterns that `kernel.h` recognizes and will automatically convert:

#### Return type conversions

| `op()` type              | `kernel::run()` type   |
|-------------------------:|:-----------------------|
| `variant<T, error_code>` | `maybe<T>`             |
| `error_code`             | `option<error>`        |
| `void`                   | `option<error>`        |
| `T`                      | `maybe<T>`             |

_Note:_ `maybe<T>` is an alias for `variant<T, error>`.

<br>

## <a name="Part2-4"></a> Unit Tests

We now have a complete `mwe` module ready for binding to our application. Before we do so, we should write a simple unit test for the `add` kernel. Kernels are typically good candidates for testing because as the name suggests, they tend to be cohesive and mostly independent units of functionality. To do this we'll use the popular [google test](https://github.com/google/googletest) C++ test framework. 

Within `src/kernels/add_test.cpp` we declare a _test case_ like so:

```c++
#include "gtest/gtest.h"
#include "mwe.h"

TEST(mwe, add)
{
    auto c = mwe::add(5, 4);
    EXPECT_EQ(c, 9);
}
```

<br>

---

### 🔨&nbsp; Step 2 – Build and run the unit tests

```
mkdir build && cd build                              (1)
cmake -G "Visual Studio 14 2015 Win64" ..            (2)
cmake --build . --config Debug                       (3)
ctest . -VV -C Debug                                 (4)
```
1. Create the build directory
2. (Windows only) Generate the 64-bit build for Visual Studio
3. Build the mwe tests in debug mode
4. Run the mwe test suite with `ctest` (a tool distributed as a part of CMake)

 A successful test run should produce output similar to:

```
1: Test command: build\Debug\mwe_test.exe
1: Test timeout computed to be: 1500
1: Running main() from gmock_main.cc
1: [----------] 1 tests from mwe
1: [ RUN      ] mwe.add
1: [       OK ] mwe.add (0 ms)
1: [----------] Global test environment tear-down
1: [==========] 1 tests from 1 test cases ran. (575 ms total)
1: [  PASSED  ] 1 tests.
```

---

<br>

If you wish to find out more on the Google Test framework, the [documentation](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md) is comprehensive and easy to follow.

<br>

## <a name="Part2-5"></a> Build Options

By default the build wont enable (or require) CUDA. If you have CUDA installed, enable it by supplying the option `mwe_WITH_CUDA` to CMake at the configuration stage (below). A complete list of CMake options is maintained on the `mwe` README.

```
cmake -G "Visual Studio 14 2015 Win64" -Dmwe_WITH_CUDA=ON ..
```

---

_In the [final part](./blog-article-pt3.md) we implement our language bindings to mwe._