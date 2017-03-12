# *CUDA language bindings* – Minimum working examples

_This is the accompanying code for the tutorial **CUDA language bindings – with modern C++**. This repository contains everything you need to follow that tutorial, and serve as a reference for extending Python, Node.js and Java applications with CUDA._

### Example Bindings

|               |  Windows 7 |  Linux (Ubuntu 15/16) | Android       |
|---------------|:----------:|:---------------------:|:-------------:|
| Python        |    3.5     | 3.5                   |  ✗           |
| Node.js (v8)  |    4.3     | 5.2                   |  ✗           |
| Java          |    1.7     | 1.7                   |  ✓           |

<br>

## Build

__Requirements:__ you'll need CMake 3.5. If you want to build for Android (see below) 3.7 is required. Refer to the tutorial for concrete examples.

```
mkdir build && cd build
cmake [-G <generator>] <cmake-options> ..
cmake --build . [--config Debug]
```

- ### CMake options

    | CMake option             | Description            | Default |
    |--------------------------|:-----------------------|:--------|
    | `cufoo_WITH_TESTS`       | Enable unit tests      | ON      |
    | `cufoo_WITH_CUDA`        | Enable cuda support    | OFF     |
    | `cufoo_WITH_PYTHON`      | Enable python binding  | OFF     |
    | `cufoo_WITH_NODEJS`      | Enable nodejs binding  | OFF     |
    | `cufoo_WITH_JAVA`        | Enable java binding    | OFF     |

- ### Compilers / Runtimes 
    _(Minimum tested versions)_

    |               |  Windows 7 |  Linux (Ubuntu 15/16) | Android       |
    |---------------|:----------:|:---------------------:|:-------------:|
    | C++ compiler  | VS 2015    | gcc 5.4 / clang 3.6   |  NDK r13      |
    | CUDA SDK      | 7.5/8.0    | 7.5/8.0               |  ✗           |

<br>

### Build & Test for Android

As of version 3.7 of CMake, it's possible to cross-compile for Android out-of-the-box. Here I assume you've installed [Android Studio](https://developer.android.com/studio/index.html#downloads), or at a minimum, the Android command-line tools. I also assume you've installed the NDK.

1. Install the [Ninja](https://ninja-build.org/) build system. We'll use Ninja in place of Visual Studio because VS doesn't support the various NDK toolchains.

2. From the cufoo repository root:
    ```bash
    $ mkdir build-android && cd build-android
    $ cmake -G "Ninja"                      \
        -DCMAKE_SYSTEM_NAME=Android         \
        -DCMAKE_SYSTEM_VERSION=24           \
        -DCMAKE_ANDROID_NDK_TOOLCHAIN_VERSION=clang \
        -DCMAKE_ANDROID_ARCH_ABI=x86        \
        -DCMAKE_ANDROID_NDK="<path to ndk>" \
        -DCMAKE_ANDROID_STL_TYPE=c++_static \
        -DCMAKE_BUILD_TYPE=Debug ..
    ```
    The documentation for the various options is vailable [here](https://cmake.org/cmake/help/v3.7/manual/cmake-toolchains.7.html#cross-compiling-for-android). Next, we build in the usual way:
    ```bash
    cmake --build . --config Debug
    ```
3. Launch your virtual device. In Android studio, you can use the [AVD Manager](https://developer.android.com/studio/run/managing-avds.html). You should make sure the `CMAKE_ANDROID_ARCH_ABI` and `CMAKE_SYSTEM_VERSION` you gave above reflects the Android device you're testing on.

4. With the Android Debug Bridge (adb) tool (usually located in the `platform-tools` directory) upload the unit-test binary to the device, make it executable, and run:
    ```
    adb push cufoo_test /data/local/tmp/cufoo_test
    adb shell
    chmod 755 /data/local/tmp/cufoo_test
    /data/local/tmp/cufoo_test
    ```
