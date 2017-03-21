# *CUDA language bindings* – Minimum working examples

_This is the accompanying code for the tutorial **CUDA language bindings – with modern C++** (available [here](https://rayglover-ibm.github.io/cuda-bindings/).) This repository contains everything you need to follow that tutorial, and serves as a reference for your own projects._

### Example Bindings

|                | Linux <br> [![Build Status](https://travis-ci.org/rayglover-ibm/cuda-bindings.svg?branch=master)](https://travis-ci.org/rayglover-ibm/cuda-bindings) | Mac <br> [![Build Status](https://travis-ci.org/rayglover-ibm/cuda-bindings.svg?branch=master)](https://travis-ci.org/rayglover-ibm/cuda-bindings)   | Windows <br> &nbsp; |  Android  <br> &nbsp;    |
|----------------|:----------:|:--------:|:---------------------:|:-------------:|
| _Python_       |    3       |  3       | 3                     |  ✗           |
| _Node.js (v8)_ |    4/5     |  4/5     | 4/5                   |  ✗           |
| _Java_         |    1.7     |  1.7     | 1.7                   |  ✓           |

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
    | `kernelpp_WITH_CUDA`     | Enable cuda support    | OFF     |
    | `mwe_WITH_TESTS`         | Enable unit tests      | ON      |
    | `mwe_WITH_PYTHON`        | Enable python binding  | OFF     |
    | `mwe_WITH_NODEJS`        | Enable nodejs binding  | OFF     |
    | `mwe_WITH_JAVA`          | Enable java binding    | OFF     |

- ### Compilers / Runtimes 
    
    _(Minimum tested versions)_

    |                | Windows    | Linux (Ubuntu 15/16)  | Android         | Mac     |
    |----------------|:----------:|:---------------------:|:---------------:|:-------:|
    | *C++ compiler* | VS 2015    | gcc 5.3 / clang 3.6   | NDK r13 (clang) | XCode 7 |
    | *CUDA SDK*     | 8          | 8                     | ✗               | 8       |

<br>

### Build & Test for Android

As of version 3.7 of CMake, it's possible to cross-compile for Android out-of-the-box. Here I assume you've installed [Android Studio](https://developer.android.com/studio/index.html#downloads), or at a minimum, the Android command-line tools. I also assume you've installed the NDK.

1. Install the [Ninja](https://ninja-build.org/) build system. We'll use Ninja in place of Visual Studio because VS doesn't support the various NDK toolchains.

2. From the mwe repository root:
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
    adb push mwe_test /data/local/tmp/mwe_test
    adb shell
    chmod 755 /data/local/tmp/mwe_test
    /data/local/tmp/mwe_test
    ```

<br>

## License

```
Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
