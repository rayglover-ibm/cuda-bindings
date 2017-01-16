#include "cufoo.h"
#include "cufoo_config.h"

#include <jni/jni.hpp>

namespace
{
    struct fascade { static constexpr auto Name() { return "com/cufoo/Binding"; } };

    void register_fascade(JavaVM* vm)
    {
        auto add = [] (jni::JNIEnv&, jni::Class<fascade>, jni::jint a, jni::jint b) -> jni::jint {
            return cufoo::add(a, b);
        };
        auto get_version = [](jni::JNIEnv& env, jni::Class<fascade>) -> jni::Array<jni::jint> {
            auto vec = std::vector<jni::jint>{ 
                cufoo_VERSION_MAJOR,
                cufoo_VERSION_MINOR,
                cufoo_VERSION_PATCH
            };
            return jni::Make<jni::Array<jni::jint>>(env, vec);
        };
        
        jni::JNIEnv& env { jni::GetEnv(*vm) };
        jni::RegisterNatives(env, jni::Class<fascade>::Find(env),
            jni::MakeNativeMethod("add", add),
            jni::MakeNativeMethod("version", get_version));
    }
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    register_fascade(vm);
    return jni::Unwrap(jni::jni_version_1_2);
}
