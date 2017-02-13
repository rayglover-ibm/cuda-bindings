#include "cufoo.h"
#include "cufoo_config.h"

#include <jni/jni.hpp>

namespace {
    struct IntBuffer
    {
        static constexpr auto Name() { return "java/nio/IntBuffer"; }
        using element_type = int32_t;
    };
    
    template<
        typename Buffer,
        typename Elem = typename Buffer::element_type
        >
    gsl::span<Elem> to_span(
        jni::JNIEnv& env, jni::Object<Buffer>& o
        )
    {
        void* buff = jni::GetDirectBufferAddress(env, *o);
        size_t len = jni::GetDirectBufferCapacity(env, *o);

        return gsl::span<Elem>{ reinterpret_cast<Elem*>(buff) , len };
    }
}

namespace
{
    struct fascade { static constexpr auto Name() { return "com/cufoo/Binding"; } };

    void register_fascade(JavaVM* vm)
    {
        auto get_version = [](jni::JNIEnv& env, jni::Class<fascade>) -> jni::Array<jni::jint> {
            auto vec = std::vector<jni::jint>{ 
                cufoo_VERSION_MAJOR, cufoo_VERSION_MINOR, cufoo_VERSION_PATCH
            };
            return jni::Make<jni::Array<jni::jint>>(env, vec);
        };
        
        auto add = [] (jni::JNIEnv&, jni::Class<fascade>,
                jni::jint a, jni::jint b) -> jni::jint
        {
            return cufoo::add(a, b);
        };

        auto add_all = [](jni::JNIEnv& env, jni::Class<fascade>, 
                jni::Object<IntBuffer> a, jni::Object<IntBuffer> b, jni::Object<IntBuffer> result) -> void
        {
            cufoo::add(::to_span<::IntBuffer>(env, a), 
                       ::to_span<::IntBuffer>(env, b),
                       ::to_span<::IntBuffer>(env, result));
        };
        
        jni::JNIEnv& env { jni::GetEnv(*vm) };
        jni::RegisterNatives(env, jni::Class<fascade>::Find(env),
            jni::MakeNativeMethod("add", add),
            jni::MakeNativeMethod("addAll", add_all),
            jni::MakeNativeMethod("version", get_version));
    }
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*)
{
    register_fascade(vm);
    return jni::Unwrap(jni::jni_version_1_2);
}
