#include "cufoo.h"
#include "cufoo_config.h"

#include <jni/jni.hpp>

namespace util {
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

    template<typename R>
    bool try_throw(jni::JNIEnv& env, const cufoo::maybe<R>& r)
    {
        if (r.is<cufoo::error>()) {
            jni::ThrowNew(env,
                jni::FindClass(env, "java/lang/Error"),
                r.get<cufoo::error>().data());
            return true;
        }
        return false;
    }

    inline
    bool try_throw(jni::JNIEnv& env, const cufoo::status& r)
    {
        if (r) {
            jni::ThrowNew(env,
                jni::FindClass(env, "java/lang/Error"),
                r.get().data());
            return true;
        }
        return false;
    }
}

namespace
{
    struct fascade { static constexpr auto Name() { return "com/cufoo/Binding"; } };

    void register_fascade(JavaVM* vm)
    {
        using namespace cufoo;
        using namespace ::util;

        auto get_version = [](jni::JNIEnv& env, jni::Class<fascade>) -> jni::Array<jni::jint> {
            auto vec = std::vector<jni::jint>{
                cufoo_VERSION_MAJOR, cufoo_VERSION_MINOR, cufoo_VERSION_PATCH
            };
            return jni::Make<jni::Array<jni::jint>>(env, vec);
        };

        auto add = [] (jni::JNIEnv& env, jni::Class<fascade>,
                jni::jint a, jni::jint b) -> jni::jint
        {
            maybe<int> r = cufoo::add(a, b);
            return try_throw(env, r) ? 0 : r.get<int>();
        };

        auto add_all = [](jni::JNIEnv& env, jni::Class<fascade>,
                jni::Object<IntBuffer> a, jni::Object<IntBuffer> b, jni::Object<IntBuffer> result) -> void
        {
            try_throw(env, cufoo::add(
                to_span<IntBuffer>(env, a),
                to_span<IntBuffer>(env, b),
                to_span<IntBuffer>(env, result)));
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
