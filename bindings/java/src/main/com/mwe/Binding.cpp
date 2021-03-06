#include "mwe.h"
#include "mwe_config.h"

#include <jni/jni.hpp>

struct IntBuffer
{
    static constexpr auto Name() { return "java/nio/IntBuffer"; }
    using element_type = int32_t;
};

namespace util
{
    template<
        typename Buffer,
        typename Elem = typename Buffer::element_type
        >
    gsl::span<Elem> to_span(
        jni::JNIEnv& env, jni::Object<Buffer>& o)
    {
        void* buff = jni::GetDirectBufferAddress(env, *o);
        size_t len = jni::GetDirectBufferCapacity(env, *o);

        return gsl::span<Elem>{ reinterpret_cast<Elem*>(buff) , len };
    }

    template<typename R>
    bool try_throw(jni::JNIEnv& env, const kernelpp::maybe<R>& r)
    {
        if (r.template is<kernelpp::error>()) {
            jni::ThrowNew(env,
                jni::FindClass(env, "java/lang/Error"),
                r.template get<kernelpp::error>().data());
            return true;
        }
        return false;
    }

    inline
    bool try_throw(jni::JNIEnv& env, const kernelpp::status& r)
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

using namespace jni;

struct Binding
{
    static constexpr auto Name() { return "com/mwe/Binding"; };
    using _this = jni::Class<Binding>;

    static Array<jint> version(JNIEnv& env, _this)
    {
        auto vec = std::vector<jint>{
            mwe_VERSION_MAJOR, mwe_VERSION_MINOR, mwe_VERSION_PATCH
        };
        return Make<Array<jint>>(env, vec);
    }

    static jint add(JNIEnv& env, _this, jint a, jint b)
    {
        kernelpp::maybe<int> r = mwe::add(a, b);
        return util::try_throw(env, r) ? 0 : r.get<int>();
    }

    static void addAll(JNIEnv& env, _this,
        Object<IntBuffer> a, Object<IntBuffer> b, Object<IntBuffer> result)
    {
        util::try_throw(env, mwe::add(
            util::to_span<IntBuffer>(env, a),
            util::to_span<IntBuffer>(env, b),
            util::to_span<IntBuffer>(env, result)));
    }

    static void register_jni(JNIEnv& env)
    {
        RegisterNatives(env, _this::Find(env),
            MakeNativeMethod<decltype(&add), &add>("add"),
            MakeNativeMethod<decltype(&addAll), &addAll>("addAll"),
            MakeNativeMethod<decltype(&version), &version>("version"));
    }
};

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*)
{
    Binding::register_jni(GetEnv(*vm));
    return Unwrap(jni_version_1_2);
}
