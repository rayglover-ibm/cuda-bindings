/*
 *   For more nan examples, see:
 *   https://github.com/nodejs/nan/tree/master/test/cpp
 */
#include "cufoo.h"
#include "cufoo_config.h"

#include <v8.h>
#include <node.h>
#include <v8pp/module.hpp>

namespace util
{
    template <typename T>
    gsl::span<T> as_span(v8::Isolate* iso, v8::Local<v8::ArrayBufferView>& from)
    {
        void* data = nullptr;
        size_t length = 0;

        const size_t    byte_length = from->ByteLength();
        const ptrdiff_t byte_offset = from->ByteOffset();

        v8::HandleScope scope(iso);
        v8::Local<v8::ArrayBuffer> buffer = from->Buffer();

        length = byte_length / sizeof(T);
        data = static_cast<char*>(buffer->GetContents().Data()) + byte_offset;

        /* todo: alignment check */
        assert(reinterpret_cast<uintptr_t>(data) % sizeof(T) == 0);

        return gsl::span<T>{ static_cast<T*>(data), length };
    }

    template <typename R>
    bool try_throw(const cufoo::maybe<R>& r)
    {
        if (r.is<cufoo::error>()) {
            throw std::invalid_argument(r.get<cufoo::error>().data());
        }
        return false;
    }

    inline
    bool try_throw(const cufoo::status& r)
    {
        if (r) { throw std::invalid_argument(r.get().data()); }
        return false;
    }
}

/*
 *  Conversion to span<T> from a ArrayBufferView.
 *  Doesn't permit ArrayBufferView to span<T>.
 */
template<typename T>
struct ::v8pp::convert<gsl::span<T>>
{
    using from_type = gsl::span<T>;
    using to_type = v8::Local<v8::ArrayBufferView>;

    static bool is_valid(v8::Isolate*, v8::Local<v8::Value> value) {
        return value->IsArrayBufferView();
    }

    static from_type from_v8(v8::Isolate* iso, v8::Local<v8::Value> from)
    {
        if (!is_valid(iso, from)) {
            throw std::invalid_argument("expected ArrayBufferView");
        }
        v8::HandleScope scope(iso);
        auto view = v8::Local<v8::ArrayBufferView>::Cast(from);

        return util::as_span<T>(iso, view);
    }
};

template<typename T>
struct ::v8pp::is_wrapped_class<gsl::span<T>> : std::false_type {};

namespace
{
    std::vector<int> version() {
        return { cufoo_VERSION_MAJOR, cufoo_VERSION_MINOR, cufoo_VERSION_PATCH };
    }

    int add(int a, int b)
    {
        cufoo::maybe<int> r = cufoo::add(a, b);
        return util::try_throw(r) ? 0 : r.get<int>();
    }

    v8::Local<v8::ArrayBufferView> add_all(
        v8::Isolate* iso, gsl::span<int> a, gsl::span<int> b)
    {
        using namespace v8;
        EscapableHandleScope hatch(iso);

        Local<ArrayBuffer> buffer = ArrayBuffer::New(iso, a.length() * sizeof(int));
        Local<ArrayBufferView> result = Int32Array::New(buffer, 0, a.length());

        util::try_throw(cufoo::add(a, b, util::as_span<int>(iso, result)));
        return hatch.Escape(result);
    }
}

void init(v8::Handle<v8::Object> exports)
{
    v8pp::module addon(v8::Isolate::GetCurrent());

    addon.set("version", &version);
    addon.set("addAll", &add_all);
    addon.set("add", &add);

    exports->SetPrototype(addon.new_instance());
}

NODE_MODULE(binding, init)