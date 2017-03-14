/*
 *   For more nan examples, see:
 *   https://github.com/pmed/v8pp/tree/master/examples
 */
#include "mwe.h"
#include "mwe_config.h"
#include "binding_util.h"

#include <v8.h>
#include <node.h>
#include <v8pp/module.hpp>

namespace
{
    std::vector<int> version() {
        return { mwe_VERSION_MAJOR, mwe_VERSION_MINOR, mwe_VERSION_PATCH };
    }

    v8::Local<v8::ArrayBufferView> add_all(
        v8::Isolate* iso, gsl::span<int> a, gsl::span<int> b)
    {
        using namespace v8;
        EscapableHandleScope hatch(iso);

        Local<ArrayBuffer> buffer = ArrayBuffer::New(iso, a.length() * sizeof(int));
        Local<ArrayBufferView> result = Int32Array::New(buffer, 0, a.length());

        util::try_throw(mwe::add(a, b, util::as_span<int>(iso, result)));
        return hatch.Escape(result);
    }
}

void init(v8::Local<v8::Object> exports)
{
    v8pp::module m(v8::Isolate::GetCurrent());

    m.set("version", &version)
     .set("add",     [](int a, int b) { return mwe::add(a, b); })
     .set("addAll",  &add_all);

    exports->SetPrototype(m.new_instance());
}

NODE_MODULE(binding, init)