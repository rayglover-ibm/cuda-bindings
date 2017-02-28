/*
 *   For more nan examples, see:
 *   https://github.com/nodejs/nan/tree/master/test/cpp
 */
#include "cufoo.h"
#include "cufoo_config.h"

#include <nan.h>

using namespace Nan;

namespace util
{
    template <typename T>
    gsl::span<T> as_span(TypedArrayContents<T>& info) {
        return gsl::span<T>{ (*info), info.length() };
    }

    template <typename R>
    bool try_throw(const cufoo::maybe<R>& r)
    {
        if (r.is<cufoo::error>()) {
            Nan::ThrowError(r.get<cufoo::error>().data());
            return true;
        }
        return false;
    }

    inline
    bool try_throw(const cufoo::status& r)
    {
        if (r) {
            Nan::ThrowError(r.get().data());
            return true;
        }
        return false;
    }
}

NAN_METHOD(version)
{
    auto arr = New<v8::Array>(3);
    arr->Set(0, New<v8::Number>(cufoo_VERSION_MAJOR));
    arr->Set(1, New<v8::Number>(cufoo_VERSION_MINOR));
    arr->Set(2, New<v8::Number>(cufoo_VERSION_PATCH));

    info.GetReturnValue().Set(arr);
}

NAN_METHOD(add)
{
    if (info.Length() != 2 || !info[0]->IsNumber() || !info[1]->IsNumber()) {
        ThrowTypeError("expected (number, number)");
        return;
    }

    int32_t a = To<int32_t>(info[0]).FromJust();
    int32_t b = To<int32_t>(info[1]).FromJust();

    cufoo::maybe<int> result = cufoo::add(a, b);

    if (!util::try_throw(result)) {
        info.GetReturnValue().Set(result.get<int>());
    }
}

NAN_METHOD(addAll)
{
    using namespace util;

    if (info.Length() != 2 || !info[0]->IsArrayBufferView() || !info[1]->IsArrayBufferView()) {
        ThrowTypeError("expected (ArrayBufferView, ArrayBufferView)");
        return;
    }
    TypedArrayContents<int> a(info[0]);
    TypedArrayContents<int> b(info[1]);

    v8::Local<v8::ArrayBuffer> buffer = v8::ArrayBuffer::New(
        v8::Isolate::GetCurrent(), a.length() * sizeof(int));

    auto result = v8::Int32Array::New(buffer, 0, a.length());
    TypedArrayContents<int> view(result);

    if (!try_throw(cufoo::add(as_span(a), as_span(b), as_span(view))))
        info.GetReturnValue().Set(result);
}

NAN_MODULE_INIT(InitAll)
{
    Set(target,
        New<v8::String>("add").ToLocalChecked(),
        GetFunction(New<v8::FunctionTemplate>(add)).ToLocalChecked());

    Set(target,
        New<v8::String>("addAll").ToLocalChecked(),
        GetFunction(New<v8::FunctionTemplate>(addAll)).ToLocalChecked());

    Set(target,
        New<v8::String>("version").ToLocalChecked(),
        GetFunction(New<v8::FunctionTemplate>(version)).ToLocalChecked());
}

NODE_MODULE(binding, InitAll)
