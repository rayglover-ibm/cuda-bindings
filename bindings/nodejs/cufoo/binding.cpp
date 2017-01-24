/*
 *   For more nan examples, see: 
 *   https://github.com/nodejs/nan/tree/master/test/cpp
 */
#include "cufoo.h"
#include "cufoo_config.h"

#include <nan.h>

using namespace Nan;

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
    int32_t c = cufoo::add(a, b);

    info.GetReturnValue().Set(c);
}

NAN_MODULE_INIT(InitAll)
{
    Set(target,
        New<v8::String>("add").ToLocalChecked(),
        GetFunction(New<v8::FunctionTemplate>(add)).ToLocalChecked());

    Set(target,
        New<v8::String>("version").ToLocalChecked(),
        GetFunction(New<v8::FunctionTemplate>(version)).ToLocalChecked());
}

NODE_MODULE(addon, InitAll)
