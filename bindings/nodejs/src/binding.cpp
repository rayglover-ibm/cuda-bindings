#include <nan.h>
#include "libsr.h"

using namespace Nan;

NAN_METHOD(add)
{
    if (info.Length() != 2 || !info[0]->IsNumber() || !info[1]->IsNumber()) {
        ThrowTypeError("expected (number, number)");
        return;
    }

    int32_t a = To<int32_t>(info[0]).FromJust();
    int32_t b = To<int32_t>(info[1]).FromJust();

    int32_t c = libsr::add(a, b);
    info.GetReturnValue().Set(c);
}

NAN_MODULE_INIT(InitAll)
{
    Set(target,
        New<v8::String>("add").ToLocalChecked(),
        GetFunction(New<v8::FunctionTemplate>(add)).ToLocalChecked());
}

NODE_MODULE(addon, InitAll)
