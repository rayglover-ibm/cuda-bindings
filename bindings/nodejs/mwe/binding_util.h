/*  Copyright 2017 International Business Machines Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#pragma once

#include "mwe.h"

#include <v8.h>
#include <v8pp/convert.hpp>

namespace util
{
    /*  Given an v8::ArrayBufferView, produces a span<T>
     *  over the view
     */
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
    bool try_throw(const mwe::maybe<R>& r)
    {
        if (r.template is<mwe::error>())
            throw std::invalid_argument(r.template get<mwe::error>().data());

        return false;
    }

    inline
    bool try_throw(const mwe::status& r)
    {
        if (r) { throw std::invalid_argument(r.get().data()); }
        return false;
    }
}


/*  v8pp type converters --------------------------------------------------- */

namespace v8pp
{
    /*
     *  Conversion to span<T> from a v8::ArrayBufferView.
     *  Doesn't permit v8::ArrayBufferView to span<T>.
     */
    template<typename T>
    struct convert<gsl::span<T>>
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
    struct is_wrapped_class<gsl::span<T>> : std::false_type {};


    /*
     *  Conversion to v8::Value from a mwe::maybe<T>; will
     *  throw an exception if the incoming mwe::maybe<T> is
     *  in an error state.
     */
    template<typename T>
    struct convert<mwe::maybe<T>>
    {
        using from_type = mwe::maybe<T>;
        using to_type = v8::Local<v8::Value>;

        static to_type to_v8(v8::Isolate* iso, const mwe::maybe<T>& val)
        {
            util::try_throw(val);

            v8::EscapableHandleScope hatch(iso);
            to_type to = v8pp::to_v8(iso, val.template get<T>());

            return hatch.Escape(to);
        }
    };

    template<typename T>
    struct is_wrapped_class<mwe::maybe<T>> : std::false_type {};
}