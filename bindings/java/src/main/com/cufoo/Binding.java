package com.cufoo;

import java.nio.IntBuffer;
import java.nio.ByteBuffer;

import com.cufoo.BindingUtil;

public class Binding {
    static { System.loadLibrary("binding"); }

    public static native int[] version();

    public static native int add(int a, int b);
    private static native void addAll(IntBuffer a, IntBuffer b, IntBuffer result);
    
    public static IntBuffer addAll(IntBuffer a, IntBuffer b) {
        IntBuffer result = BindingUtil.allocateDirectIntBuffer(a.capacity());
        addAll(a, b, result);
        return result;
    }
}