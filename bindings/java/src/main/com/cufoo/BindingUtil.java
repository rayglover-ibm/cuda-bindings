package com.cufoo;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ByteOrder;

public class BindingUtil
{
    private static int SIZEOF_INT = Integer.SIZE / 8;

    public static IntBuffer allocateDirectIntBuffer(int size) {
        int bytes = size * SIZEOF_INT; 
        
        return ByteBuffer.allocateDirect(bytes)
            .order(ByteOrder.nativeOrder())
            .asIntBuffer(); 
    }

    public static IntBuffer allocateDirectBufferFrom(int[] data) {
        IntBuffer view = allocateDirectIntBuffer(data.length);
        
        view.put(data);
        view.rewind();
        
        return view;
    }
}