package com.mwe.tests;

import java.nio.IntBuffer;
import java.util.Arrays;

import static com.mwe.Binding.*;
import static com.mwe.BindingUtil.*;

public class BindingTest
{
    public static void addTest() {
        int c = add(5, 3);
        assert c == 8;
    }

    public static void addAllTest() {
        IntBuffer a = allocateDirectBufferFrom(new int[]{ 0, 1, 2, 3 });
        IntBuffer b = allocateDirectBufferFrom(new int[]{ 4, 5, 6, 7 });

        IntBuffer c = addAll(a, b);
        assert c.equals(allocateDirectBufferFrom(new int[]{ 4, 6, 8, 10 }));
    }

    public static void main(String[] args) {
        System.out.println("[java] version=" + Arrays.toString(version()));

        addTest();
        addAllTest();
    }
}