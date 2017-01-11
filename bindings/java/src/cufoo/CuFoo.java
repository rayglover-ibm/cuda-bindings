package cufoo;

public class CuFoo {
    static { System.loadLibrary("binding"); }

    public static native int[] version();
    public static native int add(int a, int b);
}