package com.cufoo.tests;

import static com.cufoo.Binding.*;

public class BindingTest {
    public static void main(String[] args) {
        System.out.println("[java] version=" + java.util.Arrays.toString(version()));
        
        int c = add(5, 3);
        assert(c == 8);
    }
}