import cufoo.*;

public class Basic {
    public static void main(String[] args) {
        System.out.println("[java] version=" + java.util.Arrays.toString(CuFoo.version()));
        
        int c = CuFoo.add(5, 3);
        assert(c == 8);
    }
}