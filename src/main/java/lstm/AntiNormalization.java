package lstm;

public class AntiNormalization {
    public static double antiNormalization(double i, double fix) {
        return Math.tan(i * Math.PI / 2) * fix;
    }
}