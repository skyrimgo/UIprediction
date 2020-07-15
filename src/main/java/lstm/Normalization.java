package lstm;

/** data normalization */
public class Normalization {
    public double[] dataProcess(double[] dataGet,double fix) {
        double[] dataProcess = new double[dataGet.length];
        int i = 0;
        for (double var : dataGet) {
            dataProcess[i++] = Math.atan(var / fix) * 2 / Math.PI;
        }
        return dataProcess;
    }
}