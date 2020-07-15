package lstm;

class ToyLossLayer {

    /**
     * compute square loss with first element of hidden layer array
     */

    public double loss(double[] pred, double label) {
        return (pred[0] - label) * (pred[0] - label);
    }

    public double[] bottom_diff(double[] pred, double label) {
        double[] diff = new double[pred.length];
        diff[0] = 2 * (pred[0] - label);
        return diff;
    }

}