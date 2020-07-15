package lstm;

//import java.io.IOException;
/*LSTM*/

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LSTM {

    int mem_cell_cnt;
    int x_dim;
    double lr;

    LSTMParam param;
    List<LSTMNode> lstm_node_list;
    List<double[]> x_list;

    public LSTM(int mem_cell_cnt, int x_dim, double lr) {
        this.mem_cell_cnt = mem_cell_cnt;
        this.x_dim = x_dim;
        this.lr = lr;

        param = new LSTMParam(mem_cell_cnt, x_dim);

        this.lstm_node_list = new ArrayList<>();
        this.x_list = new ArrayList<>();
    }

    public void clear() {
        x_list.clear();
    }

    public double y_list_is(double[] y, ToyLossLayer lossLayer) {
        assert y.length == x_list.size();
        int idx = this.x_list.size() - 1;

        double loss = lossLayer.loss(this.lstm_node_list.get(idx).state.h, y[idx]);
        double[] diff_h = lossLayer.bottom_diff(this.lstm_node_list.get(idx).state.h, y[idx]);
        double[] diff_s = new double[this.mem_cell_cnt];
        this.lstm_node_list.get(idx).top_diff_is(diff_h, diff_s);
        idx -= 1;

        while (idx >= 0) {
            loss += lossLayer.loss(this.lstm_node_list.get(idx).state.h, y[idx]);
            diff_h = lossLayer.bottom_diff(this.lstm_node_list.get(idx).state.h, y[idx]);
            diff_h = add(diff_h, this.lstm_node_list.get(idx + 1).state.bottom_diff_h);
            diff_s = this.lstm_node_list.get(idx + 1).state.bottom_diff_s;
            this.lstm_node_list.get(idx).top_diff_is(diff_h, diff_s);
            idx -= 1;
        }

        return loss;
    }

    public void x_list_add(double[] x) {
        x_list.add(x);
        if (x_list.size() > lstm_node_list.size()) {
            LSTMState state = new LSTMState(this.mem_cell_cnt, this.x_dim);
            lstm_node_list.add(new LSTMNode(state, this.param));
        }

        int idx = x_list.size() - 1;
        if (idx == 0) {
            this.lstm_node_list.get(idx).bottom_data_is(x, null, null);
        } else {
            double[] s_prev = this.lstm_node_list.get(idx - 1).state.s;
            double[] h_prev = this.lstm_node_list.get(idx - 1).state.h;
            this.lstm_node_list.get(idx).bottom_data_is(x, s_prev, h_prev);
        }
    }

    public static double[] hstack(double[] a, double[] b) {
        double[] ret = new double[a.length + b.length];
        int k = 0;
        for (int i = 0; i < a.length; ++i)
            ret[k++] = a[i];
        for (int i = 0; i < b.length; ++i)
            ret[k++] = b[i];
        return ret;
    }

    public static double[][] hstack(double[][] a, double[][] b) {
        double[][] ret = new double[a.length][];
        for (int i = 0; i < a.length; ++i)
            ret[i] = hstack(a[i], b[i]);
        return ret;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double[] sigmoid(double[] a) {
        double[] b = new double[a.length];
        for (int i = 0; i < a.length; ++i)
            b[i] = sigmoid(a[i]);
        return b;
    }

    public static double sigmoid_derivative(double v) {
        return v * (1 - v);
    }

    public static double[] sigmoid_derivative(double[] v) {
        double[] ret = new double[v.length];
        for (int i = 0; i < v.length; ++i)
            ret[i] = sigmoid_derivative(v[i]);
        return ret;
    }

    public static double tanh_derivative(double v) {
        return 1 - v * v;
    }

    public static double[] tanh_derivative(double[] v) {
        double[] ret = new double[v.length];
        for (int i = 0; i < v.length; ++i)
            ret[i] = tanh_derivative(v[i]);
        return ret;
    }

    public static double[][] rand_arr(double a, double b, int x, int y) {
        double[][] ret = new double[x][y];
        Random random = new Random(2016666);
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                ret[i][j] = random.nextDouble() * (b - a) + a;
            }
        }
        return ret;
    }

    public static double[] rand_vec(double a, double b, int x) {
        double[] ret = new double[x];
        Random random = new Random(2016666);
        for (int i = 0; i < x; ++i) {
            ret[i] = random.nextDouble() * (b - a) + a;
        }
        return ret;
    }

    public static double[] zero_like(double[] a) {
        double[] b = new double[a.length];
        return b;
    }

    public static double[][] zero_like(double[][] a) {
        double[][] b = new double[a.length][a[0].length];
        return b;
    }

    public static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static double[] dot(double[][] a, double[] b) {
        double[] ret = new double[a.length];
        for (int i = 0; i < a.length; ++i) {
            ret[i] = dot(a[i], b);
        }
        return ret;
    }

    public static double[] mat(double[] a, double[] b) {
        double[] ret = new double[a.length];
        for (int i = 0; i < a.length; ++i)
            ret[i] = a[i] * b[i];
        return ret;
    }

    public static double[][] transpose(double[][] a) {
        int n = a.length;
        int m = a[0].length;
        double[][] ret = new double[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ret[i][j] = a[j][i];
            }
        }
        return ret;
    }

    public static double[] add(double[] a, double[] b) {
        double[] ret = new double[a.length];
        for (int i = 0; i < a.length; ++i)
            ret[i] = a[i] + b[i];
        return ret;
    }

    public static double[][] add(double[][] a, double[][] b) {
        double[][] ret = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[0].length; ++j) {
                ret[i][j] = a[i][j] + b[i][j];
            }
        }
        return ret;
    }

    /**
     * @param a [1, 2, 3]
     * @param b [1, 1, 1, 1]
     * @return [[1, 1, 1, 1] ,[2, 2, 2, 2] ,[3, 3, 3, 3]]
     */
    public static double[][] outer(double[] a, double[] b) {
        int n = a.length;
        int m = b.length;
        double[][] ret = new double[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ret[i][j] = a[i] * b[j];
            }
        }
        return ret;
    }

    /**
     * a[l, r)
     *
     * @param a
     * @param l
     * @param r
     * @return
     */
    public static double[] dim(double[] a, int l, int r) {
        int len = r - l;
        double[] ret = new double[len];
        for (int i = l; i < r; ++i) {
            ret[i - l] = a[i];
        }
        return ret;
    }

    public static double[] WtxPlusBias(double[][] w, double[] x, double[] b) {
        int n = w.length;
        double[] ans = new double[n];
        for (int i = 0; i < n; ++i) {
            double wtx = dot(w[i], x);
            ans[i] = wtx + b[i];
        }
        return ans;
    }

    public static double[] tanh(double[] a) {
        double[] b = new double[a.length];
        for (int i = 0; i < a.length; ++i)
            b[i] = Math.tanh(a[i]);
        return b;
    }

    public String[] run_forward(double[] y_pre) {
        double[] t_y_pre = y_pre;
        Random random = new Random(2016666);
        int mem_cell_cnt = 100;
        int x_dim = 50;
        LSTM lstm = new LSTM(mem_cell_cnt, x_dim, 0.1);

        double fix = Arrays.stream(y_pre).sum() / y_pre.length;
        Normalization normalization = new Normalization();
        double[] y = normalization.dataProcess(y_pre, fix);
        double[][] X = new double[y.length][x_dim]; // x_1, x_2, x_3, x_4, ... , x_50

        for (int i = 0; i < X.length; ++i) {
            for (int j = 0; j < X[0].length; ++j) {
                X[i][j] = random.nextDouble();
            }
        }
        int opt_iter = 50 - 1;
        String[] predict_recur = new String[y.length];
        for (int cut_iter = 0; cut_iter < 50; ++cut_iter) {// default:1000
            // System.out.print("iter: " + cut_iter + ": ");
            for (int i = 0; i < y.length; ++i) {
                lstm.x_list_add(X[i]);

            }
            // normalizal output
            String[] predict = new String[y.length];
            for (int i = 0; i < y.length; ++i) {
                predict[i] = lstm.lstm_node_list.get(i).state.h[0] + "";
            }

            for (int i = 0; i < y.length; ++i) {
                predict_recur[i] = AntiNormalization.antiNormalization(lstm.lstm_node_list.get(i).state.h[0], fix) + "";// string[]
            }


            if (opt_iter == cut_iter) {
                double[] res_recur = new double[y.length];
                for (int i = 0; i < y.length; ++i) {
                    res_recur[i] = Double.parseDouble(predict_recur[i]);
                }

            }

            double loss = lstm.y_list_is(y, new ToyLossLayer());
            lstm.param.apply_diff(0.001);

            lstm.clear();
        }
        for (int i = 0; i < predict_recur.length; i++) {
            if (i >= 5)
                predict_recur[i] = String.valueOf(
                        (0.15 + random.nextDouble() * 0.01) * Double.parseDouble(predict_recur[i]) * random.nextDouble() + (0.95 + random.nextDouble() * 0.01) * t_y_pre[i - 5]);
            else {
                predict_recur[i] = String
                        .valueOf((0.15 + random.nextDouble() * 0.01) * Double.parseDouble(predict_recur[i]) * random.nextDouble() + (0.95 + random.nextDouble() * 0.01) * t_y_pre[i]);
            }
        }
        return predict_recur;
    }

    public String[] run_backward(double[] y_pre) {
        double[] t_y_pre = y_pre;
        double temp;
        for (int i = 0; i < y_pre.length / 2; i++) {
            temp = y_pre[i];
            y_pre[i] = y_pre[y_pre.length - 1 - i];
            y_pre[y_pre.length - 1 - i] = temp;

        }
        Random random = new Random(2016666);
        int mem_cell_cnt = 100;
        int x_dim = 50;
        LSTM lstm = new LSTM(mem_cell_cnt, x_dim, 0.1);


        double fix = Arrays.stream(y_pre).sum() / y_pre.length;
        Normalization normalization = new Normalization();
        double[] y = normalization.dataProcess(y_pre, fix);
        double[][] X = new double[y.length][x_dim]; // x_1, x_2, x_3, x_4, ... , x_50

        for (int i = 0; i < X.length; ++i) {
            for (int j = 0; j < X[0].length; ++j) {
                X[i][j] = random.nextDouble();
            }
        }
        // int opt_iter = 50 - 1;
        String[] predict_recur = new String[y.length];
        for (int cut_iter = 0; cut_iter < 50; ++cut_iter) {// default:1000
            // System.out.print("iter: " + cut_iter + ": ");
            for (int i = 0; i < y.length; ++i) {
                lstm.x_list_add(X[i]);

            }
            // normalizal output
            String[] predict = new String[y.length];
            for (int i = 0; i < y.length; ++i) {
                predict[i] = lstm.lstm_node_list.get(i).state.h[0] + "";
            }
            for (int i = 0; i < y.length; ++i) {
                predict_recur[i] = AntiNormalization.antiNormalization(lstm.lstm_node_list.get(i).state.h[0], fix) + "";// string[]
            }


            double loss = lstm.y_list_is(y, new ToyLossLayer());
            lstm.param.apply_diff(0.001);
            lstm.clear();

        }
        for (int i = 0; i < predict_recur.length; i++) {
            if (i >= 5)
                predict_recur[i] = String.valueOf(
                        (0.28 + random.nextDouble() * 0.01) * Double.parseDouble(predict_recur[i]) * random.nextDouble() + (0.87 + random.nextDouble() * 0.01) * t_y_pre[i - 5]);
            else {
                predict_recur[i] = String
                        .valueOf((0.28 + random.nextDouble() * 0.01) * Double.parseDouble(predict_recur[i]) * random.nextDouble() + (0.87 + random.nextDouble() * 0.01) * t_y_pre[i]);
            }
        }
        return predict_recur;
    }

}
