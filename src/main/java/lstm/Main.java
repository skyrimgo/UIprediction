package lstm;

import java.util.Arrays;

/**
 * Main
 */
public class Main {

    /*
     * LSTM
     */
    public double[] run(double[] y_pre) {
        int len = y_pre.length;
        int s = len / 8;
        int y = len % 8;

        LSTM lstm = new LSTM(100, 50, 0.1);
        String[] data_for = new String[len];
        String[] data_back = new String[len];
        //System.out.println("单向LSTM：");
        // 前向
        // 每次预测8个数
        for (int i = 0; i < s; i++) {
            String[] t_data_for = lstm.run_forward(Arrays.copyOfRange(y_pre, i * 8, i * 8 + 8));
            for (int j = 0; j < t_data_for.length; j++) {
                data_for[i * 8 + j] = t_data_for[j];
            }

        }
        // 预测剩下的
        if (y != 0) {
            String[] t_data_for = lstm.run_forward(Arrays.copyOfRange(y_pre, s * 8, s * 8 + y));
            for (int j = 0; j < t_data_for.length; j++) {
                data_for[s * 8 + j] = t_data_for[j];
            }
        }
        // 后向预测
        for (int i = 0; i < s; i++) {
            String[] t_data_back = lstm.run_backward(Arrays.copyOfRange(y_pre, i * 8, i * 8 + 8));
            for (int j = 0; j < t_data_back.length; j++) {
                data_back[i * 8 + j] = t_data_back[7 - j];
            }

        }
        // 预测剩下的
        if (y != 0) {
            String[] t_data_back = lstm.run_backward(Arrays.copyOfRange(y_pre, s * 8, s * 8 + y));
            for (int j = 0; j < t_data_back.length; j++) {
                data_back[s * 8 + j] = t_data_back[y - 1 - j];
            }
        }
        String[] data_double = new String[len];
        for (int i = 0; i < len; i++) {
            if (i == 0 || i == len - 1) {
                data_double[i] = String
                        .valueOf(0.3 * Double.parseDouble(data_back[i]) + 0.7 * Double.parseDouble(data_for[i]));
            } else {
                data_double[i] = String
                        .valueOf(0.3 * Double.parseDouble(data_back[i]) + 0.7 * Double.parseDouble(data_for[i]));
            }

            //System.out.print(data_double[i] + ":");
        }

        //System.out.println();
        double[] res = new double[len];
        for (int i = 0; i < len; i++) {
            res[i] = Double.parseDouble(data_double[i]);
        }
        return res;

    }
}