package lstm;


import filter.MyParticleFilter;
import filter.ParticleFilter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class _Main {
    //public static String path = "C:\\Users\\shinelon\\Desktop\\基于粒子滤波的LSTM.xlsx";

    public static List<List<Double>> run(List<List<Double>> lists, boolean isbiprediction, boolean ispsofilter, int avgfilter, String readPath) throws Exception {
        DataRead dataRead = new DataRead(readPath);// datafile
        double[] y_pre = dataRead.getData();
        List<Double> l = new ArrayList<>();
        for (double data : y_pre) {
            l.add(data);
        }
        ParticleFilter myParticleFilter = new MyParticleFilter(1000, 50, 0.1);

        Double pre = Arrays.stream(y_pre).sum() / y_pre.length;
        if (ispsofilter) {
            for (int i = 0; i < lists.size(); i++) {
                lists.set(i, myParticleFilter.getResultValues(pre, lists.get(i)));
            }
        }
        //均值滤波次数,只对最后一层
        for (int c = 0; c < avgfilter; c++) {

            for (int i = 0; i < l.size(); i++) {
                double[] sums = new double[l.size()];
                double count = 0;
                for (int j = Math.max(i - 4, 0); j < Math.min(l.size(), i + 4); j++) {
                    sums[i] += lists.get(lists.size() - 1).get(j);
                    count++;
                }
                lists.get(lists.size() - 1).set(i, sums[i] / count);
            }
        }
        return lists;
        //POI.init(path);
        //POI.write(path, l, lists.get(lists.size() - 1), 0);
    }
}
