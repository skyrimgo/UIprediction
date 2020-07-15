package lstm;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Start
 */
public class Start {
    public static List<List<Double>> start(int lev,String path) throws IOException {

        DataRead dataRead = new DataRead(path);// datafile
        double[] y_pre = dataRead.getData();
        Main main = new Main();
        double[] t_res = main.run(y_pre);
        List<List<Double>> lists = new ArrayList<>();
        List<Double> list1 = new ArrayList<>();
        for (double var : t_res) {

            list1.add(var);
            if (list1.size() == t_res.length)
                lists.add(list1);
        }

        for (int i = 2; i <= lev; i++) {
            double[] t_res_next = main.run(t_res);
            List<Double> list = new ArrayList<>();
            for (double var : t_res_next) {

                list.add(var);
                if (list.size() == t_res.length)
                    lists.add(list);
            }
        }
        return lists;

    }
}