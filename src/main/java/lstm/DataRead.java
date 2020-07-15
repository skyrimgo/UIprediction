package lstm;
/**read data from file */
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class DataRead {
    String fileName;
    // double[] getData;

    public DataRead(String fileName) {
        this.fileName = fileName;
    }

    public double[] getData() throws NumberFormatException, IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
        // BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String line;
        int i = 0;
        List<Double> list = new ArrayList<>();
        while ((line = in.readLine()) != null) {
            list.add(Double.parseDouble(line));
        }
        in.close();
        double[] datas = new double[list.size()];
        for (double data : list) {
            datas[i++] = data;
        }
        return datas;
    }
}