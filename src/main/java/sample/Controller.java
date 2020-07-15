package sample;

import com.jfoenix.controls.*;
import javafx.fxml.FXML;

import javafx.fxml.Initializable;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;
import lombok.AllArgsConstructor;
import lstm.Start;
import lstm._Main;
import utils.POI;

import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class Controller implements Initializable {
    private double xOffset;
    private double yOffset;
    private Stage stage;

    @FXML
    private AnchorPane topbar, setPane, userPane, statsPane;
    @FXML
    private ImageView image_set, image_user, image_stats, image_user_login, image_power, setPaneArrow, userPaneArrow, statsPaneArrow, image_check;
    @FXML
    private LineChart lineChart;
    @FXML
    private JFXToggleButton biprediction, psofilter;
    @FXML
    private JFXSlider avgfilter, gatimes;
    @FXML
    private JFXSpinner spin_clear, spin_generate;
    @FXML
    private JFXPasswordField password;

    private List<List<Double>> lists;

    /**
     * 实现Initializable接口需要重写的方法
     * 这里设置界面图片
     *
     * @param location  位置信息
     * @param resources 源文件
     */
    @Override
    public void initialize(URL location, ResourceBundle resources) {
        image_set.setImage(new Image("images/set.png"));
        image_user.setImage(new Image("images/user.png"));
        image_stats.setImage(new Image("images/stats.png"));
        image_power.setImage(new Image("images/power.png"));
        image_user_login.setImage(new Image("images/user_login.png"));
        setPaneArrow.setImage(new Image("images/arrow.png"));
        userPaneArrow.setImage(new Image("images/arrow.png"));
        statsPaneArrow.setImage(new Image("images/arrow.png"));
        image_check.setImage(new Image("images/Check.png"));
    }

    /**
     * 鼠标进入界面时初始化
     */
    public void init() {
        stage = (Stage) topbar.getScene().getWindow();
    }

    /**
     * press和drag构成鼠标拖动效果
     *
     * @param event 捕捉鼠标事件
     */
    public void press(MouseEvent event) {
        xOffset = event.getSceneX();
        yOffset = event.getSceneY();
    }

    public void drag(MouseEvent event) {
        stage.setX(event.getScreenX() - xOffset);
        stage.setY(event.getScreenY() - yOffset);
    }

    public void handleMouseEvents(MouseEvent mouseEvent) {
        if (mouseEvent.getTarget() == image_set) {
            if (setPane.isVisible()) {
                setPane.setVisible(false);
            } else {
                setPane.setVisible(true);
                userPane.setVisible(false);
                statsPane.setVisible(false);

            }
        } else if (mouseEvent.getTarget() == image_user) {
            if (userPane.isVisible()) {
                userPane.setVisible(false);
            } else {
                setPane.setVisible(false);
                userPane.setVisible(true);
                statsPane.setVisible(false);

            }
        } else {
            if (statsPane.isVisible()) {
                statsPane.setVisible(false);
            } else {
                setPane.setVisible(false);
                userPane.setVisible(false);
                statsPane.setVisible(true);
            }
        }
    }

    @FXML
    public void reset() {
        biprediction.setSelected(false);
        psofilter.setSelected(false);
        avgfilter.setValue(2);
        gatimes.setValue(2);
    }


    public void drawChart() throws Exception {
        if (image_check.isVisible()) {
            boolean isbiprediction = biprediction.isSelected();
            boolean ispsofilter = psofilter.isSelected();
            int avg_filter = (int) avgfilter.getValue();
            int ga_times = (int) avgfilter.getValue();
            Calculate calculate = new Calculate(isbiprediction, ispsofilter, avg_filter, ga_times);
            calculate.getCalculate();
        }
    }

    public void clearChart() {
        spin_clear.setProgress(0);
        lineChart.getData().clear();
        spin_clear.setProgress(1);
    }


    public void ispasswordright() {
        if (password.getText().equals("TY1996@86") || password.getText().equals("960206")) {
            image_check.setVisible(true);
        } else {
            image_check.setVisible(false);
        }
    }

    public void exit() {
        stage.close();
    }

    @AllArgsConstructor
    private class Calculate {
        private boolean isbiprediction;
        private boolean ispsofilter;
        private int avgfilter;
        private int gatimes;


        private void getCalculate() throws Exception {
            String readpath = "C:\\Users\\shinelon\\Desktop\\test.txt";
            spin_generate.setProgress(0);
            List<List<Double>> res = Start.start(gatimes, readpath);
            lists = _Main.run(res, isbiprediction, ispsofilter, avgfilter, readpath);
            XYChart.Series<String, Number> series1 = new XYChart.Series<>();
            XYChart.Series<String, Number> series2 = new XYChart.Series<>();
            for (int i = 1; i <= lists.get(0).size(); i += 1) {
                series1.getData().add(
                        new XYChart.Data<>(String.valueOf(i), lists.get(0).get(i - 1))
                );
                series2.getData().add(
                        new XYChart.Data<>(String.valueOf(i), lists.get(lists.size() - 1).get(i - 1))
                );
                spin_generate.setProgress(1.0 / lists.get(0).size() * i);
            }
            //在生成数据前清除之前的数据
            lineChart.getData().clear();
            lineChart.getData().add(series1);
            lineChart.getData().add(series2);

        }

    }

    public void excelGenerate() throws Exception {
        // List<List<Double>> res = Start.start();
        //_Main.run(res);
        String path = "C:\\Users\\shinelon\\Desktop\\基于粒子滤波的LSTM.xlsx";

        if (lists.size() != 0) {
            POI.init(path);
            POI.write(path, lists.get(0), lists.get(lists.size() - 1), 0);
        }
    }
}
