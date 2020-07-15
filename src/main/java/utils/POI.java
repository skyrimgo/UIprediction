package utils;

import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.util.List;

public class POI {
    //创建并初始化表格
    public static void init(String path) throws Exception {
        File file = new File(path);
        //存在则删除
        if (file.exists()) {
            file.delete();
        }
        POI.createExcel(path);
    }

    //创建Excel文件
    private static void createExcel(String path) throws Exception { //创建Excel文件对象
        XSSFWorkbook wb = new XSSFWorkbook(); //用文件对象创建sheet对象
        XSSFSheet sheet1 = wb.createSheet("双向"); //用sheet对象创建行对象
        sheet1.createRow(0); //创建单元格样式
        //XSSFRow row1 = sheet1.createRow(0); //创建单元格样式
        //List<XSSFRow> row_list = new ArrayList<>();
        //row_list.add(row1);

        FileOutputStream output = new FileOutputStream(path);
        wb.write(output);
        output.flush();
        output.close();
        wb.close();
    }

    //写入表格操作
    public static void write(String path, List<Double> l, List<Double> list, int index) throws IOException {
        // 1.读取Excel文档对象
        XSSFWorkbook xssfWorkbook = new XSSFWorkbook(new FileInputStream(path));
        // 2.获取要解析的表格（第一个表格）
        XSSFSheet sheet = xssfWorkbook.getSheetAt(index);
        // 3.在第row列写入数据

        for (int i = 0; i < list.size(); i++) {
            FileOutputStream outputStream = new FileOutputStream(path);
            XSSFRow xssfRow = sheet.createRow(i);
            XSSFCell xssfCell = xssfRow.createCell(0);
            xssfCell.setCellValue(l.get(i));
            xssfCell = xssfRow.createCell(1);
            xssfCell.setCellValue(list.get(i));
            xssfWorkbook.write(outputStream);
            outputStream.flush();
            outputStream.close();
        }

    }
}