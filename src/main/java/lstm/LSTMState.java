package lstm;

class LSTMState {

    double[] g, i, f, o, s, h, bottom_diff_h, bottom_diff_s;

    public LSTMState(int mem_cell_cnt, int x_dim) {
        this.g = new double[mem_cell_cnt];
        this.i = new double[mem_cell_cnt];
        this.f = new double[mem_cell_cnt];
        this.o = new double[mem_cell_cnt];
        this.s = new double[mem_cell_cnt];
        this.h = new double[mem_cell_cnt];
        this.bottom_diff_h = new double[mem_cell_cnt];
        this.bottom_diff_s = new double[mem_cell_cnt];
    }
}