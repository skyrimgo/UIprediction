package lstm;

class LSTMNode {

    LSTMState state;
    LSTMParam param;

    double[] s_prev;
    double[] h_prev;

    double[] xc;

    public LSTMNode(LSTMState state, LSTMParam param) {
        this.state = state;
        this.param = param;
    }

    public void bottom_data_is(double[] x, double[] s_prev, double[] h_prev) {
        if (s_prev == null)
            s_prev = LSTM.zero_like(this.state.s);
        if (h_prev == null)
            h_prev = LSTM.zero_like(this.state.h);

        this.s_prev = s_prev;
        this.h_prev = h_prev;

        // concatenate x(t) and h(t - 1)
        this.xc = LSTM.hstack(x, h_prev);
        this.state.g = LSTM.tanh(LSTM.WtxPlusBias(this.param.wg, xc, this.param.bg));
        this.state.i = LSTM.sigmoid(LSTM.WtxPlusBias(this.param.wi, xc, this.param.bi));
        this.state.f = LSTM.sigmoid(LSTM.WtxPlusBias(this.param.wf, xc, this.param.bf));
        this.state.o = LSTM.sigmoid(LSTM.WtxPlusBias(this.param.wo, xc, this.param.bo));

        this.state.s = LSTM.add(LSTM.mat(this.state.g, this.state.i), LSTM.mat(s_prev, this.state.f));
        this.state.h = LSTM.mat(this.state.s, this.state.o);

    }

    public void top_diff_is(double[] top_diff_h, double[] top_diff_s) {
        double[] ds = LSTM.add(top_diff_s, LSTM.mat(this.state.o, top_diff_h));
        double[] dot = LSTM.mat(this.state.s, top_diff_h);
        double[] di = LSTM.mat(this.state.g, ds);
        double[] dg = LSTM.mat(this.state.i, ds);
        double[] df = LSTM.mat(this.s_prev, ds);

        double[] di_input = LSTM.mat(LSTM.sigmoid_derivative(this.state.i), di);
        double[] df_input = LSTM.mat(LSTM.sigmoid_derivative(this.state.f), df);
        double[] do_input = LSTM.mat(LSTM.sigmoid_derivative(this.state.o), dot);
        double[] dg_input = LSTM.mat(LSTM.tanh_derivative(this.state.g), dg);

        this.param.wi_diff = LSTM.add(this.param.wi_diff, LSTM.outer(di_input, this.xc));
        this.param.wf_diff = LSTM.add(this.param.wf_diff, LSTM.outer(df_input, this.xc));
        this.param.wo_diff = LSTM.add(this.param.wo_diff, LSTM.outer(do_input, this.xc));
        this.param.wg_diff = LSTM.add(this.param.wg_diff, LSTM.outer(dg_input, this.xc));

        this.param.bi_diff = LSTM.add(this.param.bi_diff, di_input);
        this.param.bf_diff = LSTM.add(this.param.bf_diff, df_input);
        this.param.bo_diff = LSTM.add(this.param.bo_diff, do_input);
        this.param.bg_diff = LSTM.add(this.param.bg_diff, dg_input);

        double[] dxc = LSTM.zero_like(this.xc);
        dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wi), di_input));
        dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wf), df_input));
        dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wo), do_input));
        dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wg), dg_input));

        this.state.bottom_diff_s = LSTM.mat(ds, this.state.f);
        this.state.bottom_diff_h = LSTM.dim(dxc, this.param.x_dim, dxc.length);
    }
}