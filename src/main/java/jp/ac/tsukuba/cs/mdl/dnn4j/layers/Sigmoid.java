package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

public class Sigmoid implements Layer {
    private NdArray input;

    @Override
    public NdArray forward(NdArray input) {
        this.input = input;
        return input.elementwise(x -> sigmoid(x));
    }

    @Override
    public NdArray backward(NdArray dout) {
        this.input = this.input.elementwise(x -> sigmoid(x) * (1 - sigmoid(x)));
        return this.input.mul(dout);
    }

    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
}
