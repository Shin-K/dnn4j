package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

public class FullyConnect implements Layer {

    private NdArray weight;

    private NdArray bias;

    private NdArray input;

    private NdArray weightGrad;

    private NdArray biasGrad;

    public FullyConnect(NdArray weight, NdArray bias) {
        this.weight = weight;
        this.bias = bias;
    }

    public NdArray getWeight() {
        return weight;
    }

    public NdArray getBias() {
        return bias;
    }

    public NdArray getWeightGrad() {
        return weightGrad;
    }

    public NdArray getBiasGrad() {
        return biasGrad;
    }

    public void setWeight(NdArray weight) {
        this.weight = weight;
    }

    public void setBias(NdArray bias) {
        this.bias = bias;
    }

    @Override
    public NdArray forward(NdArray input) {
        this.input = input;
        return input.dot(this.weight).add(this.bias);
    }

    @Override
    public NdArray backward(NdArray dout) {
        this.weightGrad = this.input.transpose().dot(dout);
        //this.input.shape()[0] -> ミニバッチサイズ
        this.biasGrad = dout.transpose().dot(NumJ.ones(this.input.shape()[0])).reshape(1, bias.size());
        return dout.dot(this.weight.transpose());
    }
}
