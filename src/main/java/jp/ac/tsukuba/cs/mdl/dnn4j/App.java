package jp.ac.tsukuba.cs.mdl.dnn4j;

import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.dataset.Cifar10Dataset;
import jp.ac.tsukuba.cs.mdl.dnn4j.dataset.Dataset;
import jp.ac.tsukuba.cs.mdl.dnn4j.dataset.MnistDataset;
import jp.ac.tsukuba.cs.mdl.dnn4j.layers.SigmoidWithLoss;
import jp.ac.tsukuba.cs.mdl.dnn4j.networks.Net;
import jp.ac.tsukuba.cs.mdl.dnn4j.networks.NeuralNet;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class App {

    public static void main(String[] args){

        /*
        データの準備
         */
        Dataset dataset = new MnistDataset();
        int[] inputShape = new int[]{
                dataset.getChannelSize(), dataset.getHeight(), dataset.getWidth()
        };
        NdArray xTrain = dataset.readTrainFeatures().reshape(
                dataset.getTrainSize(),
                dataset.getChannelSize(),
                dataset.getHeight(),
                dataset.getWidth()
        );
        NdArray tTrain = dataset.readTrainLabels();
        NdArray xTest = dataset.readTestFeatures().reshape(
                dataset.getTestSize(),
                dataset.getChannelSize(),
                dataset.getHeight(),
                dataset.getWidth()
        );
        NdArray tTest = dataset.readTestLabels();

        /*
        ネットワークアーキテクチャの設計
         */
        List<Map<String, Integer>> netArgList = constructNetArch();

        /*
        最適化アルゴリズムのパラメータの設定
         */
        Map<String, Double> optimizerParams = Maps.newHashMap();

        /*
        ネットワークの構成
         */
        Net net = new NeuralNet(
                inputShape,
                netArgList,
                new SigmoidWithLoss(),
                0.005 // weight decay lambda
        );

        /*
        訓練を行う
         */
        Trainer trainer = new TrainerImpl(
                net, // network
                xTrain, // input train data
                tTrain, // target train data
                xTest, // input test data
                tTest, // target test data
                10, // epoch num
                100, // mini batch size
                OptimizerType.ADAM, // optimizer
                optimizerParams, // optimizer parameter
                100, // evaluate batch size
                true // verbose
        );
        trainer.train();

        //ここからcsvファイルへ書き出す記述
        String trainLossFile = "trainLossCNNRMS_PROP.csv";
        String accuracyFile = "accuracyCNNRMS_PROP.csv";

        try {
            PrintWriter trainLoss = new PrintWriter(new BufferedWriter(new FileWriter(trainLossFile)));
            PrintWriter accuracy = new PrintWriter(new BufferedWriter(new FileWriter(accuracyFile)));

            for (int i = 0;i < trainer.getTrainLossList().size();i++){
                trainLoss.println(i+1 + "," + trainer.getTrainLossList().get(i));
            }

            for (int i = 0;i < trainer.getTrainAccList().size();i++){
                accuracy.println(i+1 + "," + trainer.getTrainAccList().get(i) + "," + trainer.getTestAccList().get(i));
            }

            trainLoss.close();
            accuracy.close();

        } catch (IOException e){
            e.printStackTrace();
        }

    }

    private static List<Map<String, Integer>> constructNetArch() {
        List<Map<String, Integer>> netArgList = new ArrayList<>();


/*        //課題1と2
        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FLATTEN)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 200)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.RELU) //1はシグモイド、2はReLU
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 100)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.RELU) //1はシグモイド、2はReLU
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 10)
                        .build()
        );*/


        //課題3 図1のネットワークを設計し、精度を比較
        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.CONVOLUTION)
                        .put(NetArgType.FILTER_NUM, 4)
                        .put(NetArgType.FILTER_SIZE, 3)
                        .put(NetArgType.STRIDE, 1)
                        .put(NetArgType.PADDING, 1)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.RELU)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.POOLING)
                        .put(NetArgType.FILTER_SIZE, 2)
                        .put(NetArgType.STRIDE, 2)
                        .put(NetArgType.PADDING, 0)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.CONVOLUTION)
                        .put(NetArgType.FILTER_NUM, 4)
                        .put(NetArgType.FILTER_SIZE, 3)
                        .put(NetArgType.STRIDE, 1)
                        .put(NetArgType.PADDING, 1)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.RELU)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.POOLING)
                        .put(NetArgType.FILTER_SIZE, 2)
                        .put(NetArgType.STRIDE, 2)
                        .put(NetArgType.PADDING, 0)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FLATTEN)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 50)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.RELU)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 10)
                        .build()
        );

        return netArgList;
    }
}
