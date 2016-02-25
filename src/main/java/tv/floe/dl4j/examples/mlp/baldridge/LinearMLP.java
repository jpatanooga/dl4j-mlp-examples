package tv.floe.dl4j.examples.mlp.baldridge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
//import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;

public class LinearMLP {


    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        final int numRows = 28;
        final int numColumns = 28;
        //int outputNum = 2;
        int numSamples =10000;
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = iterations/5;
        int splitTrainNum = (int) (batchSize*.8);
        double learningRate = 0.01;
        //int iterations = 1;
        //Number of epochs (full passes of the data)
        int nEpochs = 3;
        
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;
        
        
        DataSet mnist;
        //SplitTestAndTrain trainTest;
        DataSet trainInput;
        //List<INDArray> testInput = new ArrayList<>();
        //List<INDArray> testLabels = new ArrayList<>();

        //log.info("Load data....");
        DataSetIterator trainIter = new BasicCSV_DataIterator( "src/test/resources/data/baldridge/linear/linear_train.txt", "", 2, 50, 1000 );

        DataSetIterator testIter = new BasicCSV_DataIterator( "src/test/resources/data/baldridge/linear/linear_test.txt", "", 2, 200, 200 );
        
        //log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(iterations)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(learningRate)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list(2)
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
        .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("sigmoid").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
        .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

       // log.info("Train model....");
        //model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        /*
        while(trainIter.hasNext()) {
            mnist = trainIter.next();
            trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }
*/
        
        for ( int n = 0; n < nEpochs; n++) {
        	model.fit( trainIter );
        }
        
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        /*
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        */
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

        //    evaluation.evalTimeSeries(lables,predicted,outMask);
            eval.eval(lables, predicted);
            
        }
        

        System.out.println(eval.stats());
        System.out.println("****************Example finished********************");

    }	
	
}
