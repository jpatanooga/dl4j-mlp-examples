package tv.floe.dl4j.examples.mlp.baldridge.moon;

import java.util.Collections;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import tv.floe.dl4j.examples.mlp.baldridge.linear.BasicCSV_DataIterator;

/**
 * "Moon" Data
 * 
 *    https://github.com/jasonbaldridge/try-tf/blob/master/simdata/moon_data_train.jpg
 *    
 * Based on the data from Jason Baldridge:
 * 
 * 	https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 * 
 * 
 * 
 * 
 * @author Josh Patterson
 *
 */
public class MLP_Moon_NonLinear {


    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = iterations/5;
        double learningRate = 0.005;
        int nEpochs = 10;
        
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;
        
        
        DataSetIterator trainIter = new BasicCSV_DataIterator( "src/test/resources/data/baldridge/moon/moon_train.txt", "", 2, 50, 2000 );

        DataSetIterator testIter = new BasicCSV_DataIterator( "src/test/resources/data/baldridge/moon/moon_test.txt", "", 2, 200, 1000 );
        
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

        for ( int n = 0; n < nEpochs; n++) {
        	model.fit( trainIter );
        }
        
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);
            
        }
        

        System.out.println(eval.stats());
        System.out.println("****************Example finished********************");

    }		
	
}
