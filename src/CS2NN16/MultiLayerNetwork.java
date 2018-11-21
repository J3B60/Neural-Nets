/**
 * 
 */
package CS2NN16;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * @author shsmchlr
 * This a multi layer network, comprising a hidden layer of neurons with sigmoid activation
 * Followed by another layer with linear/sigmoid activation, or be another multi layer network
 * A layer is defined as a set of neurons which have the same inputs
 */
public class MultiLayerNetwork extends SigmoidLayerNetwork {
	LinearLayerNetwork nextLayer;			// this is the next layer of neurons
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 * @param nextL		the next layer in the network
	 */
	public MultiLayerNetwork(int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL) {
		super(numIns, numOuts, data);			// construct the current layer
		nextLayer = nextL;						// store link to next layer
	}
	/**
	 * calcOutputs of network. Find the Outputs of Current Layer then feed 
	 * to Next Layer to calculate its own outputs.
	 * @param nInputs	Passed to Current Layer
	 * 
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		// you write code here
		super.calcOutputs(nInputs); //calculate outputs using sigmoid calcOutputs function
		nextLayer.calcOutputs(super.outputs);// calculate nextLayers Outputs using current layers
	}
	
	/**
	 * outputsToDataSet of the network to the data set. Similar to LinearLayerNetwork
	 * function of same name. However pass next layers outputs.
	 * @param ct	selected item in the dataset
	 * @param d		the dataset
	 */
	protected void outputsToDataSet (int ct, DataSet d) {
		// you write code here ... note DataSet wants output(s) of final layer only
		d.storeOutputs(ct, nextLayer.outputs); //store next Layers outputs
	}
	
	/**
	 * find the deltas in the whole network. First by finding deltas of next 
	 * layer by using the errors arraylist. Then find 
	 * deltas of this layer by calculating the errors from the next layer.
	 *	@param errors 	Passed to Next Layer
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		// you write this
		nextLayer.findDeltas(errors); //Deltas of nextlayer
		super.findDeltas(nextLayer.weightedDeltas());//Deltas of current layer using errors calculated using nextlayer's info
	}
	
	/**
	 * change all the weights in the network, in this layer and the next. Using outputs from current layer as 
	 * Inputs to the nextLayer
	 * @param ins		array list of the inputs to the neuron
	 * @param learnRate	learning rate: change is learning rate * input * delta
	 * @param momentum	momentum constant : change is also momentun * change in weight last time
	 */
	protected void changeAllWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		// you write this
		super.changeAllWeights(ins, learnRate, momentum);//input all args
		nextLayer.changeAllWeights(super.outputs, learnRate, momentum);//Output from current as input
	}	
	
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	protected void setWeights (String[] wtsSplit) {
		super.setWeights(wtsSplit);					// copy relevant weights in this layer
		nextLayer.setWeights(Arrays.copyOfRange(wtsSplit, weights.size(), wtsSplit.length));
				// copy remaining strings in wtsSplit and pass to next layer
	}
	/**
	 * Load the weights with random values
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		super.setWeights(rgen);			// do so in this layer
		nextLayer.setWeights(rgen);		// and in next
	}
	/**
	 * return how many weights there are in the layered network
	 * @return Number of weights in the layered network
	 */
	public int numWeights() {
		// change this
		return super.numWeights() + nextLayer.numWeights();	//Outputs the same number of weights as input	
	}
	/**
	 * return the weights in the whole network as a string. Combine 
	 * weights from current layer string output with the next layer output
	 * @return the string
	 */
	public String getWeights() {
		// change this
		return "" + super.getWeights() + nextLayer.getWeights(); //Creates slightly different output than input due to rounding
	}
	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		super.doInitialise();					// initialise this layer 
		nextLayer.doInitialise();				// and then initialise next layer
	}
	
	/**
	 * function to test MLP on xor problem
	 */
	public static void TestXOR() {
		DataSet Xor = new DataSet("2 1 %.0f %.0f %.3f;x1 x2 XOR;0 0 0;0 1 1;1 0 1;1 1 0");
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Xor, new SigmoidLayerNetwork(2, 1, Xor));
		MLN.setWeights("0.862518 -0.155797 0.282885 0.834986 -0.505997 -0.864449 0.036498 -0.430437 0.481210");
		MLN.doInitialise();
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
		System.out.println(MLN.doLearn(2000, 0.4,  0.7));//EP,LR,Mo
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
	}
	/**
	 * function to test MLP on other non linear separable problem
	 */
	public static void TestOther() {
		DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Other, new SigmoidLayerNetwork(2, 2, Other));
			MLN.presentDataSet(Other);
			MLN.doInitialise();
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			System.out.println(MLN.doLearn(1000,  0.3,  0.5));
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
		
	}
	/**
	 * function to test MLP on other non linear separable problem using three layers
	 */
	public static void TestThree() {
		DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 4, Other,
										new MultiLayerNetwork (4, 3, Other,
												new SigmoidLayerNetwork(3, 2, Other)) );
			MLN.presentDataSet(Other);
			MLN.doInitialise();
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			System.out.println(MLN.doLearn(1000,  0.2,  0.6));
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
		
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
	//TestXOR();				// test MLP on the XOR problem
	TestOther();			// test MLP on the other problem
	//	TestThree();			// test that have 3 hidden layers
	}

}
