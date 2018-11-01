/**
 * 
 */
package CS2NN16;

import java.util.ArrayList;


/**
 * @author shsmchlr
 * This is a class for a layer of neurons with sigmoidal activation
 * All such neurons share the same inputs.
 */
public class SigmoidLayerNetwork extends LinearLayerNetwork {

	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 */
	public SigmoidLayerNetwork(int numIns, int numOuts, DataSet data) {
		super(numIns, numOuts, data);
	}
	
	/**
	 * calcOutputs of neuron. Polymorphism of calcOutputs from linear neural network
	 * takes the outputs from original and applies sigmoid equation and replaces original output with new
	 * sigmoid output. Arguments are the same as the original
	 * @param nInputs Arraylist with neuron inputs	
	 * 
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		// you write this
		double output;//Temporary double holder
		super.calcOutputs(nInputs);//Run the calcOutputs from the super class
		for (int i = 0; i < super.outputs.size(); i++){ //Loop for all outputs in arraylist - Edit the outputs to be sigmoidal
			output = (1/( 1 + Math.pow(Math.E, (-1*outputs.get(i)))));//Sigmoid equation using outputs
			super.outputs.set(i, output);//replaces old non sigmoid output with sigmoid output
		}
	}
	/**
	 * find deltas. Uses the same stucture as original however
	 * this function is not polymorphed and is a copy of the original with the
	 * sigmoid being set into the delta arraylist this has one loop rather than 2 if polymorphed 
	 * like the calcOutputs function and removes the need of setting delta twice.
	 * arguments are the same as the original function
	 * 
	 *	@param errors	error ArrayList from the dataset
	 */
	protected void findDeltas(ArrayList<Double> errors) {
			// write code to set delta as error * deriv activation
		for (int ct = 0; ct<errors.size(); ct++){//Loop to error arraylist size
			deltas.set(ct, errors.get(ct) * super.outputs.get(ct) * (1-super.outputs.get(ct)));//Copy so no reference + edited to find the sigmoid delta
		}
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// test with and or xor
		DataSet AndOrXor = new DataSet("2 3 %.0f %.0f %.3f;x1 x2 AND OR XOR;0 0 0 0 0;0 1 0 1 1;1 0 0 1 1;1 1 1 1 0");
		SigmoidLayerNetwork SN = new SigmoidLayerNetwork(2, 3, AndOrXor);
		SN.setWeights("0.2 0.5 0.3 0.3 0.5 0.1 0.4 0.1 0.2");
		SN.doInitialise();
		System.out.println(SN.doPresent());
		System.out.println("Weights " + SN.getWeights());
		System.out.println(SN.doLearn(1000,  0.15,  0.4));
		System.out.println(SN.doPresent());
		System.out.println("Weights " + SN.getWeights());

	}

}
