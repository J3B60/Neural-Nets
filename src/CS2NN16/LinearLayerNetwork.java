/**
 * 
 */
package CS2NN16;

import java.util.ArrayList;
import java.util.Random;

/**
 * @author shsmchlr
 * This is a class for a layer of neurons with linear activation
 * All such neurons share the same inputs.
 */
public class LinearLayerNetwork {
	/**
	 * data are arraylists of weights and the change in weights
	 * and of the outputs and deltas 
	 * and also how many inputs, neurons and weights
	 * also has data set used with network
	 */
	protected ArrayList<Double> weights;
	protected ArrayList<Double> changeInWeights;
	protected ArrayList<Double> outputs;
	protected ArrayList<Double> deltas;
	protected int numInputs, numNeurons, numWeights;
	protected DataSet trainData;
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 */
	public LinearLayerNetwork(int numIns, int numOuts, DataSet data) {
		numInputs = numIns;							// store number inputs
		numNeurons = numOuts;						// and of outputs in object
		numWeights = (numInputs + 1) * numNeurons;	// for convenience calculate number of weights
				// each neuron has numInputs + 1 weights (+1 because of bias weight)

		weights = new ArrayList<Double>();			// create array list for weights
		changeInWeights = new ArrayList<Double>();	// and for the change in weights
		outputs = new ArrayList<Double>();			// create array list for outputs
		deltas = new ArrayList<Double>();			// and for the change in deltas
		for (int ct=0; ct<numWeights; ct++) {		// for each weight
			weights.add(0.0);						// add next weight as 0
			changeInWeights.add(0.0);				// add next change in weight, value 0
		}
		for (int ct=0; ct<numNeurons; ct++) {		// for each neuron
			outputs.add(0.0);						// add a zero output
			deltas.add(0.0);						// add a zero delta
		}
		trainData = data;							// remember data set used for training
	}
	/**
	 * calcOutputs of neuron
	 * @param nInputs	arraylist with the neuron inputs
	 * Calculates weighted sum being weight(0) + inputs(0..n) * weights(1..n+1)
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		int wtIndex = 0;									// used to index weights in order
		double output;
		for (int neuronct = 0; neuronct<numNeurons; neuronct++) {
			output = weights.get(wtIndex++);						// start with bias weight( * 1)
			for (int inputct=0; inputct<numInputs; inputct++) 		// for remaining weights
				 output += nInputs.get(inputct) * weights.get(wtIndex++);
											// add weight*appropriate input and move to next weight
			outputs.set(neuronct, output);			// set calculated output as the neuron output
		}				
	}
	/**
	 * outputsToDataSet to the given data set
	 * @param ct	which item in the data set
	 * @param d		the data set
	 */
	protected void outputsToDataSet (int ct, DataSet d) {
		d.storeOutputs(ct, outputs);							// just store outputs in data set
	}
	
	/**
	 * compute outputs of network by passing it each item in data set in turn, 
	 * these outputs are put back into the data set
	 * @param d	data set
	 */
	public void presentDataSet(DataSet d) {
		for (int ct=0; ct < d.numInSet(); ct++) {			// for each item in data set
			calcOutputs(d.getIns(ct));						// calculate output
			outputsToDataSet(ct, d);							// and put in data set
		}
	}
	
	/**
	 * find deltas using list of errors. Copies the error list into a list of deltas
	 *	@param errors	error ArrayList from the dataset
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		// write code to copy each error into delta list
		for (int ct = 0; ct<errors.size(); ct++){//Loop to error arraylist size
			deltas.set(ct, errors.get(ct));//Copy so no reference
		}
	}	
	
	/**
	 * return index into weights array list of the wWeight'th weight of the wNeuron'th neuron
	 * since the array is one dimensional where every 3 weights belong to 1 neuron. 
	 * @param wNeuron The selected Neuron - should be called for all neurons	
	 * @param wWeight The Selected Weight - should be called for all weights
	 * @return	Returns the position in the array the neuron's weight is located
	 */
	private int weightIndex (int wNeuron, int wWeight) {
		return (numInputs+1)*wNeuron + wWeight;			// change this, There are 3 weights for each Neuron, so skips every 3 weights to get to specific neuron then finds the weight no -1 since both Neuron and weights start from 0//CHANGED TO numInuts rather than 3 (because num of weights is not constant)
//		return 0;
	}
	
	/**
	 * change a given weight by the selected neuron and weight and calculating the change in weight and then adding to current weight
	 * Arguments are the currently selected Neuron number, selected weights number, theInput and from dolearn, the Learnrate and momentum. This function should
	 * be called for all neurons and weights
	 * @param wNeuron	Currently selected Neuron Number - should be called for all Neurons
	 * @param wWeight	Currently selected weight Number - should be called for all weights
	 * @param theIn		The Input from dataset
	 * @param learnRate	The learnrate from dolearn
	 * @param momentum	The momentum from doLearn
	 */
	private void changeOneWeight(int wNeuron, int wWeight, double theIn, double learnRate, double momentum) {
		// first calculate index of weight, then the change in weight, then change the weight
		int CalculatedWeightIndex = weightIndex(wNeuron, wWeight); //Find the correct weight from correct neuron
		changeInWeights.set(CalculatedWeightIndex, theIn * deltas.get(wNeuron) * learnRate + changeInWeights.get(CalculatedWeightIndex) * momentum);//Calculate the Change in weight
		weights.set(CalculatedWeightIndex, weights.get(CalculatedWeightIndex) +  changeInWeights.get(CalculatedWeightIndex));//Set new weight by adding change in weight to original
		
	}
	
	/**
	 * change all the weights in layer. This method goes through each weight in each neuron and changes its weight
	 * by calling the ChangeOneWeight function. Bias weight is set to 1. Arguments
	 * are an arraylist of inputs from the dataset. Learnrate and momentum is from doLearn.
	 * @param ins		
	 * @param learnRate	
	 * @param momentum	
	 */
	protected void changeAllWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		// you write this
		int numWeightsEachNeuron = numInputs+1; //Weights equal to Number of inputs +1 because of bias weight
		double theIn; //The input as a double
		for (int nct = 0; nct < numNeurons; nct++){ //For each neuron
			for (int wct = 0; wct < numWeightsEachNeuron; wct++){ //For each weight of each neuron
				if (wct == 0) theIn =1; else theIn = ins.get(wct -1);// if the bias then 1 else get weight from the neuron
				changeOneWeight(nct, wct, theIn, learnRate, momentum);//pass information to change each weight
			}
		}
	}
	
	/**
	 * adapt the network, by inputting each item from the data set in turn, calculating
	 * the output, the error and delta, and adjusting all the weights
	 * @param d			data set
	 * @param learnRate	learning rate constant
	 * @param momentum	momentum constant
	 */
	public void learnDataSet(DataSet d, double learnRate, double momentum) {
		for (int ct=0; ct < d.numInSet(); ct++) {				// for each item in set
			calcOutputs(d.getIns(ct));							// calc outputs
			outputsToDataSet(ct, d);								// put in data set
			findDeltas(d.getErrors(ct));						// calc deltas, from the errors
			changeAllWeights(d.getIns(ct), learnRate, momentum);// change the weights
		}
		d.addToSSELog();
	}
	
	/**
	 * return the array list containing the outputs of this layer of neurons
	 * @return
	 */
	protected ArrayList<Double> getOutputs() {
		return outputs;
	}
	
	/**
	 * Calculate the errors in the previous layer, being the deltas in this layer * associated weights
	 * this is used in the back propagation algorithm. For each neuron, multiply the neurons delta by each weight and
	 * feed this to hiddenlayer as errors
	 * 
	 * @return	arraylist of errors passed hidden layer
	 */
	public ArrayList<Double> weightedDeltas() {
		ArrayList<Double> wtDeltas = new ArrayList<Double>();	// create array for answer
			// now add suitable 'errors'
		wtDeltas.clear();//Initialise 
		for (int ct = 0; ct < numInputs; ct++) { //Setup Arraylist similar to how Constructor does (not including the bias), numInputs = num neurons in hidden layer
			wtDeltas.add(0.0);
		}
		int index = 0;//index in 
		double ans =0;//store answer for wtDelta arraylist
		for (int weight = 0; weight<numInputs; weight++){ //Eg(for first weight run) First weight
			ans=0;//Reset ans for next hidden layer neuron delta
			for (int neuron = 0; neuron<numNeurons; neuron++){//Eg Of each neuron
				ans += deltas.get(neuron) * weights.get(weightIndex(neuron, weight+1));//Sum to make delta, skip bias
			}
			wtDeltas.set(index, ans);//set in weighted deltas arraylist
			index++;//Position in weighted deltas arraylist
		}
		return wtDeltas;
	}
	
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	protected void setWeights (String[] wtsSplit) {
		for (int ct=0; ct<weights.size(); ct++) weights.set(ct, Double.parseDouble(wtsSplit[ct])); 
	}			// for each item, set weight by converting string to double
	/**
	 * Load the weights with the values in the String wts
	 * @param wts
	 */
	public void setWeights (String wts) {
		setWeights(wts.split(" "));			// split string into array of string and so set weights
	}
	/**
	 * Load the weights with random values in range -1 to 1
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		for (int ct=0; ct<weights.size(); ct++) weights.set(ct,2.0*rgen.nextDouble() - 1);
	}
	/**
	 * return how many weights there are in the neuron
	 * @return
	 */
	public int numWeights() {
		return weights.size();
	}
	/**
	 * return all the weights in the layer to 5 decimal places
	 * as a string each separated by spaces. Used to output to console
	 * @return String of all weights in 0.00000 format
	 */
	public String getWeights() {
		String s = "";									// set string to empty
				// add code here
		for (int ct = 0; ct<weights.size(); ct++) {//Loop from 0 until end of weights Arraylist Note: ct used for consistency with other code
			s += String.format("%.5f", weights.get(ct)) + " ";//Add to s weight to 5 decimal places
		}
			
		return s;										// return the result
	}
	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		for (int ct=0; ct<changeInWeights.size(); ct++) changeInWeights.set(ct, 0.0);
				// set the change in weights to be 0
		trainData.clearSSELog();
	}
	/**
	 * present the data to the network and return string describing result
	 * @return
	 */
	public String doPresent() {
		presentDataSet(trainData);
		return trainData.toString(true, true) + "\nOver Set : " + trainData.dataAnalysis()+"\n";
	}

	
	/**
	 * create string which says Epoch then adds the actual epoch in a fixed width field
	 * @param epoch
	 * @return
	 */
	protected String addEpochString (int epoch) {
		return "Epoch " + String.format("%4d", epoch);
	}
	
	/**
	 * get network to learn for numEpochs
	 * @param numEpochs		number of epochs to learn
	 * @param lRate			learning rate
	 * @param momentum		momentum
	 * @return				String with data about learning eg SSEs at relevant epochs
	 * 						At each epoch if numEpochs low, or do so at 10 of the epochs
	 */
	public String doLearn (int numEpochs, double lRate, double momentum) {
		int epochsSoFar = trainData.sizeSSELog();		// SSE log indicates how many epochs so far
		String s = "";
		for (int ct=1; ct<=numEpochs; ct++) {			// for n epochs
			learnDataSet(trainData, lRate, momentum);	// present data and adapt weights
			if (numEpochs<20 || ct % (numEpochs/10) == 0) // print appropriate number of times
				s = s + addEpochString(ct+epochsSoFar) + " : " + trainData.dataAnalysis()+"\n";
		}				// Epoch, and SSE, and if appropriate % correctly classified
		return s;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// Test network on example data set
		DataSet AndOrXor = new DataSet("2 3 %.0f %.0f %.3f;x1 x2 AND OR XOR;0 0 0 0 0;0 1 0 1 1;1 0 0 1 1;1 1 1 1 0");
		LinearLayerNetwork LN = new LinearLayerNetwork(2, 3, AndOrXor);
		LN.setWeights("0.2 0.5 0.3 0.3 0.5 0.1 0.4 0.1 0.2");
		LN.doInitialise();
		System.out.println(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());
		System.out.println(LN.doLearn(10,  0.1,  0.3));
		System.out.println(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());
		
	}

}
