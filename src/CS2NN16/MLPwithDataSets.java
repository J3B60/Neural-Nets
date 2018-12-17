/**
 * 
 */
package CS2NN16;

import java.util.ArrayList;

/**
 * @author shsmchlr
 * Class of a multi layer perceptron network with training, unseen and validation data sets
 * MLP has hidden layer of sigmnoidally activated neurons and then output layer(s)
 * Such a network can learn using the training set, and be tested on teh unseen set
 * In addition, it can use the validation set to decide when to stop learning
 */
public class MLPwithDataSets extends MultiLayerNetwork {

	// HINT you may need extra variables here
	
	protected DataSet unseenData;			// unseen data set
	protected DataSet validationData;		// validation set : is set to null if that set is not being used
	private double previousValidationDataSSE;//Holds previous learn run validation SSE for comparison against the current SSE 
	private boolean validationSSEhasRisen; //For If condition, to prevent more learning
	private ArrayList<Double> lastTenValidationSSE; //Holds previous 10 SSEs for the validationDataSet
	
	/**
	 * Constructor for the MLP
	 * @param numIns			number of inputs	of hidden layer
	 * @param numOuts			number of outputs	of hidden layer
	 * @param data				training data set used
	 * @param nextL				next layer		
	 * @param unseen			unseen data set
	 * @param valid				validation data set
	 */
	MLPwithDataSets (int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL,
						DataSet unseen, DataSet valid) {
		super(numIns, numOuts, data, nextL);	// create the MLP
												// and store the data sets
		unseenData = unseen;
		validationData = valid;
	}

	/** 
	 * initialise network before learning ...
	 * Including some setup of variables for the doLearn Method
	 */
	public void doInitialise() {
		super.doInitialise();
		unseenData.clearSSELog();
		validationData.clearSSELog();
		// you may need extra initialisation here, both of the other data and any other variables
		lastTenValidationSSE = new ArrayList<Double>();//Initialise arraylist
		lastTenValidationSSE.clear();//Clear just in case errors happen
		for (int i = 0; i < 10; i++){//Setup empty array
			lastTenValidationSSE.add(0.0);//Setup of arraylist
		}
		validationSSEhasRisen = false;//Start Default with "SSE has not risen"
		previousValidationDataSSE = 1000;//Default value is set to a value which is beyond SSE range. This allows for the first run when there is nothing to compare to, to continue to else condition
	}
	/**
	 * present the data to the set and return a String describing results
	 * Here it returns the performance when the training, unseen (and if available) validation
	 * sets are passed - typically responding with SSE and if appropriate % correct classification
	 */
	public String doPresent() {
		String S;
		presentDataSet(trainData);
		S = "Train: " +  trainData.dataAnalysis();
		presentDataSet(unseenData);
		S = S + " Unseen: " + unseenData.dataAnalysis();
		if (validationData != null) {
			presentDataSet(validationData);
			S = S + " Valid: " + validationData.dataAnalysis();
		}
		return S;
	}

	/**
	 * learn training data, printing SSE at 10 of the epochs, evenly spaced
	 * if a validation set available, learning stops when SSE on validation set rises
	 * this check is done by summing SSE over 10 epochs
	 * @param numEpochs		number of epochs
	 * @param lRate			learning rate
	 * @param momentum		momentum
	 * @return				String with data about learning eg SSEs at relevant epoch
	 */
	public String doLearn (int numEpochs, double lRate, double momentum) {
		String s = "";
		if (validationData==null) s = super.doLearn(numEpochs, lRate, momentum);
					// if no validation set, just use normal doLearn
		else {
//			s = super.doLearn(numEpochs, lRate, momentum); 
			// delete the above and write and comment code to use validation
			if (validationSSEhasRisen) {//No learning
				return "";//nothing
			}
			else {
				double sumOfSSE = 0; //The sum of previous 10 SSEs
				double sumOfSSEAvg = 0; //Added another variable to keep things categorised for now, (if I ever need to refer back to sumOfSSE otherwise I could just write over the variable)
				int epochsSoFar = trainData.sizeSSELog(); //Maybe (to replace ct <= numEpochs)?
				for (int ct = 1; ct<=numEpochs; ct++) {//?Maybe or use the SSE one//Go through epochs and learn just like original doLearn
					super.learnDataSet(trainData, lRate, momentum);//"AdaptNetwork" -> learnDataSet//Learning with trainData
					//System.out.println(doPresent());//Pass the validationData//doPresent//dont need the string here
					doPresent();//Modified to not show all outputs to console (Less work for cpu)
					//presentDataSet(validationData);//This is what RJM suggests but he has not seen my doPresent, so I'll keep mine which presents info for all three sets (otherwise I only see Training and validation SSE changing but with mine I can see the SSE of the unseen dataset just as extra Info) Not used
					if (numEpochs<20 || ct % (numEpochs/10) == 0) // print appropriate number of times just like original doLearn
						s = s + addEpochString(ct+epochsSoFar) + " : Train " + trainData.dataAnalysis() + " : Unseen " + unseenData.dataAnalysis() + " : Valid " + validationData.dataAnalysis() + "\n";//Print to Interface
//					sumOfSSE = sumOfSSE + validationData.getSSE().get(ct);//Old, needs loop to check through all validationData SSEs
					lastTenValidationSSE.set((ct-1)%10, validationData.getTotalSSE());//Add the current validationSSE to the arraylist holding the 10 previous SSEs 
					if (ct % 10 == 0) {//Change this to epochsSoFar maybe?//Checks if the at epoch is a multiple of 10
						for (int cts = 0; cts < 10; cts++){ //Go through all items in arraylist
							sumOfSSE += lastTenValidationSSE.get(cts);//Sum of previous SSEs
						}
						sumOfSSEAvg = sumOfSSE/10;//Average of the 10 SSEs
//						if (previousValidationDataSSE == 0){//If its the first run then just set the average to be equal to the previousSSE then the next if statement shouldn't run
//							previousValidationDataSSE = sumOfSSEAvg;//First Setting of the previous validationSet
//						}
						if (sumOfSSEAvg > previousValidationDataSSE) {//If SSE has risen
							//StopLearning
							validationSSEhasRisen = true;//Prevents else section of doLearn
							return s + "Stop Learning after " + String.valueOf(epochsSoFar+ct) + " epochs\n"; //Output to Interface the stopped Epoch
						}
						else {
							//sumOfSSESavedInThisClass = sumOfSSE
							previousValidationDataSSE = sumOfSSEAvg;//New previousValidationDataSSE
							sumOfSSE = 0;//Reset
							sumOfSSEAvg = 0;
						}
					}
				}
			}
		}
		return s;											// return string showing learning
	}

}
