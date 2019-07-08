/***************************************************************************************
  CS540 - Section 2
  Homework Assignment 5: Naive Bayes

  NBClassifierImpl.java
  This is the main class that implements functions for Naive Bayes Algorithm!
  ---------
 *Free to modify anything in this file, except the class name 
  	You are required:
  		- To keep the class name as NBClassifierImpl for testing
  		- Not to import any external libraries
  		- Not to include any packages 
 *Notice: To use this file, you should implement 2 methods below.

	@author: TA 
	@date: April 2017
 *****************************************************************************************/

import java.util.ArrayList;
import java.util.List;


public class NBClassifierImpl implements NBClassifier {

	private int nFeatures; 		// The number of features including the class 
	private int[] featureSize;	// Size of each feature
	private List<List<Double[]>> logPosProbs;	// parameters of Naive Bayes
	
	private double[] priorProb = new double[2]; // Index 0 for negClassProb
	                                            // Index 1 for posClassProb

	/**
	 * Constructs a new classifier without any trained knowledge.
	 */
	public NBClassifierImpl() 
	{

	}

	/**
	 * Construct a new classifier 
	 * 
	 * @param int[] sizes of all attributes
	 */
	public NBClassifierImpl(int[] features) 
	{
		this.nFeatures = features.length;

		// initialize feature size
		this.featureSize = features.clone();

		this.logPosProbs = new ArrayList<List<Double[]>>(this.nFeatures);	
	}


	/**
	 * Read training data and learn parameters
	 * 
	 * @param int[][] training data
	 */
	@Override
	public void fit(int[][] data) 
	{
		//	TODO
		double negClassCount = 0.0; 
		double posClassCount = 0.0;
		double negClassProb = 0.0;
		double posClassProb = 0.0;
		// Calculate the prior probabilities. Account for Add-1 smoothing
		for (int i = 0; i < data.length; ++i)
		{
			if (0 == data[i][data[i].length - 1])
			{
				++negClassCount;
			}
			else
			{
				++posClassCount;
			}
		}
		negClassProb = (negClassCount + 1) / 
				((double) data.length + (double) featureSize[featureSize.length - 1]);
		posClassProb = (posClassCount + 1) / 
				((double) data.length + (double) featureSize[featureSize.length - 1]);
		// Store the prior probabilities in a global DS
		// Note: these are not the final values to be stored
		priorProb[0] = negClassProb;
		priorProb[1] = posClassProb;
		// Build the Bayesian Network structure
		for (int i = 0; i < featureSize.length - 1; ++i)
		{
			logPosProbs.add(new ArrayList<Double[]>());
			for (int j = 0; j < featureSize[i]; ++j)
			{
				Double[] newArray = new Double[2];
				newArray[0] = 0.0;
				newArray[1] = 0.0;
				logPosProbs.get(i).add(newArray);
			}
		}
		// Count number of instances having X = x and Y = y. Store in logPosProbs.
		// Note: these are not the final values to be stored.
		// Iterate through each instance
		for (int i = 0; i < data.length; ++i)
		{
			int classVal = data[i][data[i].length - 1];
			// Iterate through each feature and update value in logPosProbs
			for (int j = 0; j < featureSize.length - 1; ++j)
			{
				int attributeVal = data[i][j];
				logPosProbs.get(j).get(attributeVal)[classVal]++;
			}
		}
		// Calculate the conditional probabilities (with logs) for each index in logPosProbs
		// Iterate through the attributes of logPosProbs
		for (int i = 0; i < logPosProbs.size(); ++i)
		{
			// Iterate through the values of each attribute
			for (int j = 0; j < logPosProbs.get(i).size(); ++j)
			{
				// Iterate through the 2 conditional probabilities
				for (int k = 0; k < 2; ++k)
				{
					double sizeOfAttribute = featureSize[i];
					// Calculate the raw probabilities. Account for Add-1 smoothing
					if (0 == k)
					{
						logPosProbs.get(i).get(j)[k] = (logPosProbs.get(i).get(j)[k] + 1) / 
								(negClassCount + sizeOfAttribute);
					}
					else
					{
						logPosProbs.get(i).get(j)[k] = (logPosProbs.get(i).get(j)[k] + 1) / 
								(posClassCount + sizeOfAttribute);
					}
					// Compute the log of conditional probabilities
					logPosProbs.get(i).get(j)[k] = Math.log(logPosProbs.get(i).get(j)[k]);
				}
			}
		}
		// Compute the log of the class probabilities
		priorProb[0] = Math.log(priorProb[0]);
		priorProb[1] = Math.log(priorProb[1]);
	}

	/**
	 * Classify new dataset
	 * 
	 * @param int[][] test data
	 * @return Label[] classified labels
	 */
	@Override
	public Label[] classify(int[][] instances) {

		int nrows = instances.length;
		Label[] yPred = new Label[nrows]; // predicted data
		//	TODO
		// Iterate through each instance
		for (int i = 0; i < nrows; ++i)
		{
			double negativeSum = 0.0;
			double positiveSum = 0.0;
			// Iterate through each attribute
			for (int j = 0; j < instances[i].length - 1; ++j)  // CHANGED
			{
				int attributeVal = instances[i][j];
				negativeSum += logPosProbs.get(j).get(attributeVal)[0];
				positiveSum += logPosProbs.get(j).get(attributeVal)[1];
			}
			negativeSum += priorProb[0];
			positiveSum += priorProb[1];
			// Store the classification in yPred based on calculations
			if (positiveSum >= negativeSum)
			{
				yPred[i] = Label.Positive;
			}
			else
			{
				yPred[i] = Label.Negative;
			}
		}
		return yPred;
	}
}