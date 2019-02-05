package nn;

import org.jblas.FloatMatrix;
import org.jblas.ranges.RangeUtils;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * Represents an entire neural network.
 * @author Vladimir Mihov
 * @version 1.0
 */
public class NeuralNet {
	/**
	 * Array containing all the layers.
	 */
	private final NNLayer[] layers;
	/**
	 * Learning rate of the nn.
	 */
	private final float learningRate;

	/**
	 * Creates the basic layout of the neural network.
	 * @implNote The input layer is not represented in the layers array.
	 * @param config Number of neurons for each layer.
	 * @param activations Activation functions for each layer.
	 * @param learningRate Learning rate of the neural network.
	 */
	public NeuralNet(int[] config, List<Function<FloatMatrix,FloatMatrix>> activations, float learningRate ) {
		this.learningRate = learningRate;

		int layerCount = config.length - 1;
		this.layers = new NNLayer[layerCount];

		for( int i = 0 ; i < layerCount ; ++i ) {
			this.layers[i] = new NNLayer( config[i], config[i+1], activations.get(i) );
		}
	}

	/**
	 * Trains the neural network for the specified amount of epochs.
	 * @param inputs All of the training data. Each row corresponds to one training sample.
	 * @param labels Labels for the training data.
	 * @param n Number of training epochs.
	 */
	public void epochs( FloatMatrix inputs, byte[] labels, int n ) {
		int batchSize = 100,
			batchCount = (int) Math.ceil( (double)inputs.getRows() / batchSize );

		FloatMatrix[] batches = new FloatMatrix[batchCount];
		byte[][] batchLabels = new byte[batchCount][];
		for( int i = 0 ; i < batchCount ; ++i ) {
			int firstBatchIndex = i * batchSize;
			batches[i] = getRows(inputs,firstBatchIndex,firstBatchIndex+batchSize);
			batchLabels[i] = Arrays.copyOfRange(labels,firstBatchIndex,firstBatchIndex+batchSize);
		}

		for( int e = 0 ; e < n ; ++e ) {
			for( int b = 0 ; b < batchCount ; ++b ) {
				batch( batches[b], batchLabels[b] );
			}
		}
	}

	/**
	 * Trains the neural network with the specified amount of inputs.
	 * @param inputs Some of the training data.
	 * @param labels Labels for the training data.
	 */
	private void batch( FloatMatrix inputs, byte[] labels ) {

	}

	/**
	 * Puts the specified input through the neural network.
	 * @param input Input vector.
	 * @return Output vector.
	 */
	public FloatMatrix test( FloatMatrix input ) {
		FloatMatrix result = input.dup();
		for( NNLayer layer : layers ) {
			result = layer.activate(result);
		}
		return result;
	}

	/**
	 * Wrapper around the FloatMatrix.getRows function to make it suitable.
	 * @implNote If b is out of bounds, instead of throwing an exception we return the tail of the matrix.
	 *
	 */
	private FloatMatrix getRows( FloatMatrix original, int from, int to ) {
		if( to > original.getRows() ) {
			to = original.getRows();
		}
		return original.getRows( RangeUtils.interval(from,to) );
	}
}
