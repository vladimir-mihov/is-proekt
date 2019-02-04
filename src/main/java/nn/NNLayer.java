package nn;

import org.jblas.FloatMatrix;

import java.util.function.Function;

/**
 * Represents a single layer in a neural network.
 * @author Vladimir Mihov
 * @version 1.0
 */
class NNLayer {

	/**
	 * Weights matrix.
	 * @implNote Each column contains the weights of a single neuron.
	 */
	private final FloatMatrix weights;

	/**
	 * Biases vector.
	 */
	private final FloatMatrix biases;

	/**
	 * Activation function of the layer.
	 */
	private final Function<FloatMatrix,FloatMatrix> activation;

	/**
	 * @param inputs Number of input nodes to the layer.
	 * @param neurons Number of neurons in the layer.
	 * @param activation Activation function.
	 * @implNote The weights get initialized in the [-0.05,0.05] range randomly.
	 * @implNote The biases get initialized to 0.
	 */
	NNLayer(int inputs, int neurons, Function<FloatMatrix, FloatMatrix> activation) {
		this.weights = FloatMatrix.rand(inputs,neurons).divi(10).subi(0.05f);
		this.biases = FloatMatrix.zeros(1,neurons);
		this.activation = activation;
	}

	FloatMatrix getWeights() {
		return weights;
	}

	FloatMatrix getBiases() {
		return biases;
	}

	FloatMatrix activate( FloatMatrix inputs ) {
		return activation.apply( inputs.mmul(weights).addiRowVector(biases) );
	}
}