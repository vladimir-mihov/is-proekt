package main;

import mnist.Data;
import nn.NeuralNet;
import org.jblas.FloatMatrix;

import static org.jblas.MatrixFunctions.expi;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * @author Vladimir Mihov
 * @version 1.0
 */
public class App {

	private static Function<FloatMatrix,FloatMatrix> sigmoid = x -> expi(x.neg()).addi(1f).rdivi(1f);

	private static Function<FloatMatrix,FloatMatrix> softmax = x -> {
		x = expi(x.sub(x.max()));
		return x.divi(x.sum());
	};

	public static void main( String[] args ) throws IOException {
		String x = "hahaha";
		for( char i : x ) {
			System.out.println(i);
		}
	}
}