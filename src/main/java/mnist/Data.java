package mnist;

import main.Test;
import org.jblas.FloatMatrix;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Parses the mnist files and stores the data.
 */
public class Data {
	private final FloatMatrix mnistTrainData;
	private final byte[] mnistTrainLabels;
	private final FloatMatrix mnistTestData;
	private final byte[] mnistTestLabels;

	private static final int numTrainImages = 60000;
	private static final int numTestImages = 10000;
	private static final int pixels = 784;

	public Data() throws IOException {
		mnistTrainData = new FloatMatrix(numTrainImages,pixels);
		mnistTrainLabels = new byte[numTrainImages];

		mnistTestData = new FloatMatrix(numTestImages,pixels);
		mnistTestLabels = new byte[numTestImages];

		try(
				DataInputStream trainData = new DataInputStream( new FileInputStream("mnist/train-images") );
				DataInputStream testData = new DataInputStream( new FileInputStream("mnist/test-images") );
				DataInputStream trainLabels = new DataInputStream( new FileInputStream("mnist/train-labels") );
				DataInputStream testLabels = new DataInputStream( new FileInputStream("mnist/test-labels") )
		) {
			trainData.skipBytes(16); testData.skipBytes(16);
			trainLabels.skipBytes(8); testLabels.skipBytes(8);

			int readTrainLabels = trainLabels.read(mnistTrainLabels);
			int readTestLabels = testLabels.read(mnistTestLabels);
			assert readTrainLabels == numTestImages && readTestLabels == numTestImages;

			read( trainData, numTrainImages, mnistTrainData );
			read( testData, numTestImages, mnistTestData );
		}
	}

	private void read( DataInputStream stream, int images, FloatMatrix output ) throws IOException {
		byte[] buf = new byte[pixels];
		for( int i = 0 ; i < images ; ++i ) {
			int readBytes = stream.read(buf);
			assert readBytes == pixels;

			float[] data = new float[pixels];
			for( int j = 0 ; j < pixels ; ++j ) {
				data[j] = (buf[j] & 0xFF) * 0.046875f - 6.0f;
			}

			output.putRow(i,new FloatMatrix(data).transpose());
		}
	}

	public FloatMatrix getTrainData() {
		return mnistTrainData;
	}

	public byte[] getTrainLabels() {
		return mnistTrainLabels;
	}

	public FloatMatrix getTestData() {
		return mnistTestData;
	}

	public byte[] getTestLabels() {
		return mnistTestLabels;
	}
}
