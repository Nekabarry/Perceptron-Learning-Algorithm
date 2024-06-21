public class Pclass{
    private double[] weights;
    private double learningRate;

    public Pclass(int inputSize, double learningRate) {
        this.weights = new double[inputSize + 1];
        this.learningRate = learningRate;
        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }

    public int predict(double[] inputs) {
        double sum = weights[0]; // Bias weight
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i + 1];
        }
        return (sum >= 0) ? 1 : 0;
    }

    public void train(double[][] trainingInputs, int[] trainingLabels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingInputs.length; i++) {
                int prediction = predict(trainingInputs[i]);
                int error = trainingLabels[i] - prediction;

                weights[0] += learningRate * error;
                for (int j = 0; j < trainingInputs[i].length; j++) {
                    weights[j + 1] += learningRate * error * trainingInputs[i][j];
                }
            }
        }
    }

    public static void main(String[] args) {
        double[][] trainingInputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        int[] trainingLabels = {0, 0, 0, 1};

        Pclass perceptron = new Pclass(2, 0.1);
        perceptron.train(trainingInputs, trainingLabels, 1000);

        System.out.println("Predictions:");
        for (double[] inputs : trainingInputs) {
            System.out.println(perceptron.predict(inputs));
        }
    }
}
