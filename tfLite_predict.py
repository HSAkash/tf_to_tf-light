import tflite_runtime.interpreter as tflite
import numpy as np
import argparse

class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, data_path):
        """
        input data path with extension .npy
        """
        try:
            data = np.load(data_path)

            self.interpreter.set_tensor(self.input_details[0]['index'], np.array([data], dtype=np.float32))
            self.interpreter.invoke()
            pred_prob = self.interpreter.get_tensor(self.output_details[0]['index'])
            self.interpreter.reset_all_variables()
            self.interpreter.close()
            arg_max_pred = pred_prob.argmax()
            return arg_max_pred
        except Exception as e:
            pass

        return None




if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Description of program')

    # Add arguments to the parser
    parser.add_argument('-m', '--model', type=str, help='tfLite model path with extension .tflite')
    parser.add_argument('-d', '--data', type=str, help='input data path with extension .npy')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    model_path = args.model
    data_path = args.data

    if model_path is None or data_path is None:
        parser.print_help()
        exit()

    model = TFLiteModel(model_path)
    pred = model.predict(data_path)
    print(pred)


