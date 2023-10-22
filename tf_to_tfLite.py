import argparse

# Convert a tf model to a tfLite model
def convert_model(saved_model_dir):

    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    return tflite_model


# save lite model
def save_model(tflite_model, model_path):
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Description of program')

    # Add arguments to the parser
    parser.add_argument('-s', '--source', type=str, help='tf model directory')
    parser.add_argument('-d', '--destination', type=str, help='tfLite model path with extension .tflite')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    tf_model_dir = args.source
    lite_model_apth = args.destination

    if tf_model_dir is None or lite_model_apth is None:
        parser.print_help()
        exit()
        


    tflite_model = convert_model(tf_model_dir)
    save_model(tflite_model, lite_model_apth)