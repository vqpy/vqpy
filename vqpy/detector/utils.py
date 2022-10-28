
def onnx_inference(img_data, model_path):
    import onnxruntime as rt
    sess = rt.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: img_data})
    return outputs
