import onnx
from onnx import helper

# Load the original ONNX model
model_path = './pretrained/webface600_r50.onnx'
model = onnx.load(model_path)

# Assuming the output tensor's name is known (e.g., '683')
output_tensor_name = '683'  # Change this to your actual output tensor's name

# Find the output tensor in the graph and modify its shape
for output in model.graph.output:
    if output.name == output_tensor_name:
        # Change the first dimension to None to indicate a dynamic batch size
        output.type.tensor_type.shape.dim[0].dim_param = 'None'

# Save the modified model
modified_model_path = './pretrained/webface600_r50_fixoutput.onnx'
onnx.save(model, modified_model_path)

print(f"Modified model saved to {modified_model_path}")
