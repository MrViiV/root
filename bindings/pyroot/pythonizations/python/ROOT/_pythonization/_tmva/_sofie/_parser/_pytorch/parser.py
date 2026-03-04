import os
import time

from .layers.elu import MakePyTorchELU

# Map from ONNX node types to their parsing functions
mapPyTorchNode = {
    "onnx::Elu": MakePyTorchELU,
}

def _node_get(node, key):
    """Helper to get node attribute without depending on onnx submodule."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

class PyTorch:

    @staticmethod
    def Parse(filename, input_shapes, input_dtypes=None):
        """
        Parse a PyTorch .pt model file into a SOFIE RModel.

        Parameters:
        filename (str): Path to the .pt model file
        input_shapes (list of list): Shapes of input tensors e.g. [[2, 16]]
        input_dtypes (list of str, optional): Defaults to all 'float'.

        Returns:
        RModel: Parsed SOFIE RModel object
        """
        import torch
        from ROOT.TMVA.Experimental import SOFIE
        from torch.onnx.utils import _model_to_graph

        if not os.path.exists(filename):
            raise RuntimeError("Model file {} not found!".format(filename))

        if input_dtypes is None:
            input_dtypes = ["float"] * len(input_shapes)

        sep = "\\" if os.name == "nt" else "/"
        isep = filename.rfind(sep)
        filename_nodir = filename[isep + 1:] if isep != -1 else filename

        parsetime = time.asctime(time.gmtime(time.time()))
        rmodel = SOFIE.RModel.RModel(filename_nodir, parsetime)

        print("PyTorch Python Parser: parsing model", filename)

        # Load and prepare model
        model = torch.jit.load(filename)
        model.cpu()
        model.eval()

        # Build dummy inputs and get ONNX graph
        dummy_inputs = [torch.rand(*shape) for shape in input_shapes]
        graph, weights = _model_to_graph(model, dummy_inputs)

        # Parse operators
        for node in graph.nodes():
            node_type = node.kind()

            attr_names = [x for x in node.attributeNames()]
            node_data = {
                "nodeType":       node_type,
                "nodeAttributes": {k: _node_get(node, k) for k in attr_names},
                "nodeInputs":     [x.debugName() for x in node.inputs()],
                "nodeOutputs":    [x.debugName() for x in node.outputs()],
                "nodeDType":      [x.type().scalarType() for x in node.outputs()],
            }

            parse_fn = mapPyTorchNode.get(node_type)
            if parse_fn is None:
                print(f"  [WARNING] Skipping unsupported node: {node_type}")
                continue

            try:
                op = parse_fn(node_data)
                rmodel.AddOperatorReference(op)
            except Exception as e:
                raise RuntimeError(
                    f"TMVA::SOFIE - Failed to parse node {node_type}: {e}"
                )

        # Add weights
        for name, tensor in weights.items():
            dtype_str = str(tensor.dtype).replace("torch.", "")
            value = tensor.detach().numpy()
            shape = list(value.shape)
            if SOFIE.ConvertStringToType(dtype_str) == SOFIE.ETensorType.FLOAT:
                rmodel.AddInitializedTensor["float"](name, shape, value.flatten())
            else:
                raise TypeError("Unsupported weight type: " + dtype_str)

        # Add input tensor info
        input_names = [x.debugName() for x in list(model.graph.inputs())[1:]]
        for name, shape, dtype in zip(input_names, input_shapes, input_dtypes):
            sofie_dtype = SOFIE.ConvertStringToType(dtype)
            if sofie_dtype == SOFIE.ETensorType.FLOAT:
                rmodel.AddInputTensorInfo(name, sofie_dtype, shape)
                rmodel.AddInputTensorName(name)
            else:
                raise TypeError("Unsupported input type: " + dtype)

        # Add output tensor names
        output_names = [x.debugName() for x in graph.outputs()]
        rmodel.AddOutputTensorNameList(output_names)

        return rmodel
