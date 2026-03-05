import os
import time
import tempfile

from .layers.elu import MakePyTorchELU
from .layers.maxpool2d import MakePyTorchMaxPool2D
from .layers.batchnorm2d import MakePyTorchBatchNorm2D

def _node_get(node, key):
    """Helper to get node attribute without depending on onnx submodule."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

def MakePyTorchGemm(node_data):
    """Parse Gemm (Linear layer) operator."""
    from ROOT.TMVA.Experimental import SOFIE
    attrs   = node_data["nodeAttributes"]
    inputs  = node_data["nodeInputs"]
    outputs = node_data["nodeOutputs"]
    dtype   = node_data["nodeDType"][0]

    nameA = inputs[0]
    nameB = inputs[1]
    nameC = inputs[2]
    nameY = outputs[0]

    alpha  = float(attrs.get("alpha", 1.0))
    beta   = float(attrs.get("beta", 1.0))
    transB = int(attrs.get("transB", 0))
    transA = int(not transB)

    if SOFIE.ConvertStringToType(dtype) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Gemm["float"](alpha, beta, transA, transB, nameA, nameB, nameC, nameY)
        return op
    else:
        raise RuntimeError("Unsupported type for Gemm: " + dtype)

RECURRENT_OPS = {"onnx::RNN", "onnx::LSTM", "onnx::GRU"}

def _has_recurrent_ops(graph):
    """Check if graph contains any recurrent operators."""
    for node in graph.nodes():
        if node.kind() in RECURRENT_OPS:
            return True
    return False


# Design decision: Recurrent model parsing via ONNX fallback
#
# For recurrent layers (RNN, LSTM, GRU), the PyTorch exporter generates
# complex ONNX graphs containing many helper operators (Transpose, Constant,
# Gather, Unsqueeze, Expand etc.) that would require significant duplicated
# logic to handle correctly in Python.
#
# Instead of reimplementing this logic, the PyTorch parser detects recurrent
# operators and delegates to the existing SOFIE RModelParser_ONNX via
# torch.onnx.export. This ensures correctness, avoids code duplication,
# and leverages the tested ONNX parsing infrastructure already in SOFIE.


def _parse_recurrent_model(model, input_shapes, model_name=None):
    """
    Parse a recurrent model by exporting to ONNX and using
    the existing SOFIE ONNX parser. Used as fallback for
    RNN/LSTM/GRU which have complex helper nodes in their graphs.
    """
    import torch
    import ROOT

    if not hasattr(ROOT.TMVA.Experimental.SOFIE, "RModelParser_ONNX"):
        raise RuntimeError("SOFIE ONNX parser not available")

    dummy_inputs = tuple(torch.rand(*shape) for shape in input_shapes)

    if model_name is None:
        model_name = "recurrent_model"
    onnx_path = os.path.join(tempfile.gettempdir(), model_name + ".onnx")

    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            export_params=True,
            dynamo=False,
            input_names=["input"],
            output_names=["output"]
        )
        parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
        rmodel = parser.Parse(onnx_path)
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    return rmodel

# Map from ONNX node types to their parsing functions
mapPyTorchNode = {
    "onnx::Gemm":              MakePyTorchGemm,
    "onnx::Elu":               MakePyTorchELU,
    "onnx::MaxPool":           MakePyTorchMaxPool2D,
    "onnx::BatchNormalization": MakePyTorchBatchNorm2D,
}

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

        model = torch.jit.load(filename)
        model.cpu()
        model.eval()

        dummy_inputs = [torch.rand(*shape) for shape in input_shapes]
        result  = _model_to_graph(model, dummy_inputs)
        graph   = result[0]
        weights = result[1]

        # Check for recurrent ops - use ONNX parser fallback
        if _has_recurrent_ops(graph):
            print("  [INFO] Recurrent model detected, using ONNX parser fallback")
            return _parse_recurrent_model(model, input_shapes, filename_nodir.replace(".pt", ""))

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
                if node_type == "onnx::Gemm":
                    rmodel.AddBlasRoutines({"Gemm", "Gemv"})
            except Exception as e:
                raise RuntimeError(f"TMVA::SOFIE - Failed to parse node {node_type}: {e}")

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
