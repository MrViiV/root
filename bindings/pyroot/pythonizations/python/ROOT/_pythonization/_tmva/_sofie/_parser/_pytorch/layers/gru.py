def MakePyTorchGRU(node_data, raw_node, weights, rmodel, input_name="x"):
    """
    Create a PyTorch-compatible GRU operation using the SOFIE framework.

    Directly maps onnx::GRU node to ROperator_GRU by extracting
    weight tensor names deterministically from node inputs.
    Helper nodes (Constant, Transpose, Expand etc.) are skipped
    in the parser dispatcher.

    Parameters:
    node_data (dict): parsed node info dict
    raw_node:         raw torch graph node (for input name extraction)
    weights (dict):   model weights from _model_to_graph
    rmodel:           SOFIE RModel to register weight tensors into
    input_name (str): actual model input tensor name

    Returns:
    ROperator_GRU: A SOFIE operator for GRU.
    """
    from ROOT.TMVA.Experimental import SOFIE

    fNodeDType  = node_data["nodeDType"][0]
    fAttributes = node_data["nodeAttributes"]
    fOutputs    = node_data["nodeOutputs"]

    # Store dtype conversion once
    fTensorType = SOFIE.ConvertStringToType(fNodeDType)

    # Attributes with ONNX defaults
    fHiddenSize        = int(fAttributes.get("hidden_size", 1))
    fDirection         = str(fAttributes.get("direction", "forward")).lower()
    fActivations       = list(fAttributes.get("activations", ["Sigmoid", "Tanh"]))
    if len(fActivations) == 0:
        fActivations = ["Sigmoid", "Tanh"]
    fActivationAlpha   = list(fAttributes.get("activation_alpha", []))
    fActivationBeta    = list(fAttributes.get("activation_beta", []))
    fClip              = float(fAttributes.get("clip", 0.0))
    fLinearBeforeReset = int(fAttributes.get("linear_before_reset", 0))
    # batch_first=True in PyTorch means layout=1 (batch, seq, feature)
    fLayout            = 1

    if fHiddenSize <= 0:
        raise RuntimeError("TMVA::SOFIE GRU hidden_size must be positive")

    # Extract weight names deterministically from node inputs
    inputs = list(raw_node.inputs())
    if len(inputs) < 4:
        raise RuntimeError("TMVA::SOFIE GRU node missing required inputs (X, W, R, B)")

    fNameX        = input_name
    fNameW        = inputs[1].debugName()
    fNameR        = inputs[2].debugName()
    fNameB        = inputs[3].debugName()
    fNameSeqLens  = ""
    fNameInitialH = ""

    # Register weight tensors into rmodel
    for name in [fNameW, fNameR, fNameB]:
        if name and name in weights and not rmodel.IsInitializedTensor(name):
            tensor = weights[name]
            value  = tensor.detach().cpu().numpy()
            shape  = list(value.shape)
            if fTensorType == SOFIE.ETensorType.FLOAT:
                rmodel.AddInitializedTensor["float"](name, shape, value.flatten())

    # Output tensor names
    fNameY   = fOutputs[0] if len(fOutputs) > 0 else ""
    fNameY_h = fOutputs[1] if len(fOutputs) > 1 else ""

    if fTensorType == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_GRU["float"](
            fActivationAlpha, fActivationBeta, fActivations,
            fClip, fDirection, fHiddenSize, fLayout, fLinearBeforeReset,
            fNameX, fNameW, fNameR, fNameB,
            fNameSeqLens, fNameInitialH,
            fNameY, fNameY_h
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - GRU does not yet support input type " + fNodeDType
        )
