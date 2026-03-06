def MakePyTorchRNN(node_data, raw_node, weights, rmodel, input_name="x"):
    """
    Create a PyTorch-compatible RNN operation using the SOFIE framework.

    Directly maps onnx::RNN node to ROperator_RNN by extracting
    weight tensor names from node inputs (deterministic).
    Helper nodes (Constant, Transpose etc.) are skipped in dispatcher.

    Parameters:
    node_data (dict): parsed node info dict
    raw_node:         raw torch graph node (for input name extraction)
    weights (dict):   model weights from _model_to_graph
    rmodel:           SOFIE RModel to register weight tensors into

    Returns:
    ROperator_RNN: A SOFIE operator for RNN.
    """
    from ROOT.TMVA.Experimental import SOFIE

    fNodeDType  = node_data["nodeDType"][0]
    fAttributes = node_data["nodeAttributes"]
    fOutputs    = node_data["nodeOutputs"]

    # Attributes with ONNX defaults
    fHiddenSize      = int(fAttributes.get("hidden_size", 1))
    fDirection       = str(fAttributes.get("direction", "forward")).lower()
    fActivations     = list(fAttributes.get("activations", ["Tanh"]))
    if len(fActivations) == 0:
        fActivations = ["Tanh"]
    fActivationAlpha = list(fAttributes.get("activation_alpha", []))
    fActivationBeta  = list(fAttributes.get("activation_beta", []))
    fClip            = float(fAttributes.get("clip", 0.0))
    fLayout          = 1

    if fHiddenSize <= 0:
        raise RuntimeError("TMVA::SOFIE RNN hidden_size must be positive")

    # Extract weight names deterministically from node inputs
    inputs = list(raw_node.inputs())
    fNameX        = input_name                         # X is the original model input, not the transposed helper node output
    fNameW        = inputs[1].debugName() if len(inputs) > 1 else ""
    fNameR        = inputs[2].debugName() if len(inputs) > 2 else ""
    fNameB        = inputs[3].debugName() if len(inputs) > 3 else ""
    fNameSeqLens  = ""
    fNameInitialH = ""

    # Register weight tensors into rmodel
    for name in [fNameW, fNameR, fNameB]:
        if name and name in weights:
            tensor = weights[name]
            value  = tensor.detach().numpy()
            shape  = list(value.shape)
            if SOFIE.ConvertStringToType(fNodeDType) == SOFIE.ETensorType.FLOAT:
                rmodel.AddInitializedTensor["float"](name, shape, value.flatten())

    # Output tensor names
    fNameY   = fOutputs[0] if len(fOutputs) > 0 else ""
    fNameY_h = fOutputs[1] if len(fOutputs) > 1 else ""

    if SOFIE.ConvertStringToType(fNodeDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_RNN["float"](
            fActivationAlpha, fActivationBeta, fActivations,
            fClip, fDirection, fHiddenSize, fLayout,
            fNameX, fNameW, fNameR, fNameB,
            fNameSeqLens, fNameInitialH,
            fNameY, fNameY_h
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - RNN does not yet support input type " + fNodeDType
        )
