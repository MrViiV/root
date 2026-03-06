def MakePyTorchLSTM(node_data, raw_node, weights, rmodel, input_name="x"):
    """
    Create a PyTorch-compatible LSTM operation using the SOFIE framework.

    Directly maps onnx::LSTM node to ROperator_LSTM by extracting
    weight tensor names from node inputs (deterministic) rather than
    shape matching. Helper nodes (Constant, Transpose etc.) are skipped
    in the parser dispatcher.

    Parameters:
    node_data (dict): parsed node info dict
    raw_node:         raw torch graph node (for input name extraction)
    weights (dict):   model weights from _model_to_graph
    rmodel:           SOFIE RModel to register weight tensors into

    Returns:
    ROperator_LSTM: A SOFIE operator for LSTM.
    """
    from ROOT.TMVA.Experimental import SOFIE
    import numpy as np

    fNodeDType  = node_data["nodeDType"][0]
    fAttributes = node_data["nodeAttributes"]
    fOutputs    = node_data["nodeOutputs"]

    # Extract attributes
    fHiddenSize      = int(fAttributes.get("hidden_size", 1))
    fDirection       = str(fAttributes.get("direction", "forward")).lower()
    fActivations     = list(fAttributes.get("activations", ["Sigmoid", "Tanh", "Tanh"]))
    if len(fActivations) == 0:
        fActivations = ["Sigmoid", "Tanh", "Tanh"]
    fActivationAlpha = list(fAttributes.get("activation_alpha", []))
    fActivationBeta  = list(fAttributes.get("activation_beta", []))
    fClip            = float(fAttributes.get("clip", 0.0))
    fInputForget     = int(fAttributes.get("input_forget", 0))
    fLayout          = int(fAttributes.get("layout", 0))

    if fHiddenSize <= 0:
        raise RuntimeError("TMVA::SOFIE LSTM hidden_size must be positive")

    # Extract weight names deterministically from node inputs
    inputs = list(raw_node.inputs())
    fNameX        = input_name
    fNameW        = inputs[1].debugName() if len(inputs) > 1 else ""
    fNameR        = inputs[2].debugName() if len(inputs) > 2 else ""
    fNameB        = inputs[3].debugName() if len(inputs) > 3 else ""
    fNameSeqLens  = inputs[4].debugName() if len(inputs) > 4 else ""
    fNameInitialH = inputs[5].debugName() if len(inputs) > 5 else ""
    fNameInitialC = inputs[6].debugName() if len(inputs) > 6 else ""
    fNameP        = inputs[7].debugName() if len(inputs) > 7 else ""

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
    fNameY_c = fOutputs[2] if len(fOutputs) > 2 else ""

    if SOFIE.ConvertStringToType(fNodeDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_LSTM["float"](
            fActivationAlpha, fActivationBeta, fActivations,
            fClip, fDirection, fHiddenSize, fInputForget, fLayout,
            fNameX, fNameW, fNameR, fNameB,
            fNameSeqLens, fNameInitialH, fNameInitialC, fNameP,
            fNameY, fNameY_h, fNameY_c
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - LSTM does not yet support input type " + fNodeDType
        )
