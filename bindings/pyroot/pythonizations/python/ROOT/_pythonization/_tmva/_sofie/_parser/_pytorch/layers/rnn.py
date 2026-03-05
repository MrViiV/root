def MakePyTorchRNN(node):
    """
    Create a PyTorch-compatible RNN operation using the SOFIE framework.

    Supports forward, reverse and bidirectional RNNs.

    Parameters:
    node (dict): {
        'nodeType':       str  - 'onnx::RNN'
        'nodeAttributes': dict - activations, clip, direction, hidden_size, layout
        'nodeInputs':     list - [X, W, R, B, sequence_lens, initial_h]
        'nodeOutputs':    list - [Y, Y_h]
        'nodeDType':      list - data types
    }

    Returns:
    ROperator_RNN: A SOFIE operator for RNN.
    """
    from ROOT.TMVA.Experimental import SOFIE

    fNodeDType  = node["nodeDType"][0]
    fInputs     = node["nodeInputs"]
    fOutputs    = node["nodeOutputs"]
    fAttributes = node["nodeAttributes"]

    # Validate required inputs
    if len(fInputs) < 3:
        raise RuntimeError(
            "TMVA::SOFIE RNN expects at least 3 inputs (X, W, R)"
        )

    # Extract input tensor names (some may be optional)
    fNameX        = fInputs[0] if len(fInputs) > 0 else ""
    fNameW        = fInputs[1] if len(fInputs) > 1 else ""
    fNameR        = fInputs[2] if len(fInputs) > 2 else ""
    fNameB        = fInputs[3] if len(fInputs) > 3 else ""
    fNameSeqLens  = fInputs[4] if len(fInputs) > 4 else ""
    fNameInitialH = fInputs[5] if len(fInputs) > 5 else ""

    # Extract output tensor names
    fNameY   = fOutputs[0] if len(fOutputs) > 0 else ""
    fNameY_h = fOutputs[1] if len(fOutputs) > 1 else ""

    # Attributes with ONNX defaults
    fActivationAlpha = list(fAttributes.get("activation_alpha", []))
    fActivationBeta  = list(fAttributes.get("activation_beta", []))
    fActivations     = list(fAttributes.get("activations", ["Tanh"]))
    if len(fActivations) == 0:
        fActivations = ["Tanh"]
    fClip        = float(fAttributes.get("clip", 0.0))
    fDirection   = str(fAttributes.get("direction", "forward")).lower()
    fHiddenSize  = int(fAttributes.get("hidden_size", 1))
    fLayout      = int(fAttributes.get("layout", 0))

    if fHiddenSize <= 0:
        raise RuntimeError(
            "TMVA::SOFIE RNN hidden_size must be positive"
        )

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
            "TMVA::SOFIE - Unsupported - Operator RNN does not yet support input type " + fNodeDType
        )
