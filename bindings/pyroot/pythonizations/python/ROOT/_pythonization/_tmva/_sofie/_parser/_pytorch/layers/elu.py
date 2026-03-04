def MakePyTorchELU(node):
    """
    Create a PyTorch-compatible ELU activation operation using SOFIE framework.

    ELU applies: f(x) = x if x >= 0 else alpha * (exp(x) - 1)

    Parameters:
    node (dict): {
        'nodeType':       str  - ONNX operator type ('onnx::Elu')
        'nodeAttributes': dict - operator attributes (e.g. alpha)
        'nodeInputs':     list - input tensor names
        'nodeOutputs':    list - output tensor names
        'nodeDType':      list - data types
    }

    Returns:
    ROperator_Elu: A SOFIE operator for ELU activation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    fNodeDType  = node["nodeDType"][0]
    fInputName  = node["nodeInputs"][0]
    fOutputName = node["nodeOutputs"][0]
    attributes  = node["nodeAttributes"]

    # alpha defaults to 1.0 per ONNX spec
    fAlpha = float(attributes.get("alpha", 1.0))

    if SOFIE.ConvertStringToType(fNodeDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Elu("float")(fAlpha, fInputName, fOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator ELU does not yet support input type " + fNodeDType
        )
