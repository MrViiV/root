def MakePyTorchBatchNorm2D(node):
    """
    Create a PyTorch-compatible BatchNorm2D operation using the SOFIE framework.

    Normalizes inputs to have zero mean and unit variance, then applies
    learnable scale and bias parameters.

    Parameters:
    node (dict): {
        'nodeType':       str  - 'onnx::BatchNormalization'
        'nodeAttributes': dict - epsilon, momentum, training_mode
        'nodeInputs':     list - [input, scale, bias, mean, var]
        'nodeOutputs':    list - output tensor names
        'nodeDType':      list - data types
    }

    Returns:
    ROperator_BatchNormalization: A SOFIE operator for BatchNorm2D.
    """
    from ROOT.TMVA.Experimental import SOFIE

    fNodeDType  = node["nodeDType"][0]
    fInputs     = node["nodeInputs"]
    fOutputs    = node["nodeOutputs"]
    fAttributes = node["nodeAttributes"]

    if len(fInputs) !=5:
	raise RuntimeError(
	    "TMVA::SOFIE BatchNorm2D expects 5 inputs but got {}".format(len(fInputs))
	)

    # Input tensors: X, scale, bias, mean, var
    fNameX     = fInputs[0]
    fNameScale = fInputs[1]
    fNameBias  = fInputs[2]
    fNameMean  = fInputs[3]
    fNameVar   = fInputs[4]
    fNameY     = fOutputs[0]

    # Attributes with ONNX defaults
    fEpsilon      = float(fAttributes.get("epsilon", 1e-5))
    fMomentum     = float(fAttributes.get("momentum", 0.9))
    fTrainingMode = int(fAttributes.get("training_mode", 0))

    if SOFIE.ConvertStringToType(fNodeDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_BatchNormalization["float"](
            fEpsilon, fMomentum, fTrainingMode,
            fNameX, fNameScale, fNameBias,
            fNameMean, fNameVar, fNameY
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator BatchNorm2D does not yet support input type " + fNodeDType
        )
