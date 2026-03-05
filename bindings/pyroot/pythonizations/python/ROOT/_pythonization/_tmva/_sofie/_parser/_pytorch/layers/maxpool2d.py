def MakePyTorchMaxPool2D(node):
    """
    Create a PyTorch-compatible MaxPool2D operation using the SOFIE framework.

    MaxPool2D downsamples by taking the maximum value over a sliding window.

    Parameters:
    node (dict): {
        'nodeType':       str  - 'onnx::MaxPool'
        'nodeAttributes': dict - kernel_shape, pads, strides, dilations, ceil_mode
        'nodeInputs':     list - input tensor names
        'nodeOutputs':    list - output tensor names
        'nodeDType':      list - data types
    }

    Returns:
    ROperator_Pool: A SOFIE operator for MaxPool2D.
    """
    from ROOT.TMVA.Experimental import SOFIE

    fNodeDType  = node["nodeDType"][0]
    fInputName  = node["nodeInputs"][0]
    fOutputName = node["nodeOutputs"][0]
    fAttributes = node["nodeAttributes"]

    # Normalize kernel_shape - ONNX may give int instead of list
    fAttrKernelShape = fAttributes.get("kernel_shape", [1, 1])
    if isinstance(fAttrKernelShape, int):
        fAttrKernelShape = [fAttrKernelShape, fAttrKernelShape]

    # Normalize strides - same issue
    fAttrStrides = fAttributes.get("strides", [1, 1])
    if isinstance(fAttrStrides, int):
        fAttrStrides = [fAttrStrides, fAttrStrides]

    # Normalize pads - ONNX gives [pH_begin, pW_begin, pH_end, pW_end]
    # SOFIE expects symmetric [pH, pW]
    fAttrPads = fAttributes.get("pads", [0, 0, 0, 0])
    if len(fAttrPads) == 4:
        fAttrPads = [fAttrPads[0], fAttrPads[1]]

    fAttrDilations    = list(fAttributes.get("dilations", [1, 1]))
    fAttrCeilMode     = int(fAttributes.get("ceil_mode", 0))
    fAttrStorageOrder = int(fAttributes.get("storage_order", 0))

    fPoolAttr = SOFIE.RAttributes_Pool()
    fPoolAttr.kernel_shape  = list(fAttrKernelShape)
    fPoolAttr.strides       = list(fAttrStrides)
    fPoolAttr.pads          = list(fAttrPads)
    fPoolAttr.dilations     = fAttrDilations
    fPoolAttr.ceil_mode     = fAttrCeilMode
    fPoolAttr.storage_order = fAttrStorageOrder
    fPoolAttr.auto_pad      = "NOTSET"

    if SOFIE.ConvertStringToType(fNodeDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Pool["float"](SOFIE.PoolOpMode.MaxPool, fPoolAttr, fInputName, fOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator MaxPool2D does not yet support input type " + fNodeDType
        )
