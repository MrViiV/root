from .. import get_keras_version

def MakeKerasConv2DTranspose(layer):
    """
    Create a Keras-compatible Conv2DTranspose layer operation using SOFIE framework.
    """
    from ROOT.TMVA.Experimental import SOFIE

    finput           = layer["layerInput"]
    foutput          = layer["layerOutput"]
    fLayerDType      = layer["layerDType"]
    fLayerInputName  = finput[0]
    fLayerOutputName = foutput[0]
    attributes       = layer["layerAttributes"]
    fWeightNames     = layer["layerWeight"]
    fKernelName      = fWeightNames[0]
    fBiasName        = fWeightNames[1] if len(fWeightNames) > 1 else ""

    fAttrDilations   = list(attributes["dilation_rate"])
    fAttrKernelShape = list(attributes["kernel_size"])
    fAttrStrides     = list(attributes["strides"])
    fAttrGroup       = 1
    fAttrOutputPads  = []
    fAttrOutputShape = []

    fKerasPadding = str(attributes["padding"]).lower()

    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
        fAttrPads    = []
    elif fKerasPadding == "same":
        # Compute explicit pads for NOTSET mode
        # For ConvTranspose same padding: pad_total = kernel - stride (per dim)
        fAttrAutopad = "NOTSET"
        pads_begin = []
        pads_end   = []
        for k, s in zip(fAttrKernelShape, fAttrStrides):
            pad_total = max(k - s, 0)
            pad_begin = pad_total // 2
            pad_end   = pad_total - pad_begin
            pads_begin.append(pad_begin)
            pads_end.append(pad_end)
        fAttrPads = pads_begin + pads_end
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Conv2DTranspose does not yet support padding: " + fKerasPadding
        )

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad,
            fAttrDilations,
            fAttrGroup,
            fAttrKernelShape,
            fAttrOutputPads,
            fAttrOutputShape,
            fAttrPads,
            fAttrStrides,
            fLayerInputName,
            fKernelName,
            fBiasName,
            fLayerOutputName,
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Conv2DTranspose does not support type " + fLayerDType
        )
