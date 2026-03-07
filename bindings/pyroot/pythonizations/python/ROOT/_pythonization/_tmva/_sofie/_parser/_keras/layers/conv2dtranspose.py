import math
from .. import get_keras_version

def MakeKerasConv2DTranspose(layer):
    """
    Create a Keras-compatible Conv2DTranspose layer operation using SOFIE framework.

    Conv2DTranspose (transposed convolution) is the transpose of a regular convolution,
    used to upsample feature maps. It is commonly used in decoder networks and generative
    models such as autoencoders and GANs.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  data type (must be float), weight and bias name, kernel size,
                  dilations, padding, strides and output_padding.

    Returns:
    ROperator_ConvTranspose: A SOFIE framework operator representing the Conv2DTranspose operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    keras_version = get_keras_version()
    finput          = layer["layerInput"]
    foutput         = layer["layerOutput"]
    fLayerDType     = layer["layerDType"]
    fLayerInputName  = finput[0]
    fLayerOutputName = foutput[0]
    attributes      = layer["layerAttributes"]
    fWeightNames    = layer["layerWeight"]
    fKernelName     = fWeightNames[0]
    fBiasName       = fWeightNames[1] if len(fWeightNames) > 1 else ""

    fAttrDilations   = list(attributes["dilation_rate"])
    fAttrKernelShape = list(attributes["kernel_size"])
    fAttrStrides     = list(attributes["strides"])
    fAttrGroup       = 1
    fAttrPads        = []
    fAttrOutputPads  = []
    fAttrOutputShape = []

    # Handle output_padding
    fOutputPadding = attributes.get("output_padding", None)
    if fOutputPadding is not None:
        fAttrOutputPads = list(fOutputPadding) if hasattr(fOutputPadding, '__iter__') else [fOutputPadding, fOutputPadding]

    fKerasPadding = str(attributes["padding"])
    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "SAME_UPPER"
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
            "TMVA::SOFIE - Unsupported - Conv2DTranspose does not yet support input type " + fLayerDType
        )
