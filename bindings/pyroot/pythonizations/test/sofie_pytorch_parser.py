import unittest
import os
import shutil
import sys
import torch
import torch.nn as nn
import numpy as np
import ROOT

# Add parent of _pytorch to sys.path so relative imports work
sys.path.insert(0, '/home/mrviiv9/root/bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser')
from _pytorch.parser import PyTorch

def is_accurate(tensor_a, tensor_b, tolerance=1e-3):
    tensor_a = tensor_a.flatten()
    tensor_b = tensor_b.flatten()
    for i in range(len(tensor_a)):
        if abs(tensor_a[i] - tensor_b[i]) > tolerance:
            print(f"  Mismatch at {i}: SOFIE={tensor_a[i]}, PyTorch={tensor_b[i]}")
            return False
    return True

def generate_and_test_pytorch_inference(model, input_tensor, model_name, output_dir):
    model.eval()
    pt_file  = os.path.join(output_dir, model_name + ".pt")
    hxx_file = os.path.join(output_dir, model_name + ".hxx")
    dat_file = hxx_file.replace(".hxx", ".dat")

    scripted = torch.jit.trace(model, input_tensor)
    torch.jit.save(scripted, pt_file)

    input_shape = list(input_tensor.shape)
    rmodel = PyTorch.Parse(pt_file, [input_shape])
    rmodel.Generate()
    rmodel.OutputGenerated(hxx_file)

    compile_status = ROOT.gInterpreter.Declare(f'#include "{hxx_file}"')
    if not compile_status:
        raise AssertionError(f"Error compiling {hxx_file}")

    sofie_ns = getattr(ROOT, "TMVA_SOFIE_" + model_name)
    session  = sofie_ns.Session(dat_file)
    sofie_result = np.asarray(session.infer(input_tensor.numpy()))

    with torch.no_grad():
        pytorch_result = model(input_tensor).numpy()

    if not is_accurate(sofie_result, pytorch_result):
        raise AssertionError(f"SOFIE and PyTorch results do not match for {model_name}")

    print(f"  {model_name}: PASSED ✓")


class SOFIE_PyTorch_Parser(unittest.TestCase):

    def setUp(self):
        self.test_dir = self._testMethodName
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def test_elu(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ELU(alpha=1.0))
        x = torch.randn(2, 16)
        generate_and_test_pytorch_inference(model, x, "ELU_model", self.test_dir)

    def test_maxpool2d(self):
    	model = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2)
    	)
    	x = torch.randn(1, 1, 8, 8)
    	generate_and_test_pytorch_inference(model, x, "MaxPool2D_model", self.test_dir)

    @classmethod
    def tearDownClass(cls):
        for test_dir in ["test_elu", "test_maxpool2d"]:
            if os.path.isdir(test_dir):
                shutil.rmtree(test_dir)

if __name__ == "__main__":
    unittest.main()
