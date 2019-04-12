package onnxtest

// this file is auto-generated... DO NOT EDIT

import (
	"github.com/owulveryck/onnx-go/backend/testbackend"
	"gorgonia.org/tensor"
)

// NewTestSumTwoInputs version: 3.
func NewTestSumTwoInputs() *testbackend.TestCase {
	return &testbackend.TestCase{
		Title:  "TestSumTwoInputs",
		ModelB: []byte{0x8, 0x3, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x76, 0xa, 0x1d, 0xa, 0x6, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x30, 0xa, 0x6, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x31, 0x12, 0x6, 0x72, 0x65, 0x73, 0x75, 0x6c, 0x74, 0x22, 0x3, 0x53, 0x75, 0x6d, 0x12, 0x13, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x73, 0x75, 0x6d, 0x5f, 0x74, 0x77, 0x6f, 0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x73, 0x5a, 0x14, 0xa, 0x6, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x30, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x3, 0x5a, 0x14, 0xa, 0x6, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x31, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x3, 0x62, 0x14, 0xa, 0x6, 0x72, 0x65, 0x73, 0x75, 0x6c, 0x74, 0x12, 0xa, 0xa, 0x8, 0x8, 0x1, 0x12, 0x4, 0xa, 0x2, 0x8, 0x3, 0x42, 0x2, 0x10, 0x9},

		/*

		   &pb.NodeProto{
		     Input:     []string{"data_0", "data_1"},
		     Output:    []string{"result"},
		     Name:      "",
		     OpType:    "Sum",
		     Attributes: ([]*pb.AttributeProto) <nil>
		   ,
		   },


		*/

		Input: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(3),
				tensor.WithBacking([]float32{3, 0, 2}),
			),

			tensor.New(
				tensor.WithShape(3),
				tensor.WithBacking([]float32{1, 3, 4}),
			),
		},
		ExpectedOutput: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(3),
				tensor.WithBacking([]float32{4, 3, 6}),
			),
		},
	}
}
