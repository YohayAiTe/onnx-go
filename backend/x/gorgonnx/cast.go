package gorgonnx

import (
	"encoding/binary"
	"errors"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	register("Cast", newCast)
}

type cast struct {
	to    tensor.Dtype
	shape tensor.Shape
}

func newCast() operator {
	return &cast{}
}

func (c *cast) Arity() int {
	return 1
}

func (c *cast) Type() hm.Type {
	a := hm.TypeVariable('a')
	d := hm.TypeVariable('d')
	dataType := gorgonia.TensorType{Dims: len(c.shape), Of: a}
	retType := gorgonia.TensorType{Dims: len(c.shape), Of: d}
	return hm.NewFnType(dataType, retType)
}

func (c *cast) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, errors.New("gather: infershape failed, nil shape")
	}
	return inputs[0].(tensor.Shape), nil
}

func doCast[T float32 | float64 | int8 | int16 | int32 | int64](input *tensor.Dense, to tensor.Dtype) (*tensor.Dense, error) {
	output := tensor.NewDense(to, input.Shape())
	switch input.Dtype() {
	case tensor.Float32:
		vals := input.Data().([]float32)
		for i := 0; i < input.Size(); i++ {
			output.Set(i, T(vals[i]))
		}
	case tensor.Float64:
		vals := input.Data().([]float64)
		for i := 0; i < input.Size(); i++ {
			output.Set(i, T(vals[i]))
		}
	case tensor.Int8:
		vals := input.Data().([]int8)
		for i := 0; i < input.Size(); i++ {
			output.Set(i, T(vals[i]))
		}
	case tensor.Int16:
		vals := input.Data().([]int16)
		for i := 0; i < input.Size(); i++ {
			output.Set(i, T(vals[i]))
		}
	case tensor.Int32:
		vals := input.Data().([]int32)
		for i := 0; i < input.Size(); i++ {
			output.Set(i, T(vals[i]))
		}
	case tensor.Int64:
		vals := input.Data().([]int64)
		for i := 0; i < input.Size(); i++ {
			output.Set(i, T(vals[i]))
		}
	default:
		return nil, errors.New("cast Unsupported type")
	}
	return output, nil
}

func (c *cast) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != c.Arity() {
		return nil, errors.New("cast: wrong number of arguments")
	}
	input, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.New("cast: only dense are supported")

	}

	switch c.to {
	case tensor.Float32:
		return doCast[float32](input, c.to)
	case tensor.Float64:
		return doCast[float64](input, c.to)
	case tensor.Int8:
		return doCast[int8](input, c.to)
	case tensor.Int16:
		return doCast[int16](input, c.to)
	case tensor.Int32:
		return doCast[int32](input, c.to)
	case tensor.Int64:
		return doCast[int64](input, c.to)
	default:
		return nil, errors.New("cast Unsupported type")
	}
}

func (c *cast) ReturnsPtr() bool {
	return false
}

func (c *cast) CallsExtern() bool {
	return false
}

func (c *cast) OverwritesInput() int {
	return -1
}

func (c *cast) WriteHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, []byte(`cast`)); err != nil {
		panic(err)
	}
}

func (c *cast) Hashcode() uint32 {
	h := fnv.New32a()
	c.WriteHash(h)
	return h.Sum32()
}

func (c *cast) String() string {
	return "cast"
}

func (c *cast) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	var err error
	children := getOrderedChildren(g.g, n)
	if err := checkCondition(children, 1); err != nil {
		return err
	}
	input := children[0]
	c.shape = input.gorgoniaNode.Shape()
	n.gorgoniaNode, err = gorgonia.ApplyOp(c, input.gorgoniaNode)
	if err != nil {
		return err
	}

	return nil
}

func (c *cast) init(o onnx.Operation) error {
	var err error
	c.to, err = ir.TensorProto_DataType(o.Attributes["to"].(int64)).Dtype()
	return err
}
