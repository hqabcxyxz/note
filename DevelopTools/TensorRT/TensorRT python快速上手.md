#TensorRT 

# 引入TensorRT
```python
import tensorrt as trt

TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
```

# 转换网络
目的是为了将目标网络变为一个[`INetworkDefinition`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html#tensorrt.INetworkDefinition)实例.   
常见的方法用用 tensorRT 内建的方法搭建网络和从 ONNX 或者 Caffe TensorFlow 转.这里以内建方法搭建网络为例演示流程:

```python
# Create the builder and network
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
	# Configure the network layers based on the weights provided. In this case, the weights are imported from a pytorch model. 
	# Add an input layer. The name is a string, dtype is a TensorRT dtype, and the shape can be provided as either a list or tuple.
	input_tensor = network.add_input(name=INPUT_NAME, dtype=trt.float32, shape=INPUT_SHAPE)

	# Add a convolution layer
	conv1_w = weights['conv1.weight'].numpy()
	conv1_b = weights['conv1.bias'].numpy()
	conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
	conv1.stride = (1, 1)

	pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
	pool1.stride = (2, 2)
	conv2_w = weights['conv2.weight'].numpy()
	conv2_b = weights['conv2.bias'].numpy()
	conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
	conv2.stride = (1, 1)

	pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
	pool2.stride = (2, 2)

	fc1_w = weights['fc1.weight'].numpy()
	fc1_b = weights['fc1.bias'].numpy()
	fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

	relu1 = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)

	fc2_w = weights['fc2.weight'].numpy()
	fc2_b = weights['fc2.bias'].numpy()
	fc2 = network.add_fully_connected(relu1.get_output(0), OUTPUT_SIZE, fc2_w, fc2_b)

	fc2.get_output(0).name =OUTPUT_NAME
	network.mark_output(fc2.get_output(0))
```

这里流程就是使用上一步的 `Logger` 创建一个 `tnesorrt.Builder` 实例,然后使用这个实例创建一个 `INetworkDefinition` 实例.然后使用 `INetworkDefinnition` 实例设置输入,添加层构建网络,然后标记输出.

# 构建引擎
```python
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
    config.max_workspace_size = 1 << 20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    with builder.build_engine(network, config) as engine:

```

# 序列与反序列化引擎
序列化:  
```python
with open(“sample.engine”, “wb”) as f:
		f.write(engine.serialize())
```

反序列化:  
```python
with open(“sample.engine”, “rb”) as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())
```