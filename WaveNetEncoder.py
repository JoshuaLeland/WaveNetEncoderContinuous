import math
import torch

class Conv(torch.nn.Module):
	"""
	A convolution with the option to be causal and use xavier initialization
	"""
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
				 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
		super(Conv, self).__init__()
		self.is_causal = is_causal
		self.kernel_size = kernel_size
		self.dilation = dilation

		self.conv = torch.nn.Conv1d(in_channels, out_channels,
									kernel_size=kernel_size, stride=stride,
									dilation=dilation, bias=bias)

		torch.nn.init.xavier_uniform(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, signal):
		if self.is_causal:
				padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
				signal = torch.nn.functional.pad(signal, padding) 
		return self.conv(signal)

class WaveNetEncoder(torch.nn.Module):
	def __init__(self, n_layers, max_dilation, n_residual_channels, n_dilated_channels, encoding_factor, encoding_stride=1):

		#Super
		super(WaveNetEncoder, self).__init__()

		#Assign the parameters
		self.n_layers = n_layers
		self.max_dilation = max_dilation
		self.n_residual_channels = n_residual_channels
		self.encoding_factor = encoding_factor
		self.encoding_stride = encoding_stride
		self.n_dilated_channels = n_dilated_channels

		#Make layers
		self.dilate_layers = torch.nn.ModuleList()
		self.res_layers = torch.nn.ModuleList()

		#Build the encoder net.
		self.NCInput = Conv(1, self.n_residual_channels)

		#Build the resnet. According to paper we don't use NC conv
		loop_factor = math.floor(math.log2(max_dilation)) + 1
		for i in range(self.n_layers):

			#Double dilation up to max dilation.
			dilation = 2**(i % loop_factor)

			#Build Dialation later 
			#We made n_dilated_channels a hyperparameter, but there is no mention of what they used in Engel, Resnick et al
			d_layer = Conv(self.n_residual_channels, self.n_dilated_channels,kernel_size=2, dilation=dilation, w_init_gain='tanh')
			self.dilate_layers.append(d_layer)

			#Build Res layer 
			res_layer = Conv(self.n_dilated_channels, self.n_residual_channels, w_init_gain='linear')
			self.res_layers.append(res_layer)

		#Final Layer
		self.final_layer = Conv(self.n_residual_channels, 1)

		#Pooling layer.
		self.pooling_layer = torch.nn.AvgPool1d(self.encoding_factor, stride = self.encoding_stride)

	def forward(self, signal):
		#NC Conv
		signal = self.NCInput(signal)

		for i in range(self.n_layers):
			#Save this for now.
			skip = signal 

			#Run block.
			signal = torch.nn.functional.relu(signal, True)
			signal = self.dilate_layers[i](signal)
			signal = torch.nn.functional.relu(signal, True)
			signal = self.res_layers[i](signal)

			#Dilate layers can clip this, so resize.
			length = signal.size(2)

			skip = skip[:,:,-length:]

			signal = signal + skip

		#Run the last 1x1 layer
		signal = self.final_layer(signal)

		#Pooling for encoding.
		signal = self.pooling_layer(signal)

		return signal








