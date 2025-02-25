from torch_multiplication_network import TorchMultiplicationNetwork

def main():
    epoch = 1000
    network = TorchMultiplicationNetwork(256)
    input_data, output_data = network.create_training_data()
    network.train(input_data, output_data, epoch)
    network.test()

if __name__ == "__main__":
    main()
