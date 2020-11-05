#3х слойная нейронная сеть, которая распознаёт рукописные буквы с тренеровочным датасетом MNIST. 
import numpy
import scipy.special
import matplotlib.pyplot
import imageio

training_data_file = open("/mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#определение класса нейронной сети
class neuralNetwork:

    # инициализируем нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задаём кол-во узлов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # матрицы весовых коэф. связей wih(между входным и скрытным слоями)
        # и who(между скрытым и выходным слоями)
        # Весовые коэф. связей между узлом i и узлом j следующего слоя обозначены, как w_i_j: 
        # w11 w21
        # w12 w22 и тд
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # функция активации - сигмойда
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # тренировка сети
    def train(self, inputs_list, targets_list):
        # преобразование списка входных значений
        # в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать выходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибки выходного слоя = (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors, распределённые
        # пропорционально весовым коэф. связей и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить веса для связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # обновить весовые коэф. для связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя 
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# кол-во входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate
learning_rate = 0.1

# создать экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# грузим тренеровочный датасет
training_data_file = open("D:/makeyourownneuralnetwork-master/mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# кол-во эпох
epochs = 10

for e in range(epochs):
    # перебираем все записи в тренеровочном датасете
    for record in training_data_list:
        # разделяем ','
        all_values = record.split(',')
        # масштабируем и сдвигаем входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# грузим тестовый датасет
test_data_file = open("/mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


img_array = imageio.imread('/my_own_images/2828_my_own_6.png', as_gray=True)

# reshape from 28x28 to list of 784 values, invert values
img_data = 255.0 - img_array.reshape(784)

# then scale data to range from 0.01 to 1.0
img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

# plot image
matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')

# опрашиваем сеть
outputs = n.query(img_data)
print(outputs)

label = numpy.argmax(outputs)
print("network says ", label)