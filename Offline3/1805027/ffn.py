import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, f1_score

class DenseLayer:
    def __init__(self, input_size, output_size):
        
        var = 2. / (input_size + output_size)
        self.weights = np.random.normal(0, np.sqrt(var), (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        # self.weights = np.random.randn(input_size, output_size) * 0.01
        # self.biases = np.zeros((1, output_size))

        # Initialize Adam parameters for weights and biases
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, input):
        self.input = input
        t = np.dot(input, self.weights) + self.biases   
        if np.isnan(t).any():
            print("forward is NAN")
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate,  beta1, beta2, epsilon, t):
        # Gradients of weights and biases
        weights_gradient = np.dot(self.input.T, output_gradient) / self.input.shape[0]
        # biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        biases_gradient = np.mean(output_gradient, axis=0, keepdims=True)

        # Update Adam parameters for weights
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * weights_gradient
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (weights_gradient ** 2)

        # Compute bias-corrected first and second moment estimates for weights
        m_hat_weights = self.m_weights / (1 - beta1 ** t)
        v_hat_weights = self.v_weights / (1 - beta2 ** t)

        # Update weights with Adam
        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
        self.biases -= learning_rate * biases_gradient
        
        # # Update parameters
        # self.weights -= learning_rate * weights_gradient
        # self.biases -= learning_rate * biases_gradient
        # Return input gradient for further backpropagation
        return np.dot(output_gradient, self.weights.T)/self.input.shape[0]
    
class ReLULayer:
    def forward(self, input):
        # Save input for backpropagation
        # print("in relu")
        self.input = input
        if np.isnan(np.maximum(input,0)).any():
            print("relu forward is NAN")
        return np.maximum(input, 0)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)
    
class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        
    def forward(self, input, training=True):
        if training:
            # Create a random mask of 0s and 1s, where each element is 0 with probability dropout_rate
            self.mask = (np.random.rand(*input.shape) > self.dropout_rate).astype(int)
            # Scale the input by inverse of dropout_rate to maintain the expected value
            return input * self.mask / (1 - self.dropout_rate)
        else:
            return input
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask
    
class SoftmaxCrossEntropyLoss:
    def __init__(self):
        pass

    def softmax(self, z):
        # print(z.shape)
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            print("Invalid values detected in input to softmax")
        
        # e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        rows = z.shape[0]

        for i in range(rows):
            z[i,:] = z[i,:] - np.max(z[i,:])
            tmp = np.exp(z[i,:])
            z[i,:] = tmp / np.sum(tmp)
        e_z = z

        return e_z


    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-9, 1-1e-9) 
        m = y_true.shape[0]      
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        return np.sum(log_likelihood) / m

    def gradient(self, y_true, y_pred):
        return y_pred - y_true
    
class Model:
    def __init__(self, layer_sizes, dropout_rate=0.0):
        # Initialize Adam parameters
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.t = 1   # Time step

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2: 
                self.layers.append(ReLULayer())
            if dropout_rate > 0.0:  # add dropout if rate is specified
                self.layers.append(Dropout(dropout_rate))
        self.loss_func = SoftmaxCrossEntropyLoss()
        
    def get_model_info(self):
        num_layers = len(self.layers)  # Total number of layers
        neurons_per_layer = []

        for layer in self.layers:
            if hasattr(layer, 'weights'):  # Check if the layer has weights attribute
                neurons = layer.weights.shape[0]  # The number of neurons is the size of the first dimension of weights
                neurons_per_layer.append(neurons)

        return num_layers, neurons_per_layer

    def forward(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x) if training else x
            else:
                x = layer.forward(x)
        x = self.loss_func.softmax(x)
  
        return x

    def compute_loss_and_gradients(self, x, y_true, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Forward pass
        y_pred = self.forward(x, training=True)
        loss = self.loss_func.loss(y_true, y_pred)
       
        # # Backward pass
        # gradient = self.loss_func.gradient(y_true, y_pred)
        # for layer in reversed(self.layers):
        #     gradient = layer.backward(gradient, learning_rate)

        self.t += 1

        gradient = self.loss_func.gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):  # Ensure it's a dense layer
                gradient = layer.backward(gradient, learning_rate, beta1, beta2, epsilon, self.t)
            else:  # For non-dense layers, just do regular backprop
                gradient = layer.backward(gradient, learning_rate)

        return loss
    
    
    def get_weights_biases(self):
        # Extract the weights and biases from each layer
        params = {'weights': [], 'biases': []}
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params['weights'].append(layer.weights)
            if hasattr(layer, 'biases'):
                params['biases'].append(layer.biases)
        return params


    def train(self, train_loader, validation_loader, learning_rate, epochs, best_accuracy=0.0):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_f1s, val_f1s = [], []
        
        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0
            # total_loss = 0
            
            #Trining phase
            for batch_idx, (data, targets) in enumerate(train_loader):
               
                data = data.numpy().reshape(data.shape[0], -1)  
                targets = targets.numpy()

                targets = targets - 1

                # One-hot encode targets
                targets_one_hot = np.eye(26)[targets]  

                # Forward and backward passes
                loss = self.compute_loss_and_gradients(data, targets_one_hot, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
                total_train_loss += loss
                
            #Validation phase
            for batch_idx, (data, targets) in enumerate(validation_loader):
               
                data = data.numpy().reshape(data.shape[0], -1)  
                targets = targets.numpy()

                targets = targets - 1

                # One-hot encode targets
                targets_one_hot = np.eye(26)[targets]  

                y_pred = self.forward(data, training=False)
                loss = self.loss_func.loss(targets_one_hot, y_pred)
        
                total_val_loss += loss
                
            # Evaluate the model for both training and validation sets
            train_accuracy, train_f1, _, _ = self.evaluate(train_loader)
            validation_accuracy, validation_f1, final_val_targets, final_val_predictions = self.evaluate(validation_loader)
            
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_model = copy.deepcopy(self)
                best_params = self.get_weights_biases()
            
            train_accuracies.append(train_accuracy)
            train_f1s.append(train_f1)
            val_accuracies.append(validation_accuracy)
            val_f1s.append(validation_f1)

            # Calculate average loss for the epoch
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            avg_val_loss = total_val_loss / len(validation_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch}, Average training loss: {avg_train_loss:.3f}, Average validation loss: {avg_val_loss:.3f}")
            print(f"Training Accuracy: {train_accuracy:.3f}%, Training F1-score: {train_f1:.3f}%")
            print(f"Validation Accuracy: {validation_accuracy:.3f}%, Validation F1-score: {validation_f1:.3f}%")
        
        self.plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, final_val_predictions, final_val_targets)   
        
        return best_model, best_params, best_accuracy
       

    def predict(self, x):
        y_pred = self.forward(x, training=False)
        return np.argmax(y_pred, axis=1) + 1
    
    def evaluate(self, loader):
        all_targets = []
        all_predictions = []

        for data, targets in loader:
            data = data.numpy().reshape(data.shape[0], -1)  # Reshape data
            targets = targets.numpy()  # Get true labels
            
            predictions = self.predict(data)  # Get model predictions

            # Store predictions and targets to calculate metrics later
            all_targets.extend(targets)
            all_predictions.extend(predictions)

        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) * 100
        
        # Calculate macro F1-score
        f1 = f1_score(all_targets, all_predictions, average='macro') * 100

        return accuracy, f1,  all_targets, all_predictions
    
    def plot_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies, final_val_preds, final_val_targets):
        # Plot training and validation loss
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot confusion matrix
        plt.subplot(1, 3, 3)
        conf_mat = confusion_matrix(final_val_targets, final_val_preds)
        plt.imshow(conf_mat, cmap='Blues', interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks(np.arange(len(np.unique(final_val_targets))), np.unique(final_val_targets))
        plt.yticks(np.arange(len(np.unique(final_val_targets))), np.unique(final_val_targets))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        plt.tight_layout()
        plt.show()
        


