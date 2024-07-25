import torch
import math
from torch.nn import functional
import matplotlib.pyplot as plt
from preprocessing import Preprocessing


SEQUENCE_LENGTH = 8
START_PITCH = 21
END_PITCH = 108
NUM_PITCHES = 108 - 21 + 1 + 1 # +1 for clear pitch 0
BATCH_SIZE = 16



class Model: 
    

    def setUpParameters(self):
        self.C = torch.randn((NUM_PITCHES,1),requires_grad=True)
        self.W1 = torch.randn((8,100),requires_grad=True)
        self.b1 = torch.randn(100,requires_grad=True)
        self.bnGain1 = torch.ones((1,100),requires_grad=True)
        self.bnBias1 = torch.zeros((1,100),requires_grad=True)

        self.W2 = torch.randn((100,80),requires_grad=True)
        self.b2 = torch.randn(80,requires_grad=True)
        self.bnGain2= torch.ones((1,80),requires_grad=True)
        self.bnBias2 = torch.zeros((1,80),requires_grad=True)

        self.W3 = torch.randn((80,NUM_PITCHES),requires_grad=True)
        self.b3 = torch.randn(NUM_PITCHES,requires_grad=True)
        self.bnGain3 = torch.ones((1,89),requires_grad=True) # added.
        self.bnBias3 = torch.zeros((1,89),requires_grad=True) # added.


        self.parameters = [self.C,self.W1,self.b1,self.bnGain1,self.bnBias1,self.W2,self.b2,self.bnGain2, self.bnBias2,self.W3,self.b3,self.bnGain3, self.bnBias3]


        for p in self.parameters: ## Not sure if this is necessary.
            p.requires_grad = True
                

    def make_predictions(self,x_batch):
        embedding = self.C[x_batch].squeeze()
        Linear1 = embedding @ self.W1 + self.b1
        BatchNorm1 = self.bnGain1 * ((Linear1 - Linear1.mean(0,keepdim=True)) / Linear1.std(0,keepdim=True)) +  self.bnBias1
        h1 = torch.tanh(BatchNorm1)
        Linear2 = h1 @ self.W2 + self.b2
        BatchNorm2 = self.bnGain2 * ((Linear2 - Linear2.mean(0,keepdim=True)) / Linear2.std(0,keepdim=True)) +  self.bnBias2
        h2 = torch.tanh(BatchNorm2)
        Linear3 = h2 @ self.W3 + self.b3  
        BatchNorm3 = self.bnGain3 * ((Linear3 - Linear3.mean(0,keepdim=True))) / Linear3.std(0,keepdim=True) + self.bnBias3 
        logits = BatchNorm3
        return logits

    def trainModel(self, preprocessor, epochs: int):

        pitches_train_x, pitches_train_y, pitches_test_x, pitches_test_y = preprocessor.getData()
        assert len(pitches_train_x) == len(pitches_train_y) and len(pitches_test_x == len(pitches_test_y))
        print(f"{len(pitches_train_x)} training data.")
        print(f"{len(pitches_test_x)} validation data.")



        num_batches = math.floor(pitches_train_x.size()[0] / BATCH_SIZE)
        # num_batches = 1 ## overfitting on batch

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            for i in range(num_batches):
                batch_start = i * BATCH_SIZE
                batch_end = batch_start + BATCH_SIZE
                ## forward pass:
                constant_x_batch = pitches_train_x[batch_start:batch_end].int()
                constant_y_batch = pitches_train_y[batch_start:batch_end].int()
                logits = self.make_predictions(constant_x_batch)
                loss = functional.cross_entropy(logits,constant_y_batch.long())

                ## reseting gradient:
                for p in self.parameters: 
                    p.grad = None

                #backprop
                loss.backward() ## calculates gradients.

                for p in self.parameters:
                    if epoch  <= 0.75 * epochs:
                        p.data -= 0.1 * p.grad
                    else:
                        p.data -= 0.01 * p.grad

            train_losses.append(loss.item())  

            #validation
            logits = self.make_predictions(pitches_test_x.int())
            validation_loss = functional.cross_entropy(logits,pitches_test_y.long())
            val_losses.append(validation_loss.item())


        print("Best training loss: ",min(train_losses) )
        print("Best validation loss: ", min(val_losses))

        self.plotLossCurve(train_losses,"Training loss")
        self.plotLossCurve(val_losses,"Validation loss")

    def plotLossCurve(self,loss,label):

        plt.plot(loss)
        plt.xlabel("Batch")
        plt.ylabel(label)
        plt.show()