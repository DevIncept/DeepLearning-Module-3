# Long Short-Term Memory (LSTM)

## **what is LSTM**

*LSTM is a recurrent neural network (RNN) architecture that REMEMBERS values over arbitrary intervals. LSTM is well-suited to classify, process and predict time series given time lags of unknown duration.*

## Architecture

![l1.png](attachment:l1.png)

*RNN cell takes in two inputs, output from the last hidden state and observation at time = t. Besides the hidden state, there is no information about the past to REMEMBER.*

![l2.png](attachment:l2.png)

![l3.png](attachment:l3.png)

![l.png](attachment:l.png)

*The long-term memory is usually called the cell state. The looping arrows indicate recursive nature of the cell. This allows information from previous intervals to be stored with in the LSTM cell. Cell state is modified by the forget gate placed below the cell state and also adjust by the input modulation gate. From equation, the previous cell state forgets by multiply with the forget gate and adds new information through the output of the input gates.*

![l4.png](attachment:l4.png)

![ll.png](attachment:ll.png)

*The remember vector is usually called the forget gate. The output of the forget gate tells the cell state which information to forget by multiplying 0 to a position in the matrix. If the output of the forget gate is 1, the information is kept in the cell state. From equation, sigmoid function is applied to the weighted input/observation and previous hidden state.*

![l5.png](attachment:l5.png)

![lll.png](attachment:lll.png)

![llll.png](attachment:llll.png)

*The save vector is usually called the input gate. These gates determine which information should enter the cell state / long-term memory. The important parts are the activation functions for each gates. The input gate is a sigmoid function and have a range of [0,1]. Because the equation of the cell state is a summation between the previous cell state, sigmoid function alone will only add memory and not be able to remove/forget memory. If you can only add a float number between [0,1], that number will never be zero / turned-off / forget. This is why the input modulation gate has an tanh activation function. Tanh has a range of [-1, 1] and allows the cell state to forget memory.*

![l6.png](attachment:l6.png)

![lllll.png](attachment:lllll.png)

*The focus vector is usually called the output gate*

![l7.png](attachment:l7.png)

![llllll.png](attachment:llllll.png)
*The working memory is usually called the hidden state*

![l8.png](attachment:l8.png)
*The first sigmoid activation function is the forget gate. Which information should be forgotten from the previous cell state (Ct-1). The second sigmoid and first tanh activation function is our input gate*

## Working

*Rather than go into the equations that govern how LSTMs are fit, analogy is a useful tool to quickly get a handle on how they work.*

![w1.PNG](attachment:w1.PNG)

![w2.PNG](attachment:w2.PNG)

![w3.PNG](attachment:w3.PNG)

## Practical intution

**→Neuron Activation**

*Here is our LSTM architecture. To start off, we are going to be looking at the tangent function tanh and how it fires up. As you remember, its value ranges from -1 to 1. In our further images, “-1” is going to be red and “+1” is going to be blue.*

![p1.png](attachment:p1.png)

*Below is the first example of LSTM “thinking”. The image includes a snippet from “War and Peace” by Leo Tolstoy. The text was given to RNN, and it learned to read it and predict what text is coming next.\
As you can see, this neuron is sensitive to position in line. When you get towards the end of the line, it is activating. How does it know that it is the end of the line? You have about 80 symbols per line in this novel. So, it’s counting how many symbols have passed and that’s the way it’s trying to predict when the new line character is coming up*

![p2.png](attachment:p2.png)

*The next cell recognizes direct speech. It’s keeping track of the quotation marks and is activating inside the quotes.\
This is very similar to our example where the network was keeping track of the subject to understand if it is male or female, singular or plural, and to suggest the correct verb forms for the translation. Here we observe the same logic. It’s important to know if you are inside or outside the quotes because that affects the rest of the text.*

![p3.png](attachment:p3.png)

*On the next image, we have a snippet from the code of the Linux operating system. This example refers to the cell that activates inside if-statements. It’s completely dormant everywhere else, but as soon as you have an if-statement, it activates. Then, it’s only active for the condition of the if-statement and it stops being active at the actual body of the if-statement. That’s can be important because you’re anticipating the body of the if-statement.*

![p4.png](attachment:p4.png)

*The next cell is sensitive to how deep you are inside of the nested expression. As you go deeper, and the expression gets more and more nested, this cell keeps track of that.*

![p5.png](attachment:p5.png)

*It’s very important to remember that none of these is actually hardcoded into the neural network. All of these is learned by the network itself through thousands and thousands of iterations.*

*The network kind of thinks: okay, I have this many hidden states, an out of them I need to identify, what’s important in a text to keep track off. Then, it identifies that in this particular text understanding how deep you’re inside a nested statement is important. Therefore, it assigns one of its hidden states, or memory cells, to keep track of that.*

*So, the network is really evolving on itself and deciding how to allocate its resources to best complete the task. That’s really fascinating!*

*The next image demonstrates an example of the cell that you can’t really understand, what it’s doing. According to Andrej Karpathy, about 95% of the cells are like this. They are doing something, but that’s just not obvious to humans, while it makes sense for machines.*

![p6.png](attachment:p6.png)

**Output**\
*Now let’s move to the actual output ht. This is the resulting value after it passed the tangent function and the output valve.*

![p7.png](attachment:p7.png)

*The first line shows us if the neuron is active (green color) or not (blue color), while the next five lines say us, what the neural network is predicting, particularly, what letter is going to come next. If it’s confident about its prediction, the color of the corresponding cell is red and if it’s not confident – it is light red.*

![p8.png](attachment:p8.png)

*The first row demonstrates the neuron’s activation inside the URL www.ynetnews.com. Then, below each of the letter you can see, what is the network’s prediction for the next letter.*

*For example, after the first “w” it’s pretty confident that the next letter will be “w” as well. Conversely, its prediction about the letter or symbol after “.” is very unsure because it could actually be any website.*

![p9.png](attachment:p9.png)

*As you see from the image, the network continues generating predictions even when the actual neuron is dormant. See, for example, how it was able to predict the word “language” just from the first two letters.*

*The neuron activates again in the third row, when another URL appears (see the image below). That’s quite an interesting case.*

*You can observe that the network was pretty sure that the next letter after “co” should be “m” to get “.com”, but it was another dot instead.*

*Then, the network predicted “u” because the domain “co.uk” (for the United Kingdom) is quite popular. And again, this was the wrong prediction because the actual domain was “co.il” (for Israel), which was not at all considered by the neural network even as 2nd, 3rd, 4th or 5th best guess.*

![p10.png](attachment:p10.png)


```python

```
