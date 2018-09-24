# Components

Components are the main building blocks of the models. In particular, the sequence-to-sequence type models are well-suited
for these modular elements. In the current state of the API, there are two distinct versions of the components, encoders and decoders.
Each of these have 3 different methods:

1. Recurrent
2. Convolutional
3. Quasi-Recurrent

![component visualization](https://github.com/Mrpatekful/nmt-BMEVIAUAL01/blob/master/data/img/components.png)

***

## Encoders

### Recurrent

1. Unidirectional encoder
2. Bidirectional encoder

Unidirectional type encoders could be considered as the regular recurrent units, which may yield
different methods for calculations. The currently implemented features are the LSTM-type units and GRUs.
The problem with these type of architectures, is the ability to preserve references in long sequences. Even
the LSTMs and GRUs can't seem to resolve dependencies in longer, 40-50 unit length sequences. As a solution
Bidirectional encoders start their operations from the end of the sentence, going 'backward' in time, as well as
from the start of the sequence. This way there are 2 hidden states, one for each direction, which will then be concatenated, and
fed to the upcoming layer. This method shortens the path between dependencies, and performs considerably better in numerous tasks.

### Convolutional

*COMING SOON*


### Quasi-Recurrent

*COMING SOON*

***

## Decoders

### Recurrent

1. Regular decoder
2. Attentional decoder
    1. Bahdanau-style
    2. Luong-style
        1. Dot Attention Decoder
        2. General Attention Decoder
        3. Concat Attention Decoder
        
#### Regular

Similarly to the encoders, the basic recurrent decoders also operate with an LSTM or GRU. The encoder provides
the starting hidden state for the component, that contains the encoded latent representation of the source language.
The decoder then starts to unfold this hidden state by predicting the first word of the target sentence, which
will be then fed to the decoder at the next time step. This phase goes until the decoder predicts an <EOS> token. Although 
this simple method is the fastest, there are other techniques, which provide much better performance.

#### Bahdanau-style Attention Decoder

Considering the method of translation from a human viewpoint, the encoder-decoder method of machine translation may
not be a very intuitive approach, since when the decoder tries to predict the most probable word at the first position,
it takes the whole encoded sentence into account. It would be much more natural, if the decoder would only consider those
parts of the source sentence, which highly correlate to the currently decoded word of the target sentence.
[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) introduces a
method for this approach, that is called attention, which is an existing techniques in image related machine learning tasks, but has not been
applied to natural language processing yet.

The main idea is to integrate another layer between the decoder and encoder, which will operate this mechanism. At each
decoding step this layer calculates a weight distribution over the outputs of the encoder at each encoding time step. The new encoded
latent state will come from the linear combination of the encoder hidden states, with their corresponding weights. 

![attention visualization](https://github.com/Mrpatekful/nmt-BMEVIAUAL01/blob/master/data/img/attention.png)

Although the core concept of attention is the same, there are different methods for calculating the weights for the encoder outputs.
The already mentioned Bahdanau-style method uses a trainable layer, which takes the concatenation of the investigated encoder
state and the decoder state the at the previous time step. The output of this operation is a single scalar value (or a vector
of values in case of batched calculations), that will be the weight for the used encoder state. After each of the encoder output
states have been weighted, the recurrent layer receives the weighted sum of these vectors.

#### Luong-style Attention Decoder

Another approach has been introduced by [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025), 
which alters the order of weight calculation, and defines several alternative techniques for the scoring mechanism.
Compared to the Bahdanau method `recurrent(score(h_t-1), i_t)` where i_t is the output of the previous decoding time step,
and h_t-1 is the hidden state of the previous time step, the Luong method calculates `score(recurrent(h_t-1, i_t))`. The scoring of
hidden state with respect to the encoder outputs, happens at the t-th time step, which will be used by a final output layer,
that projects the hidden state to the vocabulary, with a softmax activation.

As mentioned above, Luong introduced 2 new methods for weight calculation (score function) additional to the method
used in Bahdanau's work.

1. *Dot Attention Decoder*

The simplest and fastest scoring function, which according to my experience, helps the convergence of the model, better
than any other scoring methods. The weight simply comes from the dot product of the encoder state and the hidden state.

    h_d * h_eT

2. *General Attention Decoder*

This method uses a trainable layer similar to the concatenative methods, but instead of concatenation, it takes the dot
product of the weight layer, with the encoder output state, and then the dot product with the decoder hidden state.

    h_d * (W_a * h_eT)
    
The positive impact of this method compared to the simple dot product, is the constrain of creating a good attention
weight distribution over the encoder output states is the responsibility of the attention weight layer, instead of the
encoder and decoder recurrent layers.


3. *Concat Attention Decoder*

This method is the same as the one used in Bahdanau's experiments, which takes the concatenation of the decoder hidden state
and encoder output state, with a dedicated attention weight layer.

    v_t * tanh(W_a * [h_d ; h_e])


### Convolutional

*COMING SOON*


### Quasi-Recurrent

*COMING SOON*



