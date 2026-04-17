---
title: "Building makemore Part 2: MLP"
video_id: TCH_1BHY58I
url: "https://www.youtube.com/watch?v=TCH_1BHY58I"
author: Andrej Karpathy
slug: zero-to-hero-03-makemore-mlp
fetched_at: "2026-04-17T20:58:29+08:00"
type: youtube-transcript-whisper
transcript_source: "https://github.com/averkij/karcaps (003-large.html)"
segments: 896
---

# Building makemore Part 2: MLP

> Video: https://www.youtube.com/watch?v=TCH_1BHY58I
> Transcript: averkij/karcaps (Whisper-large) `003-large.html`

[00:00:00.000] Hi, everyone. Today we are continuing our implementation of MakeMore.

[00:00:05.000] Now, in the last lecture, we implemented the bigram language model,

[00:00:08.000] and we implemented it both using counts and also using a super simple neural network that has a single linear layer.

[00:00:15.000] Now, this is the Jupyter Notebook that we built out last lecture.

[00:00:20.000] And we saw that the way we approached this is that we looked at only the single previous character,

[00:00:24.000] and we predicted the distribution for the character that would go next in the sequence.

[00:00:29.000] And we did that by taking counts and normalizing them into probabilities so that each row here sums to one.

[00:00:37.000] Now, this is all well and good if you only have one character of previous context.

[00:00:42.000] And this works, and it's approachable.

[00:00:44.000] The problem with this model, of course, is that the predictions from this model are not very good

[00:00:49.000] because you only take one character of context.

[00:00:52.000] So the model didn't produce very name-like sounding things.

[00:00:56.000] Now, the problem with this approach, though, is that if we are to take more context into account

[00:01:01.000] when predicting the next character in a sequence, things quickly blow up.

[00:01:05.000] And this table, the size of this table, grows, and in fact it grows exponentially with the length of the context.

[00:01:11.000] Because if we only take a single character at a time, that's 27 possibilities of context.

[00:01:16.000] But if we take two characters in the past and try to predict a third one,

[00:01:20.000] suddenly the number of rows in this matrix, you can look at it that way, is 27 times 27.

[00:01:25.000] So there's 729 possibilities for what could have come in the context.

[00:01:30.000] If we take three characters as the context, suddenly we have 20,000 possibilities of context.

[00:01:37.000] And so that's just way too many rows of this matrix.

[00:01:41.000] It's way too few counts for each possibility.

[00:01:45.000] And the whole thing just kind of explodes and doesn't work very well.

[00:01:49.000] So that's why today we're going to move on to this bullet point here.

[00:01:52.000] And we're going to implement a multilayer perceptron model to predict the next character in a sequence.

[00:01:58.000] And this modeling approach that we're going to adopt follows this paper, Benjou et al. 2003.

[00:02:04.000] So I have the paper pulled up here.

[00:02:06.000] Now this isn't the very first paper that proposed the use of multilayer perceptrons or neural networks

[00:02:11.000] to predict the next character or token in a sequence.

[00:02:14.000] But it's definitely one that was very influential around that time.

[00:02:18.000] It is very often cited to stand in for this idea.

[00:02:21.000] And I think it's a very nice write-up.

[00:02:23.000] And so this is the paper that we're going to first look at and then implement.

[00:02:27.000] Now this paper has 19 pages.

[00:02:29.000] So we don't have time to go into the full detail of this paper, but I invite you to read it.

[00:02:33.000] It's very readable, interesting, and has a lot of interesting ideas in it as well.

[00:02:37.000] In the introduction, they describe the exact same problem I just described.

[00:02:41.000] And then to address it, they propose the following model.

[00:02:44.000] Now keep in mind that we are building a character-level language model.

[00:02:48.000] So we're working on the level of characters.

[00:02:50.000] In this paper, they have a vocabulary of 17,000 possible words,

[00:02:54.000] and they instead build a word-level language model.

[00:02:57.000] But we're going to still stick with the characters, but we'll take the same modeling approach.

[00:03:01.000] Now what they do is basically they propose to take every one of these words, 17,000 words,

[00:03:07.000] and they're going to associate to each word a, say, 30-dimensional feature vector.

[00:03:13.000] So every word is now embedded into a 30-dimensional space.

[00:03:18.000] You can think of it that way.

[00:03:19.000] So we have 17,000 points or vectors in a 30-dimensional space,

[00:03:24.000] and you might imagine that's very crowded. That's a lot of points for a very small space.

[00:03:29.000] Now in the beginning, these words are initialized completely randomly, so they're spread out at random.

[00:03:35.000] But then we're going to tune these embeddings of these words using backpropagation.

[00:03:40.000] So during the course of training of this neural network,

[00:03:42.000] these points or vectors are going to basically move around in this space.

[00:03:46.000] And you might imagine that, for example, words that have very similar meanings

[00:03:49.000] or that are indeed synonyms of each other might end up in a very similar part of the space.

[00:03:54.000] And conversely, words that mean very different things would go somewhere else in the space.

[00:03:59.000] Now their modeling approach otherwise is identical to ours.

[00:04:02.000] They are using a multilayer neural network to predict the next word given the previous words.

[00:04:07.000] And to train the neural network, they are maximizing the log likelihood of the training data just like we did.

[00:04:13.000] So the modeling approach itself is identical.

[00:04:15.000] Now here they have a concrete example of this intuition. Why does it work?

[00:04:20.000] Basically, suppose that, for example, you are trying to predict a dog was running in a blank.

[00:04:25.000] Now suppose that the exact phrase, a dog was running in a, has never occurred in the training data.

[00:04:31.000] And here you are at test time later, when the model is deployed somewhere,

[00:04:36.000] and it's trying to make a sentence, and it's saying a dog was running in a blank.

[00:04:41.000] And because it's never encountered this exact phrase in the training set, you're out of distribution, as we say.

[00:04:47.000] You don't have fundamentally any reason to suspect what might come next.

[00:04:54.000] But this approach actually allows you to get around that.

[00:04:57.000] Because maybe you didn't see the exact phrase, a dog was running in a something, but maybe you've seen similar phrases.

[00:05:02.000] Maybe you've seen the phrase, the dog was running in a blank.

[00:05:06.000] And maybe your network has learned that a and the frequently are interchangeable with each other.

[00:05:12.000] And so maybe it took the embedding for a and the embedding for the, and it actually put them nearby each other in the space.

[00:05:18.000] And so you can transfer knowledge through that embedding, and you can generalize in that way.

[00:05:23.000] Similarly, the network could know that cats and dogs are animals, and they co-occur in lots of very similar contexts.

[00:05:29.000] And so even though you haven't seen this exact phrase, or if you haven't seen exactly walking or running,

[00:05:35.000] you can, through the embedding space, transfer knowledge, and you can generalize to novel scenarios.

[00:05:42.000] So let's now scroll down to the diagram of the neural network.

[00:05:45.000] They have a nice diagram here.

[00:05:47.000] And in this example, we are taking three previous words, and we are trying to predict the fourth word in a sequence.

[00:05:56.000] Now these three previous words, as I mentioned, we have a vocabulary of 17,000 possible words.

[00:06:03.000] So every one of these basically are the index of the incoming word.

[00:06:09.000] And because there are 17,000 words, this is an integer between 0 and 16,999.

[00:06:17.000] Now there's also a lookup table that they call C.

[00:06:21.000] This lookup table is a matrix that is 17,000 by, say, 30.

[00:06:26.000] And basically what we're doing here is we're treating this as a lookup table.

[00:06:30.000] So every index is plucking out a row of this embedding matrix,

[00:06:35.000] so that each index is converted to the 30-dimensional vector that corresponds to the embedding vector for that word.

[00:06:43.000] So here we have the input layer of 30 neurons for three words, making up 90 neurons in total.

[00:06:51.000] And here they're saying that this matrix C is shared across all the words.

[00:06:55.000] So we're always indexing into the same matrix C over and over for each one of these words.

[00:07:02.000] Next up is the hidden layer of this neural network.

[00:07:05.000] The size of this hidden neural layer of this neural net is a hyperparameter.

[00:07:09.000] So we use the word hyperparameter when it's kind of like a design choice up to the designer of the neural net.

[00:07:14.000] And this can be as large as you'd like or as small as you'd like.

[00:07:17.000] So for example, the size could be 100.

[00:07:19.000] And we are going to go over multiple choices of the size of this hidden layer, and we're going to evaluate how well they work.

[00:07:26.000] So say there were 100 neurons here.

[00:07:28.000] All of them would be fully connected to the 90 words or 90 numbers that make up these three words.

[00:07:36.000] So this is a fully connected layer.

[00:07:38.000] Then there's a 10-inch linearity.

[00:07:40.000] And then there's this output layer.

[00:07:42.000] And because there are 17,000 possible words that could come next,

[00:07:46.000] this layer has 17,000 neurons, and all of them are fully connected to all of these neurons in the hidden layer.

[00:07:54.000] So there's a lot of parameters here because there's a lot of words.

[00:07:58.000] So most computation is here.

[00:07:59.000] This is the expensive layer.

[00:08:01.000] Now, there are 17,000 logits here.

[00:08:04.000] So on top of there, we have the softmax layer, which we've seen in our previous video as well.

[00:08:09.000] So every one of these logits is exponentiated, and then everything is normalized to sum to one.

[00:08:14.000] So then we have a nice probability distribution for the next word in the sequence.

[00:08:19.000] Now, of course, during training, we actually have the label.

[00:08:22.000] We have the identity of the next word in the sequence.

[00:08:25.000] That word or its index is used to pluck out the probability of that word.

[00:08:32.000] And then we are maximizing the probability of that word with respect to the parameters of this neural net.

[00:08:39.000] So the parameters are the weights and biases of this output layer, the weights and biases of the hidden layer,

[00:08:46.000] and the embedding lookup table C.

[00:08:49.000] And all of that is optimized using backpropagation.

[00:08:52.000] And these dashed arrows, ignore those.

[00:08:55.000] That represents a variation of a neural net that we are not going to explore in this video.

[00:08:59.000] So that's the setup, and now let's implement it.

[00:09:02.000] Okay, so I started a brand new notebook for this lecture.

[00:09:05.000] We are importing PyTorch, and we are importing matplotlib so we can create figures.

[00:09:10.000] Then I am reading all the names into a list of words like I did before, and I'm showing the first eight right here.

[00:09:18.000] Keep in mind that we have 32,000 in total.

[00:09:21.000] These are just the first eight.

[00:09:23.000] And then here I'm building out the vocabulary of characters and all the mappings from the characters as strings to integers and vice versa.

[00:09:31.000] Now the first thing we want to do is we want to compile the dataset for the neural network.

[00:09:35.000] And I had to rewrite this code.

[00:09:38.000] I'll show you in a second what it looks like.

[00:09:41.000] So this is the code that I created for the dataset creation.

[00:09:44.000] So let me first run it, and then I'll briefly explain how this works.

[00:09:48.000] So first we're going to define something called block size.

[00:09:51.000] And this is basically the context length of how many characters do we take to predict the next one.

[00:09:57.000] So here in this example we're taking three characters to predict the fourth one, so we have a block size of three.

[00:10:02.000] That's the size of the block that supports the prediction.

[00:10:06.000] Then here I'm building out the X and Y.

[00:10:10.000] The X are the input to the neural net, and the Y are the labels for each example inside X.

[00:10:17.000] Then I'm erasing over the first five words.

[00:10:20.000] I'm doing the first five just for efficiency while we are developing all the code.

[00:10:24.000] But then later we're going to come here and erase this so that we use the entire training set.

[00:10:29.000] So here I'm printing the word Emma.

[00:10:32.000] And here I'm basically showing the five examples that we can generate out of the single word Emma.

[00:10:41.000] So when we are given the context of just dot dot dot, the first character in the sequence is E.

[00:10:48.000] In this context, the label is M. When the context is this, the label is M, and so forth.

[00:10:54.000] And so the way I build this out is first I start with a padded context of just zero tokens.

[00:11:00.000] Then I iterate over all the characters.

[00:11:02.000] I get the character in the sequence, and I basically build out the array Y of this current character,

[00:11:08.000] and the array X which stores the current running context.

[00:11:12.000] And then here I print everything, and here I crop the context and enter the new character in the sequence.

[00:11:19.000] So this is kind of like a rolling window of context.

[00:11:23.000] Now we can change the block size here to, for example, four.

[00:11:26.000] And in that case we would be predicting the fifth character given the previous four.

[00:11:30.000] Or it can be five, and then it would look like this.

[00:11:34.000] Or it can be, say, ten, and then it would look something like this.

[00:11:38.000] We're taking ten characters to predict the eleventh one.

[00:11:41.000] And we're always padding with dots.

[00:11:43.000] So let me bring this back to three, just so that we have what we have here in the paper.

[00:11:50.000] And finally, the dataset right now looks as follows.

[00:11:53.000] From these five words, we have created a dataset of 32 examples.

[00:11:58.000] And each input to the neural net is three integers.

[00:12:01.000] And we have a label that is also an integer, Y.

[00:12:05.000] So X looks like this.

[00:12:07.000] These are the individual examples.

[00:12:09.000] And then Y are the labels.

[00:12:12.000] So given this, let's now write a neural network that takes these Xs and predicts the Ys.

[00:12:19.000] First, let's build the embedding lookup table C.

[00:12:23.000] So we have 27 possible characters, and we're going to embed them in a lower-dimensional space.

[00:12:28.000] In the paper, they have 17,000 words, and they embed them in spaces as small-dimensional as 30.

[00:12:36.000] So they cram 17,000 words into 30-dimensional space.

[00:12:40.000] In our case, we have only 27 possible characters.

[00:12:43.000] So let's cram them in something as small as, to start with, for example, a two-dimensional space.

[00:12:48.000] So this lookup table will be random numbers, and we'll have 27 rows, and we'll have two columns.

[00:12:56.000] So each one of 27 characters will have a two-dimensional embedding.

[00:13:01.000] So that's our matrix C of embeddings.

[00:13:05.000] In the beginning, initialized randomly.

[00:13:08.000] Now, before we embed all of the integers inside the input X using this lookup table C,

[00:13:14.000] let me actually just try to embed a single individual integer, like, say, 5.

[00:13:19.000] So we get a sense of how this works.

[00:13:22.000] Now, one way this works, of course, is we can just take the C and we can index into row 5.

[00:13:28.000] And that gives us a vector, the fifth row of C.

[00:13:32.000] And this is one way to do it.

[00:13:34.000] The other way that I presented in the previous lecture is actually seemingly different, but actually identical.

[00:13:40.000] So in the previous lecture, what we did is we took these integers and we used the one-hot encoding to first encode them.

[00:13:46.000] So f.onehot, we want to encode integer 5, and we want to tell it that the number of classes is 27.

[00:13:53.000] So that's the 26-dimensional vector of all zeros, except the fifth bit is turned on.

[00:13:59.000] Now, this actually doesn't work.

[00:14:02.000] The reason is that this input actually must be a Georgetown tensor.

[00:14:07.000] And I'm making some of these errors intentionally, just so you get to see some errors and how to fix them.

[00:14:12.000] So this must be a tensor, not an int. Fairly straightforward to fix.

[00:14:16.000] We get a one-hot vector, the fifth dimension is 1, and the shape of this is 27.

[00:14:22.000] And now notice that, just as I briefly alluded to in a previous video,

[00:14:26.000] if we take this one-hot vector and we multiply it by c, then what would you expect?

[00:14:37.000] Well, number one, first you'd expect an error, because expected scalar type long, but found float.

[00:14:46.000] So a little bit confusing, but the problem here is that one-hot, the data type of it, is long.

[00:14:54.000] It's a 64-bit integer, but this is a float tensor.

[00:14:57.000] And so PyTorch doesn't know how to multiply an int with a float,

[00:15:01.000] and that's why we had to explicitly cast this to a float so that we can multiply.

[00:15:06.000] Now, the output actually here is identical.

[00:15:11.000] And it's identical because of the way the matrix multiplication here works.

[00:15:15.000] We have the one-hot vector multiplying columns of c, and because of all the zeros,

[00:15:21.000] they actually end up masking out everything in c except for the fifth row, which is plucked out.

[00:15:27.000] And so we actually arrive at the same result.

[00:15:30.000] And that tells you that here we can interpret this first piece here, this embedding of the integer.

[00:15:35.000] We can either think of it as the integer indexing into a lookup table c,

[00:15:40.000] but equivalently we can also think of this little piece here as a first layer of this bigger neural net.

[00:15:46.000] This layer here has neurons that have no nonlinearity, there's no tanh, they're just linear neurons,

[00:15:52.000] and their weight matrix is c.

[00:15:55.000] And then we are encoding integers into one-hot and feeding those into a neural net,

[00:16:00.000] and this first layer basically embeds them.

[00:16:02.000] So those are two equivalent ways of doing the same thing.

[00:16:05.000] We're just going to index because it's much, much faster,

[00:16:08.000] and we're going to discard this interpretation of one-hot inputs into neural nets.

[00:16:13.000] And we're just going to index integers and use embedding tables.

[00:16:17.000] Now embedding a single integer like 5 is easy enough.

[00:16:20.000] We can simply ask PyTorch to retrieve the fifth row of c, or the row index 5 of c.

[00:16:27.000] But how do we simultaneously embed all of these 32 by 3 integers stored in array x?

[00:16:34.000] Luckily PyTorch indexing is fairly flexible and quite powerful.

[00:16:38.000] So it doesn't just work to ask for a single element 5 like this.

[00:16:44.000] You can actually index using lists.

[00:16:46.000] So for example, we can get the rows 5, 6, and 7, and this will just work like this.

[00:16:51.000] We can index with a list.

[00:16:53.000] It doesn't just have to be a list, it can also be actually a tensor of integers,

[00:16:58.000] and we can index with that.

[00:17:00.000] So this is an integer tensor 5, 6, 7, and this will just work as well.

[00:17:05.000] In fact, we can also, for example, repeat row 7 and retrieve it multiple times,

[00:17:10.000] and that same index will just get embedded multiple times here.

[00:17:15.000] So here we are indexing with a one-dimensional tensor of integers,

[00:17:20.000] but it turns out that you can also index with multi-dimensional tensors of integers.

[00:17:25.000] Here we have a two-dimensional tensor of integers.

[00:17:28.000] So we can simply just do c at x, and this just works.

[00:17:33.000] The shape of this is 32 by 3, which is the original shape,

[00:17:39.000] and now for every one of those 32 by 3 integers, we've retrieved the embedding vector here.

[00:17:45.000] So basically, we have that as an example.

[00:17:49.000] The example index 13, the second dimension, is the integer 1 as an example.

[00:17:58.000] And so here, if we do c of x, which gives us that array,

[00:18:03.000] and then we index into 13 by 2 of that array, then we get the embedding here.

[00:18:10.000] And you can verify that c at 1, which is the integer at that location, is indeed equal to this.

[00:18:19.000] You see they're equal.

[00:18:21.000] So basically, long story short, PyTorch indexing is awesome,

[00:18:25.000] and to embed simultaneously all of the integers in x, we can simply do c of x,

[00:18:31.000] and that is our embedding, and that just works.

[00:18:35.000] Now let's construct this layer here, the hidden layer.

[00:18:39.000] So we have that w1, as I'll call it, are these weights, which we will initialize randomly.

[00:18:46.000] Now the number of inputs to this layer is going to be 3 times 2,

[00:18:51.000] because we have two-dimensional embeddings and we have three of them,

[00:18:54.000] so the number of inputs is 6.

[00:18:56.000] And the number of neurons in this layer is a variable up to us.

[00:19:00.000] Let's use 100 neurons as an example.

[00:19:03.000] And then biases will be also initialized randomly as an example, and we just need 100 of them.

[00:19:11.000] Now the problem with this is we can't simply, normally we would take the input,

[00:19:16.000] in this case that's embedding, and we'd like to multiply it with these weights,

[00:19:21.000] and we'd like to add the bias. This is roughly what we want to do.

[00:19:24.000] But the problem here is that these embeddings are stacked up in the dimensions of this input tensor.

[00:19:30.000] So this will not work, this matrix multiplication, because this is a shape 32 by 3 by 2,

[00:19:34.000] and I can't multiply that by 6 by 100.

[00:19:37.000] So somehow we need to concatenate these inputs here together

[00:19:41.000] so that we can do something along these lines, which currently does not work.

[00:19:45.000] So how do we transform this 32 by 3 by 2 into a 32 by 6 so that we can actually perform this multiplication over here?

[00:19:54.000] I'd like to show you that there are usually many ways of implementing what you'd like to do in Torch.

[00:20:00.000] And some of them will be faster, better, shorter, etc.

[00:20:03.000] And that's because Torch is a very large library, and it's got lots and lots of functions.

[00:20:08.000] So if we just go to the documentation and click on Torch, you'll see that my slider here is very tiny,

[00:20:14.000] and that's because there are so many functions that you can call on these tensors

[00:20:18.000] to transform them, create them, multiply them, add them, perform all kinds of different operations on them.

[00:20:25.000] And so this is kind of like the space of possibility, if you will.

[00:20:31.000] Now one of the things that you can do is we can control F for concatenate.

[00:20:36.000] And we see that there's a function Torch.cat, short for concatenate.

[00:20:41.000] This concatenates a given sequence of tensors in a given dimension.

[00:20:45.000] And these tensors must have the same shape, etc.

[00:20:48.000] So we can use the concatenate operation to, in a naive way, concatenate these three embeddings for each input.

[00:20:56.000] So in this case we have m of the shape.

[00:21:00.000] And really what we want to do is we want to retrieve these three parts and concatenate them.

[00:21:05.000] So we want to grab all the examples.

[00:21:08.000] We want to grab first the zeroth index, and then all of this.

[00:21:17.000] So this plucks out the 32x2 embeddings of just the first word here.

[00:21:26.000] And so basically we want this guy, we want the first dimension, and we want the second dimension.

[00:21:32.000] And these are the three pieces individually.

[00:21:36.000] And then we want to treat this as a sequence, and we want to Torch.cat on that sequence.

[00:21:41.000] So this is the list.

[00:21:43.000] Torch.cat takes a sequence of tensors, and then we have to tell it along which dimension to concatenate.

[00:21:51.000] So in this case all these are 32x2, and we want to concatenate not across dimension 0, but across dimension 1.

[00:21:58.000] So passing in 1 gives us the result that the shape of this is 32x6, exactly as we'd like.

[00:22:05.000] So that basically took 32 and squashed these by concatenating them into 32x6.

[00:22:11.000] Now this is kind of ugly because this code would not generalize if we want to later change the block size.

[00:22:17.000] Right now we have three inputs, three words, but what if we had five?

[00:22:22.000] Then here we would have to change the code because I'm indexing directly.

[00:22:26.000] Well Torch comes to rescue again because there turns out to be a function called unbind.

[00:22:31.000] And it removes a tensor dimension.

[00:22:35.000] So it removes a tensor dimension, returns a tuple of all slices along a given dimension without it.

[00:22:41.000] So this is exactly what we need.

[00:22:44.000] And basically when we call torch.unbind of m and pass in dimension 1, index 1, this gives us a list of tensors exactly equivalent to this.

[00:23:02.000] So running this gives us a line 3, and it's exactly this list.

[00:23:09.000] So we can call torch.cat on it and along the first dimension.

[00:23:15.000] And this works, and the shape is the same.

[00:23:19.000] But now it doesn't matter if we have block size 3 or 5 or 10, this will just work.

[00:23:24.000] So this is one way to do it.

[00:23:26.000] But it turns out that in this case there's actually a significantly better and more efficient way.

[00:23:31.000] And this gives me an opportunity to hint at some of the internals of torch.tensor.

[00:23:36.000] So let's create an array here of elements from 0 to 17, and the shape of this is just 18.

[00:23:45.000] It's a single vector of 18 numbers.

[00:23:48.000] It turns out that we can very quickly re-represent this as different sized and dimensional tensors.

[00:23:54.000] We do this by calling a view, and we can say that actually this is not a single vector of 18.

[00:24:01.000] This is a 2 by 9 tensor, or alternatively this is a 9 by 2 tensor, or this is actually a 3 by 3 by 2 tensor.

[00:24:12.000] As long as the total number of elements here multiply to be the same, this will just work.

[00:24:18.000] And in PyTorch, this operation, calling.view, is extremely efficient.

[00:24:24.000] And the reason for that is that in each tensor there's something called the underlying storage.

[00:24:30.000] And the storage is just the numbers always as a one dimensional vector.

[00:24:35.000] And this is how this tensor is represented in the computer memory.

[00:24:38.000] It's always a one dimensional vector.

[00:24:41.000] But when we call.view, we are manipulating some of the attributes of that tensor that dictate how this one dimensional sequence is interpreted to be an n dimensional tensor.

[00:24:53.000] And so what's happening here is that no memory is being changed, copied, moved, or created when we call.view.

[00:24:59.000] The storage is identical, but when you call.view, some of the internal attributes of the view of this tensor are being manipulated and changed.

[00:25:09.000] In particular, there's something called storage offset, strides, and shapes, and those are manipulated so that this one dimensional sequence of bytes is seen as different n dimensional arrays.

[00:25:20.000] There's a blog post here from Eric called PyTorch internals where he goes into some of this with respect to tensor and how the view of a tensor is represented.

[00:25:30.000] And this is really just like a logical construct of representing the physical memory.

[00:25:35.000] And so this is a pretty good blog post that you can go into.

[00:25:39.000] I might also create an entire video on the internals of torch tensor and how this works.

[00:25:44.000] For here, we just note that this is an extremely efficient operation.

[00:25:48.000] And if I delete this and come back to our EMB, we see that the shape of our EMB is 32 by 3 by 2.

[00:25:56.000] But we can simply ask for PyTorch to view this instead as a 32 by 6.

[00:26:03.000] And the way that gets flattened into a 32 by 6 array just happens that these two get stacked up in a single row.

[00:26:13.000] And so that's basically the concatenation operation that we're after.

[00:26:17.000] And you can verify that this actually gives the exact same result as what we had before.

[00:26:22.000] So this is an element y equals, and you can see that all the elements of these two tensors are the same.

[00:26:27.000] And so we get the exact same result.

[00:26:30.000] So long story short, we can actually just come here, and if we just view this as a 32 by 6 instead, then this multiplication will work and give us the hidden states that we're after.

[00:26:44.000] So if this is h, then h slash shape is now the 100-dimensional activations for every one of our 32 examples.

[00:26:53.000] And this gives the desired result.

[00:26:55.000] Let me do two things here.

[00:26:57.000] Number one, let's not use 32.

[00:26:59.000] We can, for example, do something like EMB.shape at 0 so that we don't hardcode these numbers.

[00:27:07.000] And this would work for any size of this EMB.

[00:27:10.000] Or alternatively, we can also do negative 1.

[00:27:12.000] When we do negative 1, PyTorch will infer what this should be.

[00:27:16.000] Because the number of elements must be the same, and we're saying that this is 6, PyTorch will derive that this must be 32.

[00:27:22.000] Or whatever else it is if EMB is of different size.

[00:27:26.000] The other thing is here, one more thing I'd like to point out is here when we do the concatenation, this actually is much less efficient.

[00:27:37.000] Because this concatenation would create a whole new tensor with a whole new storage.

[00:27:42.000] So new memory is being created because there's no way to concatenate tensors just by manipulating the view attributes.

[00:27:48.000] So this is inefficient and creates all kinds of new memory.

[00:27:52.000] So let me delete this now.

[00:27:55.000] We don't need this.

[00:27:57.000] And here to calculate h, we want to also dot 10h of this to get our h.

[00:28:07.000] So these are now numbers between negative 1 and 1 because of the 10h.

[00:28:10.000] And we have that the shape is 32 by 100.

[00:28:14.000] And that is basically this hidden layer of activations here for every one of our 32 examples.

[00:28:20.000] Now there's one more thing I glossed over that we have to be very careful with, and that's this plus here.

[00:28:26.000] In particular, we want to make sure that the broadcasting will do what we like.

[00:28:31.000] The shape of this is 32 by 100, and B1's shape is 100.

[00:28:36.000] So we see that the addition here will broadcast these two.

[00:28:39.000] And in particular, we have 32 by 100 broadcasting to 100.

[00:28:44.000] So broadcasting will align on the right, create a fake dimension here.

[00:28:49.000] So this will become a 1 by 100 row vector.

[00:28:52.000] And then it will copy vertically for every one of these rows of 32 and do an element-wise addition.

[00:28:59.000] So in this case, the correcting will be happening because the same bias vector will be added to all the rows of this matrix.

[00:29:08.000] So that is correct. That's what we'd like.

[00:29:10.000] And it's always good practice to just make sure so that you don't shoot yourself in the foot.

[00:29:15.000] And finally, let's create the final layer here.

[00:29:18.000] Let's create W2 and B2.

[00:29:23.000] The input now is 100, and the output number of neurons will be for us 27 because we have 27 possible characters that come next.

[00:29:32.000] So the biases will be 27 as well.

[00:29:35.000] So therefore, the logits, which are the outputs of this neural net, are going to be H multiplied by W2 plus B2.

[00:29:47.000] Logits.shape is 32 by 27, and the logits look good.

[00:29:53.000] Now exactly as we saw in the previous video, we want to take these logits and we want to first exponentiate them to get our fake counts.

[00:30:00.000] And then we want to normalize them into a probability.

[00:30:03.000] So prob is counts divide, and now counts.sum along the first dimension and keep them as true, exactly as in the previous video.

[00:30:14.000] And so prob.shape now is 32 by 27, and you'll see that every row of prob sums to 1, so it's normalized.

[00:30:26.000] So that gives us the probabilities.

[00:30:28.000] Now, of course, we have the actual letter that comes next.

[00:30:31.000] And that comes from this array Y, which we created during the data set creation.

[00:30:37.000] So Y is this last piece here, which is the identity of the next character in the sequence that we'd like to now predict.

[00:30:44.000] So what we'd like to do now is just as in the previous video, we'd like to index into the rows of prob,

[00:30:50.000] and in each row we'd like to pluck out the probability assigned to the correct character, as given here.

[00:30:56.000] So first we have torch.arrange of 32, which is kind of like an iterator over numbers from 0 to 31.

[00:31:05.000] And then we can index into prob in the following way.

[00:31:08.000] prob in torch.arrange of 32, which iterates the rows, and then in each row we'd like to grab this column, as given by Y.

[00:31:18.000] So this gives the current probabilities, as assigned by this neural network with this setting of its weights, to the correct character in the sequence.

[00:31:27.000] And you can see here that this looks OK for some of these characters, like this is basically 0.2.

[00:31:32.000] But it doesn't look very good at all for many other characters, like this is 0.070's 1 probability, and so the network thinks that some of these are extremely unlikely.

[00:31:42.000] But of course we haven't trained a neural network yet, so this will improve, and ideally all of these numbers here of course are 1, because then we are correctly predicting the next character.

[00:31:53.000] Now just as in the previous video, we want to take these probabilities, we want to look at the lock probability,

[00:31:59.000] and then we want to look at the average lock probability and the negative of it to create the negative log likelihood loss.

[00:32:07.000] So the loss here is 17, and this is the loss that we'd like to minimize to get the network to predict the correct character in the sequence.

[00:32:16.000] OK, so I rewrote everything here and made it a bit more respectable.

[00:32:20.000] So here's our data set, here's all the parameters that we defined.

[00:32:24.000] I'm now using a generator to make it reproducible.

[00:32:27.000] I clustered all the parameters into a single list of parameters, so that for example it's easy to count them and see that in total we currently have about 3400 parameters.

[00:32:37.000] And this is the forward pass as we developed it, and we arrive at a single number here, the loss, that is currently expressing how well this neural network works with the current setting of parameters.

[00:32:48.000] Now I would like to make it even more respectable.

[00:32:51.000] So in particular, see these lines here, where we take the logits and we calculate the loss.

[00:32:57.000] We're not actually reinventing the wheel here.

[00:33:00.000] This is just classification, and many people use classification, and that's why there is a functional.crossentropy function in PyTorch to calculate this much more efficiently.

[00:33:11.000] So we could just simply call f.crossentropy, and we can pass in the logits, and we can pass in the array of targets y, and this calculates the exact same loss.

[00:33:22.000] So in fact we can simply put this here, and erase these three lines, and we're going to get the exact same result.

[00:33:30.000] Now there are actually many good reasons to prefer f.crossentropy over rolling your own implementation like this.

[00:33:36.000] I did this for educational reasons, but you'd never use this in practice.

[00:33:39.000] Why is that?

[00:33:40.000] Number one, when you use f.crossentropy, PyTorch will not actually create all these intermediate tensors, because these are all new tensors in memory, and all this is fairly inefficient to run like this.

[00:33:52.000] Instead, PyTorch will cluster up all these operations, and very often have fused kernels that very efficiently evaluate these expressions that are sort of like clustered mathematical operations.

[00:34:04.000] Number two, the backward pass can be made much more efficient, and not just because it's a fused kernel, but also analytically and mathematically it's often a very much simpler backward pass to implement.

[00:34:17.000] We actually sell this with micrograd.

[00:34:19.000] You see here when we implemented 10h, the forward pass of this operation to calculate the 10h was actually a fairly complicated mathematical expression.

[00:34:27.000] But because it's a clustered mathematical expression, when we did the backward pass, we didn't individually backward through the exp and the 2 times and the minus 1 and division, etc.

[00:34:37.000] We just said it's 1 minus t squared, and that's a much simpler mathematical expression.

[00:34:42.000] And we were able to do this because we're able to reuse calculations, and because we are able to mathematically and analytically derive the derivative, and often that expression simplifies mathematically, and so there's much less to implement.

[00:34:55.000] So not only can it be made more efficient because it runs in a fused kernel, but also because the expressions can take a much simpler form mathematically.

[00:35:04.000] So that's number one.

[00:35:06.000] Number two, under the hood, ftat cross entropy can also be significantly more numerically well behaved.

[00:35:14.000] Let me show you an example of how this works.

[00:35:17.000] Suppose we have a logits of negative 2, 3, 0, and 5, and then we are taking the exponent of it and normalizing it to sum to 1.

[00:35:27.000] So when logits take on these values, everything is well and good, and we get a nice probability distribution.

[00:35:33.000] Now consider what happens when some of these logits take on more extreme values, and that can happen during optimization of a neural network.

[00:35:40.000] Suppose that some of these numbers grow very negative, let's say negative 100, then actually everything will come out fine.

[00:35:47.000] We still get probabilities that are well behaved and they sum to 1 and everything is great.

[00:35:54.000] But because of the way the exp works, if you have very positive logits, let's say positive 100 in here, you actually start to run into trouble, and we get not a number here.

[00:36:04.000] And the reason for that is that these counts have an inf here.

[00:36:10.000] So if you pass in a very negative number to exp, you just get a very small number, very near zero, and that's fine.

[00:36:19.000] But if you pass in a very positive number, suddenly we run out of range in our floating point number that represents these counts.

[00:36:28.000] So basically we're taking e and we're raising it to the power of 100, and that gives us inf because we've run out of dynamic range on this floating point number that is count.

[00:36:37.000] And so we cannot pass very large logits through this expression.

[00:36:43.000] Now let me reset these numbers to something reasonable.

[00:36:47.000] The way PyTorch solved this is that you see how we have a well behaved result here.

[00:36:53.000] It turns out that because of the normalization here, you can actually offset logits by any arbitrary constant value that you want.

[00:37:00.000] So if I add 1 here, you actually get the exact same result.

[00:37:04.000] Or if I add 2, or if I subtract 3, any offset will produce the exact same probabilities.

[00:37:12.000] So because negative numbers are OK, but positive numbers can actually overflow this exp, what PyTorch does is it internally calculates the maximum value that occurs in the logits.

[00:37:22.000] And it subtracts it. So in this case it would subtract 5.

[00:37:26.000] And so therefore the greatest number in logits will become 0, and all the other numbers will become some negative numbers.

[00:37:32.000] And then the result of this is always well behaved.

[00:37:35.000] So even if we have 100 here previously, not good.

[00:37:39.000] But because PyTorch will subtract 100, this will work.

[00:37:43.000] And so there's many good reasons to call cross entropy.

[00:37:47.000] Number one, the forward pass can be much more efficient, the backward pass can be much more efficient, and also things can be much more numerically well behaved.

[00:37:56.000] OK, so let's now set up the training of this neural net.

[00:37:59.000] We have the forward pass.

[00:38:02.000] We don't need these, because then we have that loss is equal to the fact that cross entropy does the forward pass.

[00:38:09.000] Then we need the backward pass.

[00:38:11.000] First we want to set the gradients to be 0.

[00:38:14.000] So for p in parameters we want to make sure that p.grad is none, which is the same as setting it to 0 in PyTorch.

[00:38:21.000] And then loss.backward to populate those gradients.

[00:38:24.000] Once we have the gradients we can do the parameter update.

[00:38:27.000] So for p in parameters we want to take all the data, and we want to nudge it learning rate times p.grad.

[00:38:36.000] And then we want to repeat this a few times.

[00:38:41.000] And let's print the loss here as well.

[00:38:48.000] Now this won't suffice, and it will create an error, because we also have to go for p in parameters,

[00:38:54.000] and we have to make sure that p.requiresGrad is set to true in PyTorch.

[00:38:59.000] And this should just work.

[00:39:03.000] OK, so we started off with loss of 17, and we're decreasing it.

[00:39:08.000] Let's run longer.

[00:39:10.000] And you see how the loss decreases a lot here.

[00:39:17.000] If we just run for a thousand times we get a very, very low loss.

[00:39:21.000] And that means that we're making very good predictions.

[00:39:23.000] Now the reason that this is so straightforward right now is because we're only overfitting 32 examples.

[00:39:32.000] So we only have 32 examples of the first five words,

[00:39:36.000] and therefore it's very easy to make this neural net fit only these 32 examples,

[00:39:41.000] because we have 3400 parameters and only 32 examples.

[00:39:46.000] So we're doing what's called overfitting a single batch of the data,

[00:39:50.000] and getting a very low loss and good predictions.

[00:39:54.000] But that's just because we have so many parameters for so few examples, so it's easy to make this be very low.

[00:40:00.000] Now we're not able to achieve exactly zero.

[00:40:03.000] The reason for that is we can, for example, look at logits, which are being predicted.

[00:40:09.000] And we can look at the max along the first dimension.

[00:40:14.000] And in PyTorch, max reports both the actual values that take on the maximum number, but also the indices of these.

[00:40:22.000] And you'll see that the indices are very close to the labels, but in some cases they differ.

[00:40:29.000] For example, in this very first example, the predicted index is 19, but the label is 5.

[00:40:35.000] And we're not able to make loss be zero, and fundamentally that's because here,

[00:40:40.000] the very first or the zeroth index is the example where dot dot dot is supposed to predict e,

[00:40:46.000] but you see how dot dot dot is also supposed to predict an o,

[00:40:49.000] and dot dot dot is also supposed to predict an i, and then s as well.

[00:40:53.000] And so basically e, o, a, or s are all possible outcomes in a training set for the exact same input.

[00:41:00.000] So we're not able to completely overfit and make the loss be exactly zero,

[00:41:06.000] but we're getting very close in the cases where there's a unique input for a unique output.

[00:41:12.000] In those cases we do what's called overfit, and we basically get the exact correct result.

[00:41:19.000] So now all we have to do is we just need to make sure that we read in the full dataset and optimize the neural net.

[00:41:25.000] Okay, so let's swing back up where we created the dataset, and we see that here we only used the first five words.

[00:41:31.000] So let me now erase this, and let me erase the print statements, otherwise we'd be printing way too much.

[00:41:38.000] And so when we process the full dataset of all the words, we now had 228,000 examples instead of just 32.

[00:41:45.000] So let's now scroll back down, the dataset is much larger, reinitialize the weights, the same number of parameters,

[00:41:52.000] they all require gradients, and then let's push this print.loss.item to be here,

[00:41:58.000] and let's just see how the optimization goes if we run this.

[00:42:04.000] Okay, so we started with a fairly high loss, and then as we're optimizing, the loss is coming down.

[00:42:12.000] But you'll notice that it takes quite a bit of time for every single iteration, so let's actually address that.

[00:42:17.000] Because we're doing way too much work forwarding and backwarding 228,000 examples.

[00:42:22.000] In practice what people usually do is they perform forward and backward pass and update on many batches of the data.

[00:42:29.000] So what we will want to do is we want to randomly select some portion of the dataset, and that's a mini-batch,

[00:42:35.000] and then only forward, backward, and update on that little mini-batch, and then we iterate on those mini-batches.

[00:42:41.000] So in PyTorch we can, for example, use tors.randint, and we can generate numbers between 0 and 5 and make 32 of them.

[00:42:52.000] I believe the size has to be a tuple in PyTorch.

[00:42:57.000] So we can have a tuple of 32 numbers between 0 and 5, but actually we want x.shape of 0 here.

[00:43:05.000] And so this creates integers that index into our dataset, and there's 32 of them.

[00:43:11.000] So if our mini-batch size is 32, then we can come here and we can first do mini-batch construct.

[00:43:20.000] So integers that we want to optimize in this single iteration are in the ix,

[00:43:27.000] and then we want to index into x with ix to only grab those rows.

[00:43:34.000] So we're only getting 32 rows of x, and therefore embeddings will again be 32 by 3 by 2, not 200,000 by 3 by 2.

[00:43:43.000] And then this ix has to be used not just to index into x, but also to index into y.

[00:43:50.000] And now this should be mini-batches, and this should be much, much faster, so it's instant almost.

[00:43:58.000] So this way we can run many, many examples nearly instantly and decrease the loss much, much faster.

[00:44:05.000] Now because we're only dealing with mini-batches, the quality of our gradient is lower.

[00:44:10.000] So the direction is not as reliable. It's not the actual gradient direction.

[00:44:14.000] But the gradient direction is good enough, even when it's estimating on only 32 examples, that it is useful.

[00:44:21.000] And so it's much better to have an approximate gradient and just make more steps than it is to evaluate the exact gradient and take fewer steps.

[00:44:30.000] So that's why in practice this works quite well.

[00:44:34.000] So let's now continue the optimization.

[00:44:38.000] Let me take out this lost item from here and place it over here at the end.

[00:44:45.000] So we're hovering around 2.5 or so. However, this is only the loss for that mini-batch.

[00:44:52.000] So let's actually evaluate the loss here for all of x and for all of y, just so we have a full sense of exactly how well the model is doing right now.

[00:45:04.000] So right now we're at about 2.7 on the entire training set.

[00:45:09.000] So let's run the optimization for a while. OK, we're at 2.6, 2.57, 2.53.

[00:45:22.000] OK, so one issue, of course, is we don't know if we're stepping too slow or too fast.

[00:45:28.000] So this point one, I just guessed it.

[00:45:31.000] So one question is, how do you determine this learning rate and how do we gain confidence that we're stepping in the right sort of speed?

[00:45:40.000] So I'll show you one way to determine a reasonable learning rate.

[00:45:43.000] It works as follows. Let's reset our parameters to the initial settings.

[00:45:51.000] And now let's print in every step, but let's only do 10 steps or so, or maybe 100 steps.

[00:46:01.000] We want to find a very reasonable search range, if you will.

[00:46:05.000] So for example, if this is very low, then we see that the loss is barely decreasing.

[00:46:12.000] So that's too low, basically.

[00:46:15.000] So let's try this one. OK, so we're decreasing the loss, but not very quickly.

[00:46:21.000] So that's a pretty good low range.

[00:46:23.000] Now let's reset it again.

[00:46:26.000] And now let's try to find the place at which the loss kind of explodes.

[00:46:29.000] So maybe at negative one.

[00:46:33.000] OK, we see that we're minimizing the loss, but you see how it's kind of unstable.

[00:46:37.000] It goes up and down quite a bit.

[00:46:40.000] So negative one is probably like a fast learning rate.

[00:46:43.000] Let's try negative 10.

[00:46:46.000] OK, so this isn't optimizing. This is not working very well.

[00:46:49.000] So negative 10 is way too big.

[00:46:51.000] Negative one was already kind of big.

[00:46:55.000] Therefore, negative one was somewhat reasonable if I reset.

[00:47:00.000] So I'm thinking that the right learning rate is somewhere between negative 0.001 and negative one.

[00:47:08.000] So the way we can do this here is we can use torch.lenspace.

[00:47:13.000] And we want to basically do something like this, between 0 and 1.

[00:47:17.000] But number of steps is one more parameter that's required.

[00:47:22.000] Let's do 1000 steps.

[00:47:24.000] This creates 1000 numbers between 0.001 and 1.

[00:47:30.000] But it doesn't really make sense to step between these linearly.

[00:47:33.000] So instead, let me create learning rate exponent.

[00:47:36.000] And instead of 0.001, this will be a negative 3, and this will be a 0.

[00:47:41.000] And then the actual LRs that we want to search over are going to be 10 to the power of LRE.

[00:47:48.000] So now what we're doing is we're stepping linearly between the exponents of these learning rates.

[00:47:52.000] This is 0.001, and this is 1, because 10 to the power of 0 is 1.

[00:47:58.000] And therefore, we are spaced exponentially in this interval.

[00:48:02.000] So these are the candidate learning rates that we want to search over, roughly.

[00:48:08.000] So now what we're going to do is here, we are going to run the optimization for 1000 steps.

[00:48:15.000] And instead of using a fixed number, we are going to use learning rate indexing into here, LRs of i, and make this i.

[00:48:26.000] So basically, let me reset this to be, again, starting from random, creating these learning rates between 0.001 and 1, but exponentially stepped.

[00:48:39.000] And here what we're doing is we're iterating 1000 times.

[00:48:43.000] We're going to use the learning rate that's in the beginning very, very low.

[00:48:48.000] In the beginning it's going to be 0.001, but by the end it's going to be 1.

[00:48:53.000] And then we're going to step with that learning rate.

[00:48:57.000] And now what we want to do is we want to keep track of the learning rates that we used.

[00:49:05.000] And we want to look at the losses that resulted.

[00:49:09.000] And so here, let me track stats.

[00:49:14.000] So LRI.append LR, and loss i.append loss.item.

[00:49:23.000] So again, reset everything, and then run.

[00:49:30.000] And so basically, we started with a very low learning rate, and we went all the way up to a learning rate of negative 1.

[00:49:36.000] And now what we can do is we can PLT.plot, and we can plot the two.

[00:49:40.000] So we can plot the learning rates on the x-axis, and the losses we saw on the y-axis.

[00:49:45.000] And often you're going to find that your plot looks something like this.

[00:49:49.000] Where in the beginning you had very low learning rates, so basically barely anything happened.

[00:49:56.000] Then we got to a nice spot here, and then as we increased the learning rate enough, we basically started to be kind of unstable here.

[00:50:05.000] So a good learning rate turns out to be somewhere around here.

[00:50:10.000] And because we have LRI here, we actually may want to do not the learning rate, but the exponent.

[00:50:22.000] So that would be the LRE at i is maybe what we want to log.

[00:50:26.000] So let me reset this and redo that calculation.

[00:50:30.000] But now on the x-axis, we have the exponent of the learning rate.

[00:50:35.000] And so we can see the exponent of the learning rate that is good to use.

[00:50:38.000] It would be sort of roughly in the valley here, because here the learning rates are just way too low.

[00:50:43.000] And then here we expect relatively good learning rates, somewhere here.

[00:50:47.000] And then here things are starting to explode.

[00:50:49.000] So somewhere around negative 1 as the exponent of the learning rate is a pretty good setting.

[00:50:54.000] And 10 to the negative 1 is 0.1.

[00:50:57.000] So 0.1 was actually a fairly good learning rate around here.

[00:51:02.000] And that's what we had in the initial setting.

[00:51:05.000] But that's roughly how you would determine it.

[00:51:08.000] And so here now we can take out the tracking of these.

[00:51:13.000] And we can just simply set LR to be 10 to the negative 1, or basically otherwise 0.1, as it was before.

[00:51:21.000] And now we have some confidence that this is actually a fairly good learning rate.

[00:51:24.000] And so now what we can do is we can crank up the iterations.

[00:51:27.000] We can reset our optimization.

[00:51:30.000] And we can run for a pretty long time using this learning rate.

[00:51:36.000] Oops, and we don't want to print. It's way too much printing.

[00:51:40.000] So let me again reset and run 10,000 steps.

[00:51:48.000] So we're at 2.48 roughly. Let's run another 10,000 steps.

[00:51:58.000] 2.46.

[00:52:00.000] And now let's do one learning rate decay.

[00:52:02.000] What this means is we're going to take our learning rate and we're going to 10x lower it.

[00:52:06.000] And so we're at the late stages of training potentially.

[00:52:09.000] And we may want to go a bit slower.

[00:52:12.000] Let's do one more actually at 0.1 just to see if we're making a dent here.

[00:52:19.000] Okay, we're still making a dent.

[00:52:20.000] And by the way, the bigram loss that we achieved last video was 2.45.

[00:52:25.000] So we've already surpassed the bigram model.

[00:52:29.000] And once I get a sense that this is actually kind of starting to plateau off,

[00:52:32.000] people like to do, as I mentioned, this learning rate decay.

[00:52:35.000] So let's try to decay the learning rate.

[00:52:42.000] And we achieve about 2.3 now.

[00:52:46.000] Obviously, this is janky and not exactly how you would train it in production.

[00:52:50.000] But this is roughly what you're going through.

[00:52:52.000] You first find a decent learning rate using the approach that I showed you.

[00:52:55.000] Then you start with that learning rate and you train for a while.

[00:52:58.000] And then at the end, people like to do a learning rate decay

[00:53:01.000] where you decay the learning rate by, say, a factor of 10 and you do a few more steps.

[00:53:05.000] And then you get a trained network, roughly speaking.

[00:53:08.000] So we achieved 2.3 and dramatically improved on the bigram language model

[00:53:13.000] using this simple neural net as described here, using these 3400 parameters.

[00:53:20.000] Now there's something we have to be careful with.

[00:53:22.000] I said that we have a better model because we are achieving a lower loss,

[00:53:26.000] 2.3 much lower than 2.45 with the bigram model previously.

[00:53:30.000] Now that's not exactly true.

[00:53:32.000] And the reason that's not true is that this is actually a fairly small model.

[00:53:39.000] But these models can get larger and larger if you keep adding neurons and parameters.

[00:53:43.000] So you can imagine that we don't potentially have 1000 parameters.

[00:53:46.000] We could have 10,000 or 100,000 or millions of parameters.

[00:53:49.000] And as the capacity of the neural network grows,

[00:53:52.000] it becomes more and more capable of overfitting your training set.

[00:53:56.000] What that means is that the loss on the training set, on the data that you're training on,

[00:54:01.000] will become very, very low, as low as zero.

[00:54:04.000] But all that the model is doing is memorizing your training set verbatim.

[00:54:08.000] So if you take that model and it looks like it's working really well,

[00:54:11.000] but you try to sample from it, you will basically only get examples

[00:54:15.000] exactly as they are in the training set.

[00:54:17.000] You won't get any new data.

[00:54:19.000] In addition to that, if you try to evaluate the loss on some withheld names or other words,

[00:54:24.000] you will actually see that the loss on those can be very high.

[00:54:28.000] And so basically it's not a good model.

[00:54:30.000] So the standard in the field is to split up your data set into three splits, as we call them.

[00:54:35.000] We have the training split, the dev split or the validation split, and the test split.

[00:54:41.000] So training split, dev or validation split, and test split.

[00:54:51.000] And typically this would be say 80% of your data set.

[00:54:55.000] This could be 10% and this 10% roughly.

[00:54:58.000] So you have these three splits of the data.

[00:55:01.000] Now these 80% of your trainings of the data set, the training set,

[00:55:05.000] is used to optimize the parameters of the model, just like we're doing here using gradient descent.

[00:55:10.000] These 10% of the examples, the dev or validation split,

[00:55:14.000] they're used for development over all the hyperparameters of your model.

[00:55:19.000] So hyperparameters are, for example, the size of this hidden layer, the size of the embedding.

[00:55:24.000] So this is 100 or a 2 for us, but we could try different things.

[00:55:28.000] The strength of the regularization, which we aren't using yet so far.

[00:55:32.000] So there's lots of different hyperparameters and settings that go into defining a neural net.

[00:55:36.000] And you can try many different variations of them and see whichever one works best on your validation split.

[00:55:43.000] So this is used to train the parameters.

[00:55:46.000] This is used to train the hyperparameters.

[00:55:49.000] And test split is used to evaluate basically the performance of the model at the end.

[00:55:54.000] So we're only evaluating the loss on the test split very sparingly and very few times.

[00:55:59.000] Because every single time you evaluate your test loss and you learn something from it,

[00:56:04.000] you are basically starting to also train on the test split.

[00:56:08.000] So you are only allowed to test the loss on the test set very, very few times.

[00:56:14.000] Otherwise, you risk overfitting to it as well as you experiment on your model.

[00:56:19.000] So let's also split up our training data into train, dev, and test.

[00:56:24.000] And then we are going to train on train and only evaluate on test very, very sparingly.

[00:56:29.000] Okay, so here we go.

[00:56:31.000] Here is where we took all the words and put them into X and Y tensors.

[00:56:36.000] So instead, let me create a new cell here.

[00:56:38.000] And let me just copy-paste some code here because I don't think it's that complex.

[00:56:44.000] But we're going to try to save a little bit of time.

[00:56:47.000] I'm converting this to be a function now.

[00:56:49.000] And this function takes some list of words and builds the arrays X and Y for those words only.

[00:56:56.000] And then here I am shuffling up all the words.

[00:56:59.000] So these are the input words that we get.

[00:57:02.000] We are randomly shuffling them all up.

[00:57:04.000] And then we're going to set N1 to be the number of examples that is 80% of the words and N2 to be 90% of the words.

[00:57:16.000] So basically if length of words is 30,000, N1 is 25,000 and N2 is 28,000.

[00:57:28.000] And so here we see that I'm calling buildDataset() to build the training set X and Y by indexing into up to N1.

[00:57:36.000] So we're going to have only 25,000 training words.

[00:57:39.000] And then we're going to have roughly N2 minus N1, 3000 validation examples or dev examples.

[00:57:50.000] And we're going to have length of words basically minus N2 or 3204 examples here for the test set.

[00:58:03.000] So now we have Xs and Ys for all those three splits.

[00:58:13.000] Oh yeah, I'm printing their size here inside the function as well.

[00:58:19.000] But here we don't have words, but these are already the individual examples made from those words.

[00:58:25.000] So let's now scroll down here.

[00:58:28.000] And the data set now for training is more like this.

[00:58:33.000] And then when we reset the network, when we're training, we're only going to be training using X train, X train, and Y train.

[00:58:48.000] So that's the only thing we're training on.

[00:58:58.000] Let's see where we are on a single batch.

[00:59:02.000] Let's now train maybe a few more steps.

[00:59:08.000] Training neural networks can take a while.

[00:59:10.000] Usually you don't do it inline.

[00:59:11.000] You launch a bunch of jobs and you wait for them to finish.

[00:59:14.000] It can take multiple days and so on.

[00:59:17.000] But basically this is a very small network.

[00:59:22.000] So the loss is pretty good.

[00:59:24.000] Oh, we accidentally used a learning rate that is way too low.

[00:59:28.000] So let me actually come back.

[00:59:30.000] We used the decay learning rate of 0.01.

[00:59:36.000] So this will train much faster.

[00:59:38.000] And then here when we evaluate, let's use the dev set here.

[00:59:43.000] And Y dev to evaluate the loss.

[00:59:48.000] And let's now decay the learning rate and only do say 10,000 examples.

[00:59:55.000] And let's evaluate the dev loss once here.

[00:59:59.000] So we're getting about 2.3 on dev.

[01:00:01.000] And so the neural network when it was training did not see these dev examples.

[01:00:05.000] It hasn't optimized on them.

[01:00:07.000] And yet when we evaluate the loss on these dev, we actually get a pretty decent loss.

[01:00:12.000] And so we can also look at what the loss is on all of training set.

[01:00:21.000] And so we see that the training and the dev loss are about equal.

[01:00:24.000] So we're not overfitting.

[01:00:26.000] This model is not powerful enough to just be purely memorizing the data.

[01:00:30.000] And so far we are what's called underfitting because the training loss and the dev or test losses are roughly equal.

[01:00:38.000] So what that typically means is that our network is very tiny, very small.

[01:00:43.000] And we expect to make performance improvements by scaling up the size of this neural net.

[01:00:48.000] So let's do that now.

[01:00:49.000] So let's come over here and let's increase the size of the neural net.

[01:00:53.000] The easiest way to do this is we can come here to the hidden layer, which currently has 100 neurons.

[01:00:57.000] And let's just bump this up.

[01:00:58.000] So let's do 300 neurons.

[01:01:01.000] And then this is also 300 biases.

[01:01:03.000] And here we have 300 inputs into the final layer.

[01:01:07.000] So let's initialize our neural net.

[01:01:10.000] We now have 10,000 parameters instead of 3,000 parameters.

[01:01:15.000] And then we're not using this.

[01:01:18.000] And then here what I'd like to do is I'd like to actually keep track of that.

[01:01:27.000] OK, let's just do this.

[01:01:29.000] Let's keep stats again.

[01:01:31.000] And here when we're keeping track of the loss, let's just also keep track of the steps.

[01:01:39.000] And let's just have an eye here.

[01:01:41.000] And let's train on 30,000.

[01:01:44.000] Or rather say, let's try 30,000.

[01:01:48.000] And we are at 0.1.

[01:01:51.000] And we should be able to run this and optimize the neural net.

[01:01:57.000] And then here basically I want to plt.plot the steps against the loss.

[01:02:09.000] So these are the x's and the y's.

[01:02:11.000] And this is the loss function and how it's being optimized.

[01:02:16.000] Now you see that there's quite a bit of thickness to this.

[01:02:19.000] And that's because we are optimizing over these mini-batches.

[01:02:22.000] And the mini-batches create a little bit of noise in this.

[01:02:26.000] Where are we in the defset?

[01:02:28.000] We are at 2.5.

[01:02:29.000] So we still haven't optimized this neural net very well.

[01:02:32.000] And that's probably because we made it bigger.

[01:02:34.000] It might take longer for this neural net to converge.

[01:02:37.000] And so let's continue training.

[01:02:43.000] Yeah, let's just continue training.

[01:02:47.000] One possibility is that the batch size is so low that we just have way too much noise in the training.

[01:02:53.000] And we may want to increase the batch size so that we have a bit more correct gradient.

[01:02:57.000] And we're not thrashing too much.

[01:02:59.000] And we can actually optimize more properly.

[01:03:09.000] This will now become meaningless because we've reinitialized these.

[01:03:12.000] So this looks not pleasing right now.

[01:03:16.000] But the problem is a tiny improvement, but it's so hard to tell.

[01:03:20.000] Let's go again.

[01:03:23.000] 2.52.

[01:03:25.000] Let's try to decrease the learning rate by a factor of two.

[01:03:50.000] Okay, we're at 2.32.

[01:03:52.000] Let's continue training.

[01:04:05.000] We basically expect to see a lower loss than what we had before.

[01:04:08.000] Because now we have a much, much bigger model.

[01:04:10.000] And we were underfitting.

[01:04:12.000] So we'd expect that increasing the size of the model should help the neural net.

[01:04:16.000] 2.32.

[01:04:17.000] Okay, so that's not happening too well.

[01:04:19.000] Now, one other concern is that even though we've made the hidden layer much, much bigger,

[01:04:25.000] it could be that the bottleneck of the network right now are these embeddings that are two-dimensional.

[01:04:30.000] It can be that we're just cramming way too many characters into just two dimensions.

[01:04:34.000] And the neural net is not able to really use that space effectively.

[01:04:38.000] And that is sort of like the bottleneck to our network's performance.

[01:04:42.000] Okay, 2.23.

[01:04:44.000] So just by decreasing the learning rate, I was able to make quite a bit of progress.

[01:04:47.000] Let's run this one more time.

[01:04:51.000] And then evaluate the training and the dev loss.

[01:04:56.000] Now, one more thing after training that I'd like to do is I'd like to visualize the embedding vectors for these characters

[01:05:06.000] before we scale up the embedding size from 2.

[01:05:10.000] Because we'd like to make this bottleneck potentially go away.

[01:05:13.000] But once I make this greater than 2, we won't be able to visualize them.

[01:05:17.000] So here, we're at 2.23 and 2.24.

[01:05:21.000] So we're not improving much more.

[01:05:24.000] And maybe the bottleneck now is the character embedding size, which is 2.

[01:05:28.000] So here I have a bunch of code that will create a figure.

[01:05:31.000] And then we're going to visualize the embeddings that were trained by the neural net on these characters.

[01:05:38.000] Because right now the embedding size is just 2.

[01:05:40.000] So we can visualize all the characters with the X and the Y coordinates as the two embedding locations for each of these characters.

[01:05:48.000] And so here are the X coordinates and the Y coordinates, which are the columns of C.

[01:05:53.000] And then for each one, I also include the text of the little character.

[01:05:58.000] So here what we see is actually kind of interesting.

[01:06:02.000] The network has basically learned to separate out the characters and cluster them a little bit.

[01:06:08.000] So for example, you see how the vowels A, E, I, O, U are clustered up here?

[01:06:13.000] So what that's telling us is that the neural net treats these as very similar, right?

[01:06:17.000] Because when they feed into the neural net, the embedding for all these characters is very similar.

[01:06:22.000] And so the neural net thinks that they're very similar and kind of interchangeable, if that makes sense.

[01:06:29.000] Then the points that are really far away are, for example, Q.

[01:06:33.000] Q is kind of treated as an exception, and Q has a very special embedding vector, so to speak.

[01:06:38.000] Similarly, dot, which is a special character, is all the way out here.

[01:06:42.000] And a lot of the other letters are sort of clustered up here.

[01:06:46.000] And so it's kind of interesting that there's a little bit of structure here after the training.

[01:06:51.000] And it's definitely not random, and these embeddings make sense.

[01:06:56.000] So we're now going to scale up the embedding size and won't be able to visualize it directly.

[01:07:00.000] But we expect that, because we're underfitting, and we made this layer much bigger and did not sufficiently improve the loss,

[01:07:08.000] we're thinking that the constraint to better performance right now could be these embedding vectors.

[01:07:15.000] So let's make them bigger.

[01:07:16.000] So let's scroll up here, and now we don't have two-dimensional embeddings.

[01:07:19.000] We are going to have, say, ten-dimensional embeddings for each word.

[01:07:25.000] Then this layer will receive 3 times 10, so 30 inputs will go into the hidden layer.

[01:07:36.000] Let's also make the hidden layer a bit smaller.

[01:07:38.000] So instead of 300, let's just do 200 neurons in that hidden layer.

[01:07:42.000] So now the total number of elements will be slightly bigger, at 11,000.

[01:07:47.000] And then here we have to be a bit careful, because the learning rate we set to 0.1.

[01:07:53.000] Here we are hardcoding 6.

[01:07:55.000] Obviously, if you're working in production, you don't want to be hardcoding magic numbers.

[01:07:59.000] But instead of 6, this should now be 30.

[01:08:04.000] Let's run for 50,000 iterations, and let me split out the initialization here outside,

[01:08:10.000] so that when we run this cell multiple times, it's not going to wipe out our loss.

[01:08:17.000] In addition to that, instead of logging loss.item, let's do log10.

[01:08:28.000] I believe that's a function of the loss, and I'll show you why in a second.

[01:08:34.000] Let's optimize this.

[01:08:37.000] Basically, I'd like to plot the log loss instead of the loss, because when you plot the loss,

[01:08:41.000] many times it can have this hockey stick appearance, and log squashes it in, so it just looks nicer.

[01:08:49.000] The x-axis is step i, and the y-axis will be the loss i.

[01:09:01.000] And then here this is 30.

[01:09:03.000] Ideally, we wouldn't be hardcoding these, because let's look at the loss.

[01:09:12.000] It's again very thick, because the minibatch size is very small.

[01:09:15.000] But the total loss over the training set is 2.3, and the def set is 2.38 as well.

[01:09:22.000] So far so good.

[01:09:23.000] Let's try to now decrease the learning rate by a factor of 10, and train for another 50,000 iterations.

[01:09:35.000] We'd hope that we would be able to beat 2.32.

[01:09:43.000] But again, we're just doing this very haphazardly, so I don't actually have confidence that

[01:09:48.000] our learning rate is set very well, that our learning rate decay, which we just do at random, is set very well.

[01:09:56.000] The optimization here is kind of suspect, to be honest, and this is not how you would do it typically in production.

[01:10:01.000] In production, you would create parameters or hyperparameters out of all these settings,

[01:10:05.000] and then you would run lots of experiments and see whichever ones are working well for you.

[01:10:13.000] We have 2.17 now, and 2.2.

[01:10:16.000] So you see how the training and the validation performance are starting to slightly slowly depart.

[01:10:23.000] Maybe we're getting the sense that the neural net is getting good enough, or that the number of parameters is large enough,

[01:10:31.000] that we are slowly starting to overfit.

[01:10:34.000] Let's maybe run one more iteration of this and see where we get.

[01:10:41.000] Basically, you would be running lots of experiments, and then you are slowly scrutinizing whichever ones give you the best dev performance.

[01:10:48.000] Then once you find all the hyperparameters that make your dev performance good,

[01:10:53.000] you take that model and you evaluate the test set performance a single time.

[01:10:57.000] That's the number that you report in your paper or wherever else you want to talk about and brag about your model.

[01:11:05.000] Let's then rerun the plot and rerun the train and dev.

[01:11:11.000] Because we're getting lower loss now, it is the case that the embedding size of these was holding us back very likely.

[01:11:20.000] 2.16 and 2.19 is what we're roughly getting.

[01:11:24.000] There are many ways to go from here.

[01:11:27.000] We can continue tuning the optimization.

[01:11:30.000] We can continue, for example, playing with the size of the neural net.

[01:11:33.000] Or we can increase the number of words or characters in our case that we are taking as an input.

[01:11:39.000] Instead of just three characters, we could be taking more characters than as an input.

[01:11:43.000] That could further improve the loss.

[01:11:46.000] I changed the code slightly.

[01:11:48.000] We have here 200,000 steps of the optimization.

[01:11:51.000] In the first 100,000, we're using a learning rate of 0.1.

[01:11:54.000] Then in the next 100,000, we're using a learning rate of 0.01.

[01:11:58.000] This is the loss that I achieve.

[01:12:00.000] These are the performance on the training and validation loss.

[01:12:03.000] In particular, the best validation loss I've been able to obtain in the last 30 minutes or so is 2.17.

[01:12:10.000] Now I invite you to beat this number.

[01:12:12.000] You have quite a few knobs available to you to, I think, surpass this number.

[01:12:16.000] Number one, you can of course change the number of neurons in the hidden layer of this model.

[01:12:21.000] You can change the dimensionality of the embedding lookup table.

[01:12:25.000] You can change the number of characters that are feeding in as an input, as the context, into this model.

[01:12:32.000] And then, of course, you can change the details of the optimization.

[01:12:35.000] How long are we running?

[01:12:36.000] What is the learning rate?

[01:12:37.000] How does it change over time?

[01:12:39.000] How does it decay?

[01:12:41.000] You can change the batch size, and you may be able to actually achieve a much better convergence speed

[01:12:46.000] in terms of how many seconds or minutes it takes to train the model and get your result in terms of really good loss.

[01:12:55.000] And then, of course, I actually invite you to read this paper.

[01:12:58.000] It is 19 pages, but at this point, you should actually be able to read a good chunk of this paper

[01:13:03.000] and understand pretty good chunks of it.

[01:13:06.000] And this paper also has quite a few ideas for improvements that you can play with.

[01:13:11.000] So all those are knobs available to you, and you should be able to beat this number.

[01:13:15.000] I'm leaving that as an exercise to the reader.

[01:13:17.000] And that's it for now, and I'll see you next time.

[01:13:24.000] Before we wrap up, I also wanted to show how you would sample from the model.

[01:13:28.000] So we're going to generate 20 samples.

[01:13:31.000] At first, we begin with all dots, so that's the context.

[01:13:35.000] And then until we generate the 0th character again,

[01:13:40.000] we're going to embed the current context using the embedding table C.

[01:13:46.000] Now, usually here, the first dimension was the size of the training set,

[01:13:50.000] but here we're only working with a single example that we're generating,

[01:13:53.000] so this is just dimension 1, just for simplicity.

[01:13:58.000] And so this embedding then gets projected into the hidden state.

[01:14:02.000] You get the logits.

[01:14:03.000] Now we calculate the probabilities.

[01:14:05.000] For that, you can use f.softmax of logits,

[01:14:10.000] and that just basically exponentiates the logits and makes them sum to 1.

[01:14:13.000] And similar to cross-entropy, it is careful that there's no overflows.

[01:14:18.000] Once we have the probabilities, we sample from them using torch.multinomial

[01:14:22.000] to get our next index, and then we shift the context window to append the index and record it.

[01:14:28.000] And then we can just decode all the integers to strings and print them out.

[01:14:33.000] And so these are some example samples, and you can see that the model now works much better.

[01:14:38.000] So the words here are much more word-like or name-like.

[01:14:41.000] So we have things like ham, joe's, lilla.

[01:14:48.000] It's starting to sound a little bit more name-like.

[01:14:51.000] So we're definitely making progress, but we can still improve on this model quite a lot.

[01:14:55.000] Okay, sorry, there's some bonus content.

[01:14:57.000] I wanted to mention that I want to make these notebooks more accessible,

[01:15:01.000] and so I don't want you to have to install Jupyter Notebooks and Torch and everything else.

[01:15:05.000] So I will be sharing a link to a Google Colab,

[01:15:09.000] and the Google Colab will look like a notebook in your browser.

[01:15:13.000] And you can just go to a URL, and you'll be able to execute all of the code that you saw in the Google Colab.

[01:15:19.000] And so this is me executing the code in this lecture, and I shortened it a little bit.

[01:15:24.000] But basically, you're able to train the exact same network and then plot and sample from the model,

[01:15:29.000] and everything is ready for you to tinker with the numbers right there in your browser, no installation necessary.

[01:15:35.000] So I just wanted to point that out, and the link to this will be in the video description.

