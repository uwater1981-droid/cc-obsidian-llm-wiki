---
title: "Building makemore Part 3: Activations & Gradients, BatchNorm"
video_id: P6sfmUTpUmc
url: "https://www.youtube.com/watch?v=P6sfmUTpUmc"
author: Andrej Karpathy
slug: zero-to-hero-04-makemore-batchnorm
fetched_at: "2026-04-17T20:58:34+08:00"
type: youtube-transcript-whisper
transcript_source: "https://github.com/averkij/karcaps (004-large.html)"
segments: 2713
---

# Building makemore Part 3: Activations & Gradients, BatchNorm

> Video: https://www.youtube.com/watch?v=P6sfmUTpUmc
> Transcript: averkij/karcaps (Whisper-large) `004-large.html`

[00:00:00.000] Hi everyone.

[00:00:01.180] Today we are continuing our implementation of Make More.

[00:00:04.200] Now in the last lecture,

[00:00:05.040] we implemented the multilayer perceptron

[00:00:06.760] along the lines of Benjou et al. 2003

[00:00:08.880] for character level language modeling.

[00:00:10.720] So we followed this paper,

[00:00:12.080] took in a few characters in the past,

[00:00:14.080] and used an MLP to predict the next character in a sequence.

[00:00:17.300] So what we'd like to do now

[00:00:18.280] is we'd like to move on to more complex

[00:00:19.960] and larger neural networks, like recurrent neural networks

[00:00:22.680] and their variations like the GRU, LSTM, and so on.

[00:00:26.420] Now, before we do that though,

[00:00:27.760] we have to stick around the level of multilayer perceptron

[00:00:30.320] for a bit longer.

[00:00:31.520] And I'd like to do this

[00:00:32.460] because I would like us to have

[00:00:33.520] a very good intuitive understanding

[00:00:35.440] of the activations in the neural net during training,

[00:00:38.400] and especially the gradients that are flowing backwards

[00:00:40.920] and how they behave and what they look like.

[00:00:43.520] This is going to be very important

[00:00:44.800] to understand the history of the development

[00:00:46.520] of these architectures,

[00:00:48.120] because we'll see that recurrent neural networks,

[00:00:49.880] while they are very expressive

[00:00:52.000] in that they are a universal approximator

[00:00:53.780] and can in principle implement all the algorithms,

[00:00:57.920] we'll see that they are not very easily optimizable

[00:01:00.320] with the first-order gradient-based techniques

[00:01:02.080] that we have available to us and that we use all the time.

[00:01:04.680] And the key to understanding

[00:01:06.300] why they are not optimizable easily

[00:01:08.420] is to understand the activations and the gradients

[00:01:11.100] and how they behave during training.

[00:01:12.600] And we'll see that a lot of the variants

[00:01:14.120] since recurrent neural networks

[00:01:16.200] have tried to improve that situation.

[00:01:19.120] And so that's the path that we have to take,

[00:01:21.580] and let's get started.

[00:01:22.900] So the starting code for this lecture

[00:01:24.320] is largely the code from before,

[00:01:26.400] but I've cleaned it up a little bit.

[00:01:28.320] So you'll see that we are importing

[00:01:30.680] all the torch and matplotlib utilities.

[00:01:33.560] We're reading in the words just like before.

[00:01:35.600] These are eight example words.

[00:01:37.240] There's a total of 32,000 of them.

[00:01:39.280] Here's a vocabulary of all the lowercase letters

[00:01:41.680] and the special dot token.

[00:01:44.360] Here we are reading the dataset and processing it

[00:01:47.780] and creating three splits,

[00:01:50.600] the train dev and the test split.

[00:01:53.720] Now in the MLP, this is the identical same MLP,

[00:01:56.400] except you see that I removed

[00:01:57.760] a bunch of magic numbers that we had here.

[00:01:59.760] And instead we have the dimensionality

[00:02:01.600] of the embedding space of the characters

[00:02:03.560] and the number of hidden units in the hidden layer.

[00:02:06.160] And so I've pulled them outside here

[00:02:07.920] so that we don't have to go and change

[00:02:09.660] all these magic numbers all the time.

[00:02:11.680] With the same neural net with 11,000 parameters

[00:02:14.220] that we optimize now over 200,000 steps

[00:02:16.620] with a batch size of 32.

[00:02:18.320] And you'll see that I refactored the code here a little bit,

[00:02:22.200] but there are no functional changes.

[00:02:23.640] I just created a few extra variables, a few more comments,

[00:02:27.080] and I removed all the magic numbers.

[00:02:29.280] And otherwise it's the exact same thing.

[00:02:32.000] Then when we optimize,

[00:02:32.880] we saw that our loss looked something like this.

[00:02:35.980] We saw that the train and val loss were about 2.16 and so on.

[00:02:41.720] Here I refactored the code a little bit

[00:02:44.280] for the evaluation of arbitrary splits.

[00:02:47.080] So you pass in a string of which split

[00:02:48.760] you'd like to evaluate.

[00:02:50.080] And then here, depending on train, val, or test,

[00:02:53.160] I index in and I get the correct split.

[00:02:55.560] And then this is the forward pass of the network

[00:02:57.240] and evaluation of the loss and printing it.

[00:03:00.020] So just making it nicer.

[00:03:02.720] One thing that you'll notice here is

[00:03:05.360] I'm using a decorator torch.nograd,

[00:03:07.600] which you can also look up and read documentation of.

[00:03:11.320] Basically what this decorator does on top of a function

[00:03:14.360] is that whatever happens in this function

[00:03:17.600] is assumed by a torch to never require any gradients.

[00:03:21.960] So it will not do any of the bookkeeping

[00:03:24.200] that it does to keep track of all the gradients

[00:03:26.720] in anticipation of an eventual backward pass.

[00:03:29.640] It's almost as if all the tensors that get created here

[00:03:31.920] have a requires grad of false.

[00:03:34.480] And so it just makes everything much more efficient

[00:03:36.400] because you're telling torch that I will not call

[00:03:38.560] dot backward on any of this computation,

[00:03:40.640] and you don't need to maintain the graph under the hood.

[00:03:43.720] So that's what this does.

[00:03:45.560] And you can also use a context manager with torch.nograd,

[00:03:50.120] and you can look those up.

[00:03:53.040] Then here we have the sampling from a model,

[00:03:55.760] just as before, just a forward pass of a neural net,

[00:03:58.320] getting the distribution, sampling from it,

[00:04:00.660] adjusting the context window,

[00:04:02.140] and repeating until we get the special end token.

[00:04:04.880] And we see that we are starting to get

[00:04:06.840] much nicer looking words sampled from the model.

[00:04:09.840] It's still not amazing,

[00:04:11.080] and they're still not fully name-like,

[00:04:13.360] but it's much better than what we had

[00:04:14.720] to do with the bigram model.

[00:04:17.760] So that's our starting point.

[00:04:19.200] Now, the first thing I would like to scrutinize

[00:04:20.720] is the initialization.

[00:04:22.680] I can tell that our network

[00:04:24.520] is very improperly configured at initialization,

[00:04:27.640] and there's multiple things wrong with it,

[00:04:29.120] but let's just start with the first one.

[00:04:31.200] Look here on the zeroth iteration,

[00:04:32.840] the very first iteration,

[00:04:34.880] we are recording a loss of 27,

[00:04:37.120] and this rapidly comes down to roughly one or two or so.

[00:04:40.340] So I can tell that the initialization is all messed up

[00:04:42.240] because this is way too high.

[00:04:44.440] In training of neural nets,

[00:04:45.800] it is almost always the case

[00:04:46.920] that you will have a rough idea

[00:04:48.000] for what loss to expect at initialization,

[00:04:50.880] and that just depends on the loss function

[00:04:52.860] and the problem setup.

[00:04:54.800] In this case, I do not expect 27.

[00:04:57.120] I expect a much lower number,

[00:04:58.280] and we can calculate it together.

[00:05:00.600] Basically, at initialization,

[00:05:02.360] what we'd like is that there's 27 characters

[00:05:06.120] that could come next for any one training example.

[00:05:09.080] At initialization, we have no reason to believe

[00:05:11.180] any characters to be much more likely than others,

[00:05:13.760] and so we'd expect that the probability distribution

[00:05:15.840] that comes out initially is a uniform distribution

[00:05:19.160] assigning about equal probability to all the 27 characters.

[00:05:23.420] So basically what we'd like is the probability

[00:05:25.720] for any character would be roughly one over 27.

[00:05:31.980] That is the probability we should record,

[00:05:33.880] and then the loss is the negative log probability.

[00:05:36.640] So let's wrap this in a tensor,

[00:05:38.280] and then then we can take the log of it,

[00:05:42.100] and then the negative log probability

[00:05:44.100] is the loss we would expect,

[00:05:45.980] which is 3.29, much, much lower than 27.

[00:05:49.980] And so what's happening right now

[00:05:51.420] is that at initialization,

[00:05:52.920] the neural net is creating probability distributions

[00:05:55.100] that are all messed up.

[00:05:56.320] Some characters are very confident,

[00:05:58.140] and some characters are very not confident.

[00:06:00.700] And then basically what's happening

[00:06:01.940] is that the network is very confidently wrong,

[00:06:05.300] and that's what makes it record very high loss.

[00:06:10.620] So here's a smaller four-dimensional example of the issue.

[00:06:13.420] Let's say we only have four characters,

[00:06:15.980] and then we have logics that come out of the neural net,

[00:06:18.580] and they are very, very close to zero.

[00:06:20.920] Then when we take the softmax of all zeros,

[00:06:23.820] we get probabilities that are a diffused distribution.

[00:06:27.420] So sums to one and is exactly uniform.

[00:06:31.100] And then in this case, if the label is say two,

[00:06:33.780] it doesn't actually matter if the label is two,

[00:06:36.620] or three, or one, or zero,

[00:06:38.300] because it's a uniform distribution.

[00:06:39.940] We're recording the exact same loss, in this case, 1.38.

[00:06:43.160] So this is the loss we would expect

[00:06:44.500] for a four-dimensional example.

[00:06:46.260] And I can see, of course,

[00:06:47.180] that as we start to manipulate these logics,

[00:06:50.560] we're going to be changing the loss here.

[00:06:52.460] So it could be that we lock out,

[00:06:54.160] and by chance, this could be a very high number,

[00:06:57.120] like five or something like that.

[00:06:59.300] Then in that case, we'll record a very low loss

[00:07:01.060] because we're assigning the correct probability

[00:07:02.820] at initialization by chance to the correct label.

[00:07:06.700] Much more likely it is that some other dimension

[00:07:10.380] will have a high logit.

[00:07:14.020] And then what will happen

[00:07:14.860] is we start to record much higher loss.

[00:07:17.140] And what can happen is basically the logits come out

[00:07:20.260] like something like this,

[00:07:22.180] and they take on extreme values,

[00:07:24.380] and we record really high loss.

[00:07:28.580] For example, if we have tors.random of four,

[00:07:31.680] so these are normally distributed numbers, four of them.

[00:07:40.460] Then here, we can also print the logits,

[00:07:43.780] probabilities that come out of it, and the loss.

[00:07:47.060] And so because these logits are near zero,

[00:07:50.340] for the most part, the loss that comes out is okay.

[00:07:53.940] But suppose this is like times 10 now.

[00:07:56.260] You see how, because these are more extreme values,

[00:07:59.260] it's very unlikely that you're going to be guessing

[00:08:01.660] the correct bucket, and then you're confidently wrong

[00:08:05.060] and recording very high loss.

[00:08:07.180] If your logits are coming up even more extreme,

[00:08:10.020] you might get extremely insane losses,

[00:08:12.900] like infinity even at initialization.

[00:08:17.940] So basically, this is not good,

[00:08:19.260] and we want the logits to be roughly zero

[00:08:21.500] when the network is initialized.

[00:08:24.860] In fact, the logits don't have to be just zero,

[00:08:26.860] they just have to be equal.

[00:08:28.140] So for example, if all the logits are one,

[00:08:30.980] then because of the normalization inside the softmax,

[00:08:33.540] this will actually come out okay.

[00:08:35.300] But by symmetry, we don't want it to be

[00:08:36.740] any arbitrary positive or negative number,

[00:08:38.860] we just want it to be all zeros

[00:08:40.300] and record the loss that we expect at initialization.

[00:08:43.140] So let's now concretely see

[00:08:44.260] where things go wrong in our example.

[00:08:46.540] Here we have the initialization.

[00:08:48.340] Let me reinitialize the neural net, and here, let me break

[00:08:51.660] after the very first iteration,

[00:08:52.940] so we only see the initial loss, which is 27.

[00:08:56.940] So that's way too high, and intuitively,

[00:08:58.820] now we can expect the variables involved,

[00:09:01.060] and we see that the logits here,

[00:09:02.700] if we just print some of these,

[00:09:06.460] if we just print the first row,

[00:09:07.740] we see that the logits take on quite extreme values,

[00:09:10.660] and that's what's creating the fake confidence

[00:09:13.220] in incorrect answers and making it hard for us

[00:09:16.140] to get the correct answer, and that makes the loss

[00:09:20.660] get very, very high.

[00:09:22.100] So these logits should be much, much closer to zero.

[00:09:25.380] So now let's think through how we can achieve logits

[00:09:28.020] coming out of this neural net to be more closer to zero.

[00:09:32.580] You see here that logits are calculated

[00:09:34.180] as the hidden states multiplied by w2 plus b2.

[00:09:37.700] So first of all, currently we're initializing b2

[00:09:40.500] as random values of the right size.

[00:09:44.300] But because we want roughly zero,

[00:09:46.700] we don't actually want to be adding a bias

[00:09:48.340] of random numbers.

[00:09:49.340] So in fact, I'm going to add a times a zero here

[00:09:51.980] to make sure that b2 is just basically zero

[00:09:55.740] at initialization.

[00:09:57.540] And second, this is h multiplied by w2.

[00:10:00.420] So if we want logits to be very, very small,

[00:10:03.060] then we would be multiplying w2 and making that smaller.

[00:10:07.020] So for example, if we scale down w2 by 0.1,

[00:10:10.020] all the elements, then if I do again

[00:10:13.140] just the very first iteration,

[00:10:14.500] you see that we are getting much closer to what we expect.

[00:10:17.420] So roughly what we want is about 3.29.

[00:10:20.580] This is 4.2.

[00:10:22.380] I can make this maybe even smaller, 3.32.

[00:10:26.460] Okay, so we're getting closer and closer.

[00:10:28.700] Now you're probably wondering, can we just set this to zero?

[00:10:33.220] Then we get, of course, exactly what we're looking for

[00:10:36.540] at initialization.

[00:10:38.140] And the reason I don't usually do this

[00:10:40.300] is because I'm very nervous.

[00:10:42.420] And I'll show you in a second

[00:10:43.580] why you don't wanna be setting w's

[00:10:46.140] or weights of a neural net exactly to zero.

[00:10:49.180] You usually want it to be small numbers

[00:10:51.180] instead of exactly zero.

[00:10:53.380] For this output layer in this specific case,

[00:10:55.660] I think it would be fine,

[00:10:57.380] but I'll show you in a second

[00:10:58.340] where things go wrong very quickly if you do that.

[00:11:00.780] So let's just go with 0.01.

[00:11:03.020] In that case, our loss is close enough,

[00:11:05.220] but has some entropy.

[00:11:06.660] It's not exactly zero.

[00:11:08.380] It's got some little entropy

[00:11:10.380] and that's used for symmetry breaking, as we'll see in a second.

[00:11:13.660] Logits are now coming out much closer to zero

[00:11:16.220] and everything is well and good.

[00:11:18.220] So if I just erase these

[00:11:21.220] and I now take away the break statement,

[00:11:25.020] we can run the optimization with this new initialization

[00:11:28.420] and let's just see what losses we record.

[00:11:32.620] Okay, so I'll let it run.

[00:11:33.980] And you see that we started off good

[00:11:35.780] and then we came down a bit.

[00:11:37.180] The plot of the loss now doesn't have

[00:11:40.460] this hockey shape appearance

[00:11:43.540] because basically what's happening in the hockey stick,

[00:11:45.740] the very first few iterations of the loss,

[00:11:48.060] what's happening during the optimization

[00:11:50.020] is the optimization is just squashing down the logits

[00:11:52.940] and then it's rearranging the logits.

[00:11:55.060] So basically we took away this easy part

[00:11:57.500] of the loss function where just the weights

[00:11:59.940] were just being shrunk down.

[00:12:01.820] And so therefore we don't get these easy gains

[00:12:04.940] in the beginning

[00:12:05.780] and we're just getting some of the hard gains

[00:12:07.420] of training the actual neural net.

[00:12:08.900] And so there's no hockey stick appearance.

[00:12:11.460] So good things are happening in that both,

[00:12:13.860] number one, loss at initialization is what we expect

[00:12:17.060] and the loss doesn't look like a hockey stick.

[00:12:20.700] And this is true for any neural net you might train

[00:12:23.380] and something to look out for.

[00:12:25.620] And second, the loss that came out

[00:12:27.660] is actually quite a bit improved.

[00:12:29.580] Unfortunately, I erased what we had here before.

[00:12:31.940] I believe this was 2.12

[00:12:35.020] and this was 2.16.

[00:12:37.340] So we get a slightly improved result.

[00:12:40.180] And the reason for that is

[00:12:41.620] because we're spending more cycles, more time,

[00:12:44.420] optimizing the neural net actually,

[00:12:46.540] instead of just spending the first several thousand

[00:12:49.660] iterations probably just squashing down the weights

[00:12:53.260] because they are so way too high

[00:12:54.820] in the beginning of the initialization.

[00:12:56.900] So something to look out for and that's number one.

[00:13:00.100] Now let's look at the second problem.

[00:13:01.820] Let me reinitialize our neural net

[00:13:03.540] and let me reintroduce the break statement.

[00:13:06.060] So we have a reasonable initial loss.

[00:13:08.620] So even though everything is looking good

[00:13:09.980] on the level of the loss

[00:13:11.020] and we get something that we expect,

[00:13:12.700] there's still a deeper problem lurking

[00:13:14.620] inside this neural net and its initialization.

[00:13:17.460] So the logits are now okay.

[00:13:19.940] The problem now is with the values of H,

[00:13:23.060] the activations of the hidden states.

[00:13:25.380] Now, if we just visualize this vector,

[00:13:27.620] sorry, this tensor H, it's kind of hard to see

[00:13:29.980] but the problem here, roughly speaking,

[00:13:31.780] is you see how many of the elements are one or negative one.

[00:13:36.060] Now recall that torch.tenh,

[00:13:38.100] the tenh function is a squashing function.

[00:13:40.580] It takes arbitrary numbers and it squashes them

[00:13:42.740] into a range of negative one and one

[00:13:44.420] and it does so smoothly.

[00:13:46.220] So let's look at the histogram of H

[00:13:47.980] to get a better idea of the distribution

[00:13:50.020] of the values inside this tensor.

[00:13:52.420] We can do this first.

[00:13:55.100] Well, we can see that H is 32 examples

[00:13:58.100] and 200 activations in each example.

[00:14:00.900] We can view it as negative one,

[00:14:02.580] stretch it out into one large vector

[00:14:06.420] and we can then call toList to convert this

[00:14:09.620] into one large Python list of floats.

[00:14:13.740] And then we can pass this into plt.hist for histogram

[00:14:17.700] and we say we want 50 bins

[00:14:20.100] and a semicolon to suppress a bunch of output we don't want.

[00:14:24.380] So we see this histogram and we see that most of the values

[00:14:26.740] by far take on value of negative one and one.

[00:14:30.100] So this tenh is very, very active.

[00:14:33.220] And we can also look at basically why that is.

[00:14:37.900] We can look at the pre-activations that feed into the tenh

[00:14:42.780] and we can see that the distribution of the pre-activations

[00:14:46.220] is very, very broad.

[00:14:47.380] These take numbers between negative 15 and 15

[00:14:50.100] and that's why in a torch.tenh,

[00:14:51.980] everything is being squashed and capped

[00:14:53.820] to be in the range of negative one and one

[00:14:55.740] and lots of numbers here take on very extreme values.

[00:14:59.140] Now, if you are new to neural networks,

[00:15:01.100] you might not actually see this as an issue,

[00:15:03.380] but if you're well versed in the dark arts

[00:15:05.380] of backpropagation and have an intuitive sense

[00:15:07.900] of how these gradients flow through a neural net,

[00:15:10.300] you are looking at your distribution

[00:15:11.740] of tenh activations here and you are sweating.

[00:15:14.940] So let me show you why.

[00:15:16.380] We have to keep in mind that during backpropagation,

[00:15:18.340] just like we saw in micrograd,

[00:15:19.940] we are doing backward pass starting at the loss

[00:15:22.140] and flowing through the network backwards.

[00:15:24.740] In particular, we're going to backpropagate

[00:15:26.260] through this torch.tenh.

[00:15:28.700] And this layer here is made up of 200 neurons

[00:15:31.740] for each one of these examples.

[00:15:33.700] And it implements an elementwise tenh.

[00:15:36.620] So let's look at what happens in tenh in the backward pass.

[00:15:39.820] We can actually go back to our previous micrograd code

[00:15:42.980] in the very first lecture

[00:15:44.420] and see how we implemented tenh.

[00:15:46.900] We saw that the input here was x

[00:15:49.260] and then we calculate t, which is the tenh of x.

[00:15:52.380] So that's t and t is between negative one and one.

[00:15:54.860] It's the output of the tenh.

[00:15:56.460] And then in the backward pass,

[00:15:57.500] how do we backpropagate through a tenh?

[00:16:00.100] We take out.grad and then we multiply it.

[00:16:03.980] This is the chain rule with the local gradient,

[00:16:06.220] which took the form of one minus t squared.

[00:16:09.060] So what happens if the outputs of your tenh

[00:16:11.420] are very close to negative one or one?

[00:16:14.100] If you plug in t equals one here,

[00:16:16.060] you're going to get a zero multiplying out.grad.

[00:16:19.740] No matter what out.grad is,

[00:16:21.220] we are killing the gradient

[00:16:22.940] and we're stopping effectively the backpropagation

[00:16:25.620] through this tenh unit.

[00:16:27.420] Similarly, when t is negative one,

[00:16:29.180] this will again become zero

[00:16:30.580] and out.grad just stops.

[00:16:32.900] And intuitively this makes sense

[00:16:34.380] because this is a tenh neuron.

[00:16:37.580] And what's happening is if its output is very close to one,

[00:16:41.300] then we are in the tail of this tenh.

[00:16:43.940] And so changing basically the input

[00:16:49.140] is not going to impact the output of the tenh too much

[00:16:52.100] because it's in a flat region of the tenh.

[00:16:55.660] And so therefore there's no impact on the loss.

[00:16:58.460] And so indeed the weights and the biases

[00:17:02.420] along with this tenh neuron do not impact the loss

[00:17:05.460] because the output of this tenh unit

[00:17:07.100] is in a flat region of the tenh

[00:17:08.700] and there's no influence.

[00:17:09.620] We can be changing them however we want

[00:17:13.020] and the loss is not impacted.

[00:17:14.540] That's another way to justify that indeed

[00:17:17.140] the gradient would be basically zero, it vanishes.

[00:17:20.900] Indeed, when t equals zero,

[00:17:24.380] we get one times out.grad.

[00:17:27.300] So when the tenh takes on exactly value of zero,

[00:17:31.180] then out.grad is just passed through.

[00:17:34.900] So basically what this is doing, right,

[00:17:36.380] is if t is equal to zero,

[00:17:38.220] then the tenh unit is sort of inactive

[00:17:42.340] and gradient just passes through.

[00:17:44.780] But the more you are in the flat tails,

[00:17:47.220] the more the gradient is squashed.

[00:17:49.460] So in fact, you'll see that the gradient

[00:17:51.780] flowing through tenh can only ever decrease

[00:17:54.540] and the amount that it decreases

[00:17:56.580] is proportional through a square here

[00:18:01.380] depending on how far you are in the flat tails

[00:18:03.220] of this tenh.

[00:18:05.020] And so that's kind of what's happening here.

[00:18:07.020] And the concern here is that if all of these outputs h

[00:18:12.820] are in the flat regions of negative one and one,

[00:18:14.980] then the gradients that are flowing through the network

[00:18:17.300] will just get destroyed at this layer.

[00:18:19.780] Now, there is some redeeming quality here

[00:18:22.860] and that we can actually get a sense of the problem here

[00:18:24.860] as follows.

[00:18:26.180] I wrote some code here.

[00:18:28.060] And basically what we want to do here

[00:18:29.540] is we want to take a look at h,

[00:18:31.740] take the absolute value and see how often it is

[00:18:35.180] in the flat region.

[00:18:37.300] So say greater than 0.99.

[00:18:41.300] And what you get is the following.

[00:18:43.220] And this is a Boolean tensor.

[00:18:44.580] So in the Boolean tensor, you get a white

[00:18:48.180] if this is true and a black if this is false.

[00:18:51.500] And so basically what we have here is the 32 examples

[00:18:54.060] and the 200 hidden neurons.

[00:18:56.420] And we see that a lot of this is white.

[00:18:59.420] And what that's telling us is that all these tenh neurons

[00:19:03.060] were very, very active and they're in the flat tail.

[00:19:08.060] And so in all these cases,

[00:19:11.100] the backward gradient would get destroyed.

[00:19:14.100] Now, we would be in a lot of trouble if,

[00:19:17.100] for any one of these 200 neurons,

[00:19:19.980] if it was the case that the entire column is white.

[00:19:23.620] Because in that case, we have what's called the dead neuron.

[00:19:26.140] And this could be a tenh neuron where the initialization

[00:19:28.340] of the weights and the biases could be such that

[00:19:30.420] no single example ever activates this tenh

[00:19:34.460] in the sort of active part of the tenh.

[00:19:37.380] If all the examples land in the tail,

[00:19:40.460] then this neuron will never be able to activate

[00:19:42.940] and this neuron will never learn.

[00:19:44.700] It is a dead neuron.

[00:19:46.660] And so just scrutinizing this and looking for columns

[00:19:50.300] of completely white, we see that this is not the case.

[00:19:54.100] So I don't see a single neuron that is all of white.

[00:19:59.380] And so therefore it is the case that for every one

[00:20:01.420] of these tenh neurons, we do have some examples

[00:20:05.380] that activate them in the active part of the tenh.

[00:20:08.940] And so some gradients will flow through

[00:20:10.540] and this neuron will learn.

[00:20:12.260] And the neuron will change and it will move

[00:20:14.260] and it will do something.

[00:20:16.340] But you can sometimes get yourself in cases

[00:20:18.420] where you have dead neurons.

[00:20:20.260] And the way this manifests is that for a tenh neuron,

[00:20:23.420] this would be when no matter what inputs you plug in

[00:20:26.300] from your data set, this tenh neuron always fires

[00:20:29.100] completely one or completely negative one.

[00:20:31.260] And then it will just not learn

[00:20:33.380] because all the gradients will be just zeroed out.

[00:20:36.660] This is true, not just for tenh,

[00:20:37.780] but for a lot of other nonlinearities

[00:20:39.620] that people use in neural networks.

[00:20:41.100] So we certainly use tenh a lot,

[00:20:43.140] but sigmoid will have the exact same issue

[00:20:45.020] because it is a squashing neuron.

[00:20:47.420] And so the same will be true for sigmoid,

[00:20:49.900] but basically the same will actually apply to sigmoid.

[00:20:57.020] The same will also apply to relu.

[00:20:59.060] So relu has a completely flat region here below zero.

[00:21:03.380] So if you have a relu neuron,

[00:21:04.900] then it is a pass-through if it is positive.

[00:21:08.580] And if the pre-activation is negative,

[00:21:11.020] it will just shut it off.

[00:21:12.620] Since the region here is completely flat,

[00:21:15.060] then during backpropagation,

[00:21:17.220] this would be exactly zeroing out the gradient.

[00:21:20.820] Like all of the gradient would be set exactly to zero

[00:21:22.940] instead of just like a very, very small number

[00:21:24.580] depending on how positive or negative T is.

[00:21:28.460] And so you can get, for example, a dead relu neuron.

[00:21:31.500] And a dead relu neuron would basically look like,

[00:21:35.260] basically what it is, is if a neuron

[00:21:37.500] with a relu nonlinearity never activates,

[00:21:41.100] so for any examples that you plug in in the dataset,

[00:21:43.980] it never turns on, it's always in this flat region,

[00:21:47.380] then this relu neuron is a dead neuron.

[00:21:49.420] Its weights and bias will never learn.

[00:21:52.060] They will never get a gradient

[00:21:53.220] because the neuron never activated.

[00:21:55.700] And this can sometimes happen at initialization

[00:21:57.980] because the weights and the biases just make it

[00:21:59.620] so that by chance, some neurons are just forever dead.

[00:22:02.820] But it can also happen during optimization.

[00:22:04.860] If you have like a too high of a learning rate, for example,

[00:22:07.540] sometimes you have these neurons

[00:22:08.820] that gets too much of a gradient

[00:22:10.380] and they get knocked out off the data manifold.

[00:22:13.660] And what happens is that from then on,

[00:22:15.580] no example ever activates this neuron.

[00:22:17.820] So this neuron remains dead forever.

[00:22:19.460] So it's kind of like a permanent brain damage

[00:22:21.060] in a mind of a network.

[00:22:23.820] And so sometimes what can happen is

[00:22:25.380] if your learning rate is very high, for example,

[00:22:27.340] and you have a neural net with relu neurons,

[00:22:29.620] you train the neural net and you get some last loss.

[00:22:32.620] But then actually what you do is

[00:22:34.420] you go through the entire training set

[00:22:36.300] and you forward your examples

[00:22:39.380] and you can find neurons that never activate.

[00:22:42.020] They are dead neurons in your network.

[00:22:43.980] And so those neurons will never turn on.

[00:22:46.380] And usually what happens is that during training,

[00:22:48.340] these relu neurons are changing, moving, et cetera.

[00:22:50.660] And then because of a high gradient somewhere,

[00:22:52.380] by chance, they get knocked off

[00:22:54.540] and then nothing ever activates them.

[00:22:56.340] And from then on, they are just dead.

[00:22:58.940] So that's kind of like a permanent brain damage

[00:23:00.540] that can happen to some of these neurons.

[00:23:03.060] These other nonlinearities like leaky relu

[00:23:05.380] will not suffer from this issue as much

[00:23:07.300] because you can see that it doesn't have flat tails.

[00:23:10.500] You'll almost always get gradients.

[00:23:12.860] And elu is also fairly frequently used.

[00:23:16.420] It also might suffer from this issue

[00:23:17.820] because it has flat parts.

[00:23:20.220] So that's just something to be aware of

[00:23:22.500] and something to be concerned about.

[00:23:24.060] And in this case, we have way too many activations H

[00:23:28.620] that take on extreme values.

[00:23:30.420] And because there's no column of white, I think we will be okay.

[00:23:34.260] And indeed the network optimizes

[00:23:35.620] and gives us a pretty decent loss,

[00:23:37.540] but it's just not optimal.

[00:23:38.820] And this is not something you want,

[00:23:40.380] especially during initialization.

[00:23:42.220] And so basically what's happening is that

[00:23:45.140] this H pre-activation that's flowing to 10H,

[00:23:48.540] it's too extreme, it's too large.

[00:23:50.940] It's creating a distribution that is too saturated

[00:23:55.740] in both sides of the 10H.

[00:23:57.180] And it's not something you want

[00:23:58.300] because it means that there's less training

[00:24:01.180] for these neurons because they update less frequently.

[00:24:05.660] So how do we fix this?

[00:24:07.140] Well, H pre-activation is MCAT, which comes from C.

[00:24:12.620] So these are uniform Gaussian,

[00:24:14.900] but then it's multiplied by W1 plus B1.

[00:24:17.420] And H pre-act is too far off from zero

[00:24:20.100] and that's causing the issue.

[00:24:21.420] So we want this pre-activation to be closer to zero,

[00:24:24.620] very similar to what we had with logits.

[00:24:27.220] So here we want actually something very, very similar.

[00:24:31.340] Now it's okay to set the biases to very small number.

[00:24:34.940] We can either multiply by 001

[00:24:36.700] to get like a little bit of entropy.

[00:24:39.460] I sometimes like to do that

[00:24:41.540] just so that there's like a little bit of variation

[00:24:45.020] and diversity in the original initialization

[00:24:48.020] of these 10H neurons.

[00:24:49.380] And I find in practice that that can help optimization

[00:24:52.100] a little bit.

[00:24:53.660] And then the weights, we can also just like squash.

[00:24:56.140] So let's multiply everything by 0.1.

[00:24:59.140] Let's rerun the first batch.

[00:25:01.460] And now let's look at this.

[00:25:03.060] And well, first let's look at here.

[00:25:06.980] You see now, because we multiplied W by 0.1,

[00:25:09.460] we have a much better histogram.

[00:25:11.100] And that's because the pre-activations

[00:25:12.500] are now between negative 1.5 and 1.5.

[00:25:14.900] And this we expect much, much less white.

[00:25:18.500] Okay, there's no white.

[00:25:20.740] So basically that's because there are no neurons

[00:25:23.660] that's saturated above 0.99 in either direction.

[00:25:27.820] So it's actually a pretty decent place to be.

[00:25:31.740] Maybe we can go up a little bit.

[00:25:36.620] Sorry, am I changing W1 here?

[00:25:39.140] So maybe we can go to 0.2.

[00:25:42.100] Okay, so maybe something like this is a nice distribution.

[00:25:46.340] So maybe this is what our initialization should be.

[00:25:49.060] So let me now erase these.

[00:25:52.380] And let me, starting with initialization,

[00:25:55.660] let me run the full optimization without the break.

[00:25:59.380] And let's see what we got.

[00:26:02.140] Okay, so the optimization finished and I rerun the loss.

[00:26:05.180] And this is the result that we get.

[00:26:07.060] And then just as a reminder,

[00:26:08.180] I put down all the losses that we saw previously

[00:26:10.180] in this lecture.

[00:26:11.500] So we see that we actually do get an improvement here.

[00:26:14.180] And just as a reminder,

[00:26:15.460] we started off with a validation loss of 2.17

[00:26:17.860] when we started.

[00:26:19.100] By fixing the softmax being confidently wrong,

[00:26:21.540] we came down to 2.13.

[00:26:23.140] And by fixing the 10H layer being way too saturated,

[00:26:25.660] we came down to 2.10.

[00:26:27.940] And the reason this is happening, of course,

[00:26:29.340] is because our initialization is better.

[00:26:30.940] And so we're spending more time doing productive training

[00:26:33.660] instead of not very productive training

[00:26:37.100] because our gradients are set to zero.

[00:26:39.140] And we have to learn very simple things

[00:26:41.500] like the overconfidence of the softmax in the beginning.

[00:26:44.460] And we're spending cycles

[00:26:45.300] just like squashing down the weight matrix.

[00:26:48.020] So this is illustrating basically initialization

[00:26:53.020] and its impacts on performance

[00:26:55.620] just by being aware of the internals of these neural nets

[00:26:58.420] and their activations and their gradients.

[00:27:00.420] Now, we're working with a very small network.

[00:27:02.780] This is just one layer multilayer perception.

[00:27:05.420] So because the network is so shallow,

[00:27:07.620] the optimization problem is actually quite easy

[00:27:09.900] and very forgiving.

[00:27:11.380] So even though our initialization was terrible,

[00:27:13.420] the network still learned eventually.

[00:27:15.340] It just got a bit worse result.

[00:27:17.300] This is not the case in general, though.

[00:27:19.300] Once we actually start working with much deeper networks

[00:27:22.660] that have, say, 50 layers,

[00:27:24.580] things can get much more complicated

[00:27:26.980] and these problems stack up.

[00:27:30.300] And so you can actually get into a place

[00:27:32.940] where the network is basically not training at all

[00:27:34.980] if your initialization is bad enough.

[00:27:37.180] And the deeper your network is and the more complex it is,

[00:27:39.820] the less forgiving it is to some of these errors.

[00:27:42.940] And so something to definitely be aware of

[00:27:46.380] and something to scrutinize, something to plot,

[00:27:49.420] and something to be careful with.

[00:27:50.980] And, yeah.

[00:27:53.580] Okay, so that's great that that worked for us.

[00:27:55.620] But what we have here now is all these magic numbers,

[00:27:58.180] like 0.2.

[00:27:59.020] Like, where do I come up with this?

[00:28:00.500] And how am I supposed to set these

[00:28:01.940] if I have a large neural net with lots and lots of layers?

[00:28:05.180] And so obviously no one does this by hand.

[00:28:07.460] There's actually some relatively principled ways

[00:28:09.460] of setting these scales

[00:28:11.820] that I would like to introduce to you now.

[00:28:13.980] So let me paste some code here that I prepared

[00:28:16.420] just to motivate the discussion of this.

[00:28:19.500] So what I'm doing here is we have some random input here, x,

[00:28:23.340] that is drawn from a Gaussian.

[00:28:25.220] And there's 1,000 examples that are 10-dimensional.

[00:28:28.580] And then we have a weighting layer here

[00:28:30.580] that is also initialized using Gaussian,

[00:28:33.020] just like we did here.

[00:28:34.660] And these neurons in the hidden layer look at 10 inputs

[00:28:38.420] and there are 200 neurons in this hidden layer.

[00:28:41.580] And then we have here, just like here,

[00:28:43.900] in this case, the multiplication, x multiplied by w,

[00:28:47.100] to get the pre-activations of these neurons.

[00:28:50.820] And basically the analysis here looks at,

[00:28:53.140] okay, suppose these are uniform Gaussian

[00:28:55.220] and these weights are uniform Gaussian.

[00:28:57.140] If I do x times w, and we forget for now the bias

[00:29:00.740] and the nonlinearity,

[00:29:03.220] then what is the mean and the standard deviation

[00:29:05.380] of these Gaussians?

[00:29:06.940] So in the beginning here,

[00:29:07.980] the input is just a normal Gaussian distribution.

[00:29:10.940] Mean is zero and the standard deviation is one.

[00:29:13.580] And the standard deviation, again,

[00:29:14.740] is just a measure of a spread of the Gaussian.

[00:29:18.540] But then once we multiply here

[00:29:19.820] and we look at the histogram of y,

[00:29:23.460] we see that the mean, of course, stays the same.

[00:29:25.700] It's about zero because this is a symmetric operation.

[00:29:28.860] But we see here that the standard deviation

[00:29:30.420] has expanded to three.

[00:29:32.540] So the input standard deviation was one,

[00:29:34.300] but now we've grown to three.

[00:29:36.380] And so what you're seeing in the histogram

[00:29:37.620] is that this Gaussian is expanding.

[00:29:39.780] And so we're expanding this Gaussian from the input.

[00:29:44.820] And we don't want that.

[00:29:45.660] We want most of the neural nets

[00:29:46.820] to have relatively similar activations.

[00:29:49.500] So unit Gaussian roughly throughout the neural net.

[00:29:52.900] And so the question is,

[00:29:53.740] how do we scale these w's to preserve this distribution

[00:29:59.620] to remain a Gaussian?

[00:30:02.780] And so intuitively, if I multiply here,

[00:30:05.540] these elements of w by a large number,

[00:30:08.460] let's say by five, then this Gaussian

[00:30:12.780] grows and grows in standard deviation.

[00:30:14.940] So now we're at 15.

[00:30:16.300] So basically these numbers here in the output y

[00:30:19.140] take on more and more extreme values.

[00:30:21.660] But if we scale it down, let's say 0.2,

[00:30:24.300] then conversely, this Gaussian is getting smaller and smaller

[00:30:28.700] and it's shrinking.

[00:30:30.140] And you can see that the standard deviation is 0.6.

[00:30:32.940] And so the question is, what do I multiply by here

[00:30:35.740] to exactly preserve the standard deviation to be one?

[00:30:40.140] And it turns out that the correct answer mathematically,

[00:30:42.020] when you work out through the variance

[00:30:43.900] of this multiplication here,

[00:30:46.460] is that you are supposed to divide

[00:30:48.740] by the square root of the fan in.

[00:30:51.980] The fan in is basically the number

[00:30:55.300] of input elements here, 10.

[00:30:57.180] So we are supposed to divide by 10 square root.

[00:30:59.940] And this is one way to do the square root.

[00:31:01.620] You raise it to a power of 0.5.

[00:31:03.460] That's the same as doing a square root.

[00:31:06.300] So when you divide by the square root of 10,

[00:31:09.700] then we see that the output Gaussian,

[00:31:13.300] it has exactly standard deviation of 1.

[00:31:16.660] Now, unsurprisingly, a number of papers

[00:31:18.620] have looked into how to best initialize neural networks.

[00:31:22.540] And in the case of multi-layer perceptrons,

[00:31:24.420] we can have fairly deep networks that

[00:31:26.220] have these nonlinearities in between.

[00:31:28.260] And we want to make sure that the activations are

[00:31:30.100] well-behaved and they don't expand to infinity

[00:31:32.500] or shrink all the way to 0.

[00:31:34.300] And the question is, how do we initialize the weights

[00:31:36.180] so that these activations take on reasonable values

[00:31:38.420] throughout the network?

[00:31:40.060] Now, one paper that has studied this in quite a bit of detail

[00:31:42.860] that is often referenced is this paper by Kamingha et al.

[00:31:46.060] called Delving Deep Interactifiers.

[00:31:48.420] Now, in this case, they actually study

[00:31:49.820] convolutional neural networks.

[00:31:51.500] And they study, especially, the ReLU nonlinearity

[00:31:55.180] and the P-ReLU nonlinearity instead of a 10H nonlinearity.

[00:31:58.940] But the analysis is very similar.

[00:32:00.660] And basically, what happens here is, for them,

[00:32:05.340] the ReLU nonlinearity that they care about quite a bit here

[00:32:08.340] is a squashing function where all the negative numbers

[00:32:12.340] are simply clamped to 0.

[00:32:14.900] So the positive numbers are a path through,

[00:32:16.780] but everything negative is just set to 0.

[00:32:19.500] And because you are basically throwing away

[00:32:22.020] half of the distribution, they find in their analysis

[00:32:24.980] of the forward activations in the neural net

[00:32:27.380] that you have to compensate for that with a gain.

[00:32:30.180] And so here, they find that, basically,

[00:32:34.180] when they initialize their weights,

[00:32:35.700] they have to do it with a zero-mean Gaussian

[00:32:37.820] whose standard deviation is square root of 2 over the Fannin.

[00:32:41.940] What we have here is we are initializing the Gaussian

[00:32:44.700] with the square root of Fannin.

[00:32:47.500] This NL here is the Fannin.

[00:32:49.060] So what we have is square root of 1 over the Fannin

[00:32:53.980] because we have the division here.

[00:32:56.700] Now, they have to add this factor of 2

[00:32:58.620] because of the ReLU, which basically discards

[00:33:01.300] half of the distribution and clamps it at 0.

[00:33:04.140] And so that's where you get an initial factor.

[00:33:06.540] Now, in addition to that, this paper also studies

[00:33:09.300] not just the behavior of the activations

[00:33:12.060] in the forward pass of the neural net,

[00:33:13.780] but it also studies the backpropagation.

[00:33:16.300] And we have to make sure that the gradients also

[00:33:18.300] are well-behaved because ultimately, they

[00:33:22.260] end up updating our parameters.

[00:33:24.260] And what they find here through a lot of the analysis

[00:33:26.860] that I invite you to read through, but it's not exactly

[00:33:29.180] approachable, what they find is basically

[00:33:32.380] if you properly initialize the forward pass,

[00:33:34.740] the backward pass is also approximately initialized

[00:33:37.860] up to a constant factor that has to do

[00:33:39.980] with the size of the number of hidden neurons

[00:33:43.740] in an early and late layer.

[00:33:48.180] But basically, they find empirically

[00:33:49.580] that this is not a choice that matters too much.

[00:33:52.620] Now, this kind of initialization is also

[00:33:55.060] implemented in PyTorch.

[00:33:56.860] So if you go to torch.nn.init documentation,

[00:33:59.380] you'll find climbing normal.

[00:34:01.220] And in my opinion, this is probably

[00:34:02.620] the most common way of initializing neural networks

[00:34:04.820] now.

[00:34:06.220] And it takes a few keyword arguments here.

[00:34:08.540] So number one, it wants to know the mode.

[00:34:11.580] Would you like to normalize the activations,

[00:34:13.420] or would you like to normalize the gradients to be always

[00:34:17.140] Gaussian with zero mean and a unit or one standard deviation?

[00:34:21.300] And because they find in the paper

[00:34:22.580] that this doesn't matter too much,

[00:34:23.980] most of the people just leave it as the default, which

[00:34:26.180] is fan in.

[00:34:27.220] And then second, pass in the nonlinearity

[00:34:29.060] that you are using.

[00:34:30.260] Because depending on the nonlinearity,

[00:34:32.340] we need to calculate a slightly different gain.

[00:34:34.900] And so if your nonlinearity is just linear,

[00:34:38.100] so there's no nonlinearity, then the gain here will be 1.

[00:34:41.220] And we have the exact same kind of formula

[00:34:43.620] that we've got up here.

[00:34:45.060] But if the nonlinearity is something else,

[00:34:46.660] we're going to get a slightly different gain.

[00:34:48.620] And so if we come up here to the top,

[00:34:50.900] we see that, for example, in the case of ReLU,

[00:34:52.980] this gain is a square root of 2.

[00:34:55.060] And the reason it's a square root,

[00:34:56.580] because in this paper, you see how the 2 is inside

[00:35:03.820] of the square root, so the gain is a square root of 2.

[00:35:07.780] In the case of linear or identity,

[00:35:10.660] we just get a gain of 1.

[00:35:12.460] In the case of 10H, which is what we're using here,

[00:35:14.860] the advised gain is a 5 over 3.

[00:35:17.580] And intuitively, why do we need a gain

[00:35:19.700] on top of the initialization?

[00:35:21.340] It's because 10H, just like ReLU,

[00:35:23.300] is a contractive transformation.

[00:35:26.020] So what that means is you're taking the output distribution

[00:35:28.420] from this matrix multiplication,

[00:35:30.100] and then you are squashing it in some way.

[00:35:32.180] Now, ReLU squashes it by taking everything below 0

[00:35:34.660] and clamping it to 0.

[00:35:36.260] 10H also squashes it because it's a contractive operation.

[00:35:39.140] It will take the tails, and it will squeeze them in.

[00:35:42.940] And so in order to fight the squeezing in,

[00:35:45.260] we need to boost the weights a little bit

[00:35:47.540] so that we renormalize everything back

[00:35:49.140] to unit standard deviation.

[00:35:51.980] So that's why there's a little bit of a gain that comes out.

[00:35:55.020] Now, I'm skipping through this section a little bit quickly,

[00:35:57.340] and I'm doing that actually intentionally.

[00:35:59.500] And the reason for that is because about seven years ago,

[00:36:02.700] when this paper was written, you had to actually be extremely

[00:36:05.820] careful with the activations and the gradients

[00:36:07.900] and their ranges and their histograms.

[00:36:09.980] And you had to be very careful with the precise setting

[00:36:12.220] of gains and the scrutinizing of the nonlinearities used

[00:36:14.500] and so on.

[00:36:15.660] And everything was very finicky and very frustrating.

[00:36:18.300] And it had to be very properly arranged for the neural net

[00:36:20.780] to train, especially if your neural net was very deep.

[00:36:23.540] But there are a number of modern innovations

[00:36:25.180] that have made everything significantly more stable

[00:36:27.220] and more well-behaved.

[00:36:28.420] And it's become less important to initialize these networks

[00:36:31.060] exactly right.

[00:36:32.780] And some of those modern innovations, for example,

[00:36:34.700] are residual connections, which we will cover in the future,

[00:36:38.020] the use of a number of normalization layers,

[00:36:41.780] like, for example, batch normalization,

[00:36:43.900] layer normalization, group normalization.

[00:36:45.820] We're going to go into a lot of these as well.

[00:36:47.620] And number three, much better optimizers,

[00:36:49.660] not just to cast a gradient descent,

[00:36:51.140] the simple optimizer we're basically using here,

[00:36:53.900] but slightly more complex optimizers,

[00:36:55.940] like RMSProp and especially Adam.

[00:36:58.380] And so all of these modern innovations

[00:36:59.780] make it less important for you to precisely calibrate

[00:37:02.740] the initialization of the neural net.

[00:37:04.780] All that being said, in practice, what should we do?

[00:37:08.380] In practice, when I initialize these neural nets,

[00:37:10.460] I basically just normalize my weights

[00:37:12.500] by the square root of the fan in.

[00:37:14.540] So basically, roughly what we did here is what I do.

[00:37:19.460] Now, if we want to be exactly accurate here,

[00:37:22.020] and go back in it of kind of normal,

[00:37:26.580] this is how we would implement it.

[00:37:28.540] We want to set the standard deviation

[00:37:29.980] to be gain over the square root of fan in.

[00:37:34.220] So to set the standard deviation of our weights,

[00:37:37.780] we will proceed as follows.

[00:37:40.060] Basically, when we have a torsade random,

[00:37:42.460] and let's say I just create a thousand numbers,

[00:37:44.860] we can look at the standard deviation of this,

[00:37:46.260] and of course, that's one, that's the amount of spread.

[00:37:48.940] Let's make this a bit bigger so it's closer to one.

[00:37:51.260] So this is the spread of the Gaussian of zero mean

[00:37:54.860] and unit standard deviation.

[00:37:56.980] Now, basically, when you take these

[00:37:58.580] and you multiply by, say, 0.2,

[00:38:01.220] that basically scales down the Gaussian,

[00:38:03.300] and that makes its standard deviation 0.2.

[00:38:05.940] So basically, the number that you multiply by here

[00:38:07.820] ends up being the standard deviation of this Gaussian.

[00:38:11.060] So here, this is a standard deviation 0.2 Gaussian here

[00:38:15.940] when we sample Rw1.

[00:38:18.380] But we want to set the standard deviation

[00:38:19.940] to gain over square root of fan load, which is fan in.

[00:38:25.220] So in other words, we want to multiply by gain,

[00:38:28.900] which for 10h is five over three.

[00:38:33.660] Five over three is the gain.

[00:38:35.860] And then divide square root of the fan in.

[00:38:50.780] And in this example here, the fan in was 10.

[00:38:53.620] And I just noticed that actually here,

[00:38:55.580] the fan in for W1 is actually an embed times block size,

[00:38:59.420] which as you will recall is actually 30.

[00:39:01.620] And that's because each character is 10-dimensional,

[00:39:03.860] but then we have three of them and we concatenate them.

[00:39:05.900] So actually, the fan in here was 30,

[00:39:07.940] and I should have used 30 here probably.

[00:39:10.140] But basically, we want 30 square root.

[00:39:13.260] So this is the number.

[00:39:14.460] This is what our standard deviation we want to be.

[00:39:17.060] And this number turns out to be 0.3.

[00:39:19.540] Whereas here, just by fiddling with it

[00:39:21.340] and looking at the distribution and making sure it looks OK,

[00:39:24.220] we came up with 0.2.

[00:39:25.940] And so instead, what we want to do here

[00:39:27.900] is we want to make the standard deviation be

[00:39:33.260] 5 over 3, which is our gain.

[00:39:34.820] Divide this amount times 0.2 square root.

[00:39:41.140] And these brackets here are not that necessary,

[00:39:44.220] but I'll just put them here for clarity.

[00:39:46.100] This is basically what we want.

[00:39:47.500] This is the Kaiming init in our case for a 10H nonlinearity.

[00:39:52.140] And this is how we would initialize the neural net.

[00:39:54.660] And so we're multiplying by 0.3 instead of multiplying by 0.2.

[00:40:00.900] And so we can initialize this way.

[00:40:05.020] And then we can train the neural net and see what we get.

[00:40:08.020] OK, so I trained the neural net, and we end up

[00:40:10.140] in roughly the same spot.

[00:40:12.220] So looking at the validation loss, we now get 2.10.

[00:40:15.140] And previously, we also had 2.10.

[00:40:17.140] There's a little bit of a difference,

[00:40:18.700] but that's just the randomness of the process, I suspect.

[00:40:21.460] But the big deal, of course, is we get to the same spot.

[00:40:24.340] But we did not have to introduce any magic numbers

[00:40:28.980] that we got from just looking at histograms and guess

[00:40:31.580] and checking.

[00:40:32.420] We have something that is semi-principled

[00:40:34.020] and will scale us to much bigger networks and something

[00:40:38.060] that we can use as a guide.

[00:40:40.140] So I mentioned that the precise setting of these initializations

[00:40:43.020] is not as important today due to some modern innovations.

[00:40:45.660] And I think now is a pretty good time

[00:40:47.220] to introduce one of those modern innovations,

[00:40:49.220] and that is batch normalization.

[00:40:51.260] So batch normalization came out in 2015 from a team at Google.

[00:40:55.820] And it was an extremely impactful paper

[00:40:57.820] because it made it possible to train very deep neural nets

[00:41:01.420] quite reliably.

[00:41:02.740] And it basically just worked.

[00:41:04.820] So here's what batch normalization does,

[00:41:06.460] and let's implement it.

[00:41:09.860] Basically, we have these hidden states hpreact, right?

[00:41:13.660] And we were talking about how we don't

[00:41:15.220] want these pre-activation states to be way too small

[00:41:20.380] because then the 10h is not doing anything.

[00:41:23.540] But we don't want them to be too large because then

[00:41:25.540] the 10h is saturated.

[00:41:27.460] In fact, we want them to be roughly Gaussian,

[00:41:30.420] so zero mean and a unit or one standard deviation,

[00:41:34.020] at least at initialization.

[00:41:36.020] So the insight from the batch normalization paper

[00:41:38.820] is, OK, you have these hidden states,

[00:41:41.060] and you'd like them to be roughly Gaussian.

[00:41:43.540] Then why not take the hidden states

[00:41:45.500] and just normalize them to be Gaussian?

[00:41:48.780] And it sounds kind of crazy, but you can just

[00:41:50.780] do that because standardizing hidden states

[00:41:55.260] so that they're Gaussian is a perfectly differentiable

[00:41:57.900] operation, as we'll soon see.

[00:41:59.580] And so that was kind of like the big insight in this paper.

[00:42:02.180] And when I first read it, my mind

[00:42:03.620] was blown because you can just normalize these hidden states.

[00:42:06.220] And if you'd like unit Gaussian states in your network,

[00:42:09.740] at least initialization, you can just normalize

[00:42:12.220] them to be unit Gaussian.

[00:42:14.260] So let's see how that works.

[00:42:16.540] So we're going to scroll to our pre-activations here

[00:42:18.700] just before they enter into the 10h.

[00:42:21.380] Now, the idea, again, is remember,

[00:42:22.780] we're trying to make these roughly Gaussian.

[00:42:24.940] And that's because if these are way too small numbers,

[00:42:27.260] then the 10h here is kind of inactive.

[00:42:30.340] But if these are very large numbers,

[00:42:32.860] then the 10h is way too saturated

[00:42:34.900] and gradient is no flow.

[00:42:36.580] So we'd like this to be roughly Gaussian.

[00:42:39.100] So the insight in batch normalization, again,

[00:42:41.500] is that we can just standardize these activations

[00:42:44.260] so they are exactly Gaussian.

[00:42:46.860] So here, hpreact has a shape of 32 by 200,

[00:42:51.980] 32 examples by 200 neurons in the hidden layer.

[00:42:56.020] So basically what we can do is we can take hpreact

[00:42:58.220] and we can just calculate the mean.

[00:43:01.220] And the mean we want to calculate

[00:43:02.940] across the 0th dimension.

[00:43:05.380] And we want to also keep them as true

[00:43:08.020] so that we can easily broadcast this.

[00:43:11.620] So the shape of this is 1 by 200.

[00:43:14.780] In other words, we are doing the mean over all

[00:43:17.180] the elements in the batch.

[00:43:20.860] And similarly, we can calculate the standard deviation

[00:43:23.980] of these activations.

[00:43:26.940] And that will also be 1 by 200.

[00:43:29.420] Now in this paper, they have the sort of prescription here.

[00:43:34.580] And see here, we are calculating the mean,

[00:43:36.860] which is just taking the average value of any neuron's

[00:43:42.660] activation.

[00:43:43.780] And then the standard deviation is basically

[00:43:45.620] kind of like the measure of the spread

[00:43:48.900] that we've been using, which is the distance of every one

[00:43:53.140] of these values away from the mean,

[00:43:55.060] and that squared and averaged.

[00:43:58.740] That's the variance.

[00:44:01.260] And then if you want to take the standard deviation,

[00:44:03.340] you would square root the variance

[00:44:05.300] to get the standard deviation.

[00:44:07.820] So these are the two that we're calculating.

[00:44:10.100] And now we're going to normalize or standardize

[00:44:12.620] these x's by subtracting the mean

[00:44:14.300] and dividing by the standard deviation.

[00:44:17.820] So basically, we're taking edge preact,

[00:44:20.660] and we subtract the mean, and then we

[00:44:30.500] divide by the standard deviation.

[00:44:34.380] This is exactly what these two, STD and mean, are calculating.

[00:44:38.420] Oops.

[00:44:40.460] Sorry.

[00:44:40.980] This is the mean, and this is the variance.

[00:44:43.060] You see how the sigma is the standard deviation usually.

[00:44:45.420] So this is sigma squared, which the variance

[00:44:47.460] is the square of the standard deviation.

[00:44:50.900] So this is how you standardize these values.

[00:44:53.140] And what this will do is that every single neuron now

[00:44:55.820] and its firing rate will be exactly unit Gaussian

[00:44:58.860] on these 32 examples, at least, of this batch.

[00:45:01.700] That's why it's called batch normalization.

[00:45:03.420] We are normalizing these batches.

[00:45:06.700] And then we could, in principle, train this.

[00:45:09.500] Notice that calculating the mean and the standard deviation,

[00:45:12.100] these are just mathematical formulas.

[00:45:13.660] They're perfectly differentiable.

[00:45:15.180] All this is perfectly differentiable,

[00:45:16.780] and we can just train this.

[00:45:18.860] The problem is you actually won't achieve a very good

[00:45:21.700] result with this.

[00:45:23.180] And the reason for that is we want

[00:45:26.220] these to be roughly Gaussian, but only at initialization.

[00:45:29.700] But we don't want these to be forced to be Gaussian always.

[00:45:34.220] We'd like to allow the neural net to move this around

[00:45:37.580] to potentially make it more diffuse, to make it more sharp,

[00:45:40.620] to make some 10-H neurons maybe be more trigger happy

[00:45:44.060] or less trigger happy.

[00:45:45.540] So we'd like this distribution to move around,

[00:45:47.500] and we'd like the backpropagation

[00:45:48.780] to tell us how the distribution should move around.

[00:45:52.460] And so in addition to this idea of standardizing

[00:45:55.780] the activations at any point in the network,

[00:45:59.300] we have to also introduce this additional component

[00:46:01.620] in the paper here described as scale and shift.

[00:46:05.420] And so basically what we're doing is we're

[00:46:07.140] taking these normalized inputs, and we are additionally

[00:46:10.380] scaling them by some gain and offsetting them by some bias

[00:46:14.300] to get our final output from this layer.

[00:46:17.820] And so what that amounts to is the following.

[00:46:20.420] We are going to allow a batch normalization gain

[00:46:23.860] to be initialized at just a 1s, and the 1s

[00:46:28.260] will be in the shape of 1 by n hidden.

[00:46:32.380] And then we also will have a bn bias,

[00:46:35.260] which will be torched at 0s, and it will also

[00:46:38.380] be of the shape 1 by n hidden.

[00:46:42.260] And then here, the bn gain will multiply this,

[00:46:47.340] and the bn bias will offset it here.

[00:46:51.140] So because this is initialized to 1 and this to 0,

[00:46:54.900] at initialization, each neuron's firing values in this batch

[00:46:59.700] will be exactly unit Gaussian, and will have nice numbers.

[00:47:03.580] No matter what the distribution of the HP act is coming in,

[00:47:07.100] coming out, it will be unit Gaussian for each neuron,

[00:47:09.740] and that's roughly what we want, at least at initialization.

[00:47:13.900] And then during optimization, we'll

[00:47:15.500] be able to backpropagate to bn gain and bn bias

[00:47:18.460] and change them so the network is given the full ability

[00:47:21.140] to do with this whatever it wants internally.

[00:47:25.700] Here, we just have to make sure that we include

[00:47:29.660] these in the parameters of the neural net

[00:47:32.100] because they will be trained with backpropagation.

[00:47:35.700] So let's initialize this, and then we

[00:47:38.060] should be able to train.

[00:47:45.660] And then we're going to also copy this line, which

[00:47:49.900] is the batch normalization layer,

[00:47:51.900] here on a single line of code, and we're

[00:47:53.860] going to swing down here, and we're also

[00:47:55.500] going to do the exact same thing at test time here.

[00:48:01.700] So similar to train time, we're going to normalize and then

[00:48:05.340] scale, and that's going to give us our train and validation

[00:48:08.740] loss.

[00:48:10.180] And we'll see in a second that we're actually

[00:48:12.020] going to change this a little bit, but for now,

[00:48:14.020] I'm going to keep it this way.

[00:48:15.740] So I'm just going to wait for this to converge.

[00:48:17.620] OK, so I allowed the neural nets to converge here,

[00:48:19.900] and when we scroll down, we see that our validation loss here

[00:48:22.580] is 2.10, roughly, which I wrote down here.

[00:48:25.940] And we see that this is actually kind of comparable to some

[00:48:28.420] of the results that we've achieved previously.

[00:48:31.220] Now, I'm not actually expecting an improvement in this case,

[00:48:34.860] and that's because we are dealing

[00:48:35.860] with a very simple neural net that has just

[00:48:37.660] a single hidden layer.

[00:48:39.420] So in fact, in this very simple case of just one hidden layer,

[00:48:43.100] we were able to actually calculate

[00:48:44.420] what the scale of W should be to make these pre-activations

[00:48:48.380] already have a roughly Gaussian shape.

[00:48:50.300] So the batch normalization is not doing much here.

[00:48:53.100] But you might imagine that once you

[00:48:54.500] have a much deeper neural net that

[00:48:56.300] has lots of different types of operations,

[00:48:59.060] and there's also, for example, residual connections,

[00:49:01.140] which we'll cover, and so on, it will become basically very,

[00:49:04.620] very difficult to tune the scales of your weight matrices

[00:49:08.900] such that all the activations throughout the neural net

[00:49:11.180] are roughly Gaussian.

[00:49:12.980] And so that's going to become very quickly intractable.

[00:49:16.020] But compared to that, it's going to be much, much easier

[00:49:18.820] to sprinkle batch normalization layers

[00:49:20.700] throughout the neural net.

[00:49:22.220] So in particular, it's common to look

[00:49:24.940] at every single linear layer like this one.

[00:49:27.060] This is a linear layer multiplying by a weight matrix

[00:49:29.140] and adding a bias.

[00:49:30.940] Or, for example, convolutions, which we'll cover later,

[00:49:33.380] and also perform basically a multiplication

[00:49:36.340] with a weight matrix, but in a more spatially structured

[00:49:38.820] format, it's customary to take these linear layer

[00:49:42.500] or convolutional layer and append a batch normalization

[00:49:46.060] layer right after it to control the scale

[00:49:49.100] of these activations at every point in the neural net.

[00:49:51.820] So we'd be adding these batch normal layers

[00:49:53.540] throughout the neural net, and then

[00:49:55.140] this controls the scale of these activations

[00:49:57.220] throughout the neural net.

[00:49:58.700] It doesn't require us to do perfect mathematics

[00:50:01.820] and care about the activation distributions

[00:50:04.140] for all these different types of neural network

[00:50:06.420] Lego building blocks that you might want to introduce

[00:50:08.180] into your neural net.

[00:50:09.460] And it significantly stabilizes the train,

[00:50:12.340] and that's why these layers are quite popular.

[00:50:14.940] Now, the stability offered by batch normalization

[00:50:16.940] actually comes at a terrible cost.

[00:50:19.020] And that cost is that if you think

[00:50:20.740] about what's happening here, something terribly strange

[00:50:24.180] and unnatural is happening.

[00:50:26.560] It used to be that we have a single example feeding

[00:50:29.180] into a neural net, and then we calculate its activations

[00:50:32.980] and its logits.

[00:50:34.340] And this is a deterministic process,

[00:50:37.500] so you arrive at some logits for this example.

[00:50:40.300] And then because of efficiency of training,

[00:50:42.420] we suddenly started to use batches of examples.

[00:50:44.860] But those batches of examples were processed independently,

[00:50:47.620] and it was just an efficiency thing.

[00:50:49.900] But now suddenly, in batch normalization,

[00:50:51.580] because of the normalization through the batch,

[00:50:53.740] we are coupling these examples mathematically

[00:50:56.660] and in the forward pass and the backward pass of a neural net.

[00:50:59.560] So now, the hidden state activations,

[00:51:01.960] hpreact and your logits for any one input example

[00:51:05.660] are not just a function of that example and its input,

[00:51:08.460] but they're also a function of all the other examples that

[00:51:11.100] happen to come for a ride in that batch.

[00:51:14.580] And these examples are sampled randomly.

[00:51:16.600] And so what's happening is, for example,

[00:51:17.940] when you look at hpreact that's going to feed into h,

[00:51:20.780] the hidden state activations, for example,

[00:51:23.020] for any one of these input examples,

[00:51:25.500] is going to actually change slightly,

[00:51:27.820] depending on what other examples there are in the batch.

[00:51:30.460] And depending on what other examples

[00:51:32.300] happen to come for a ride, h is going to change suddenly,

[00:51:36.380] and it's going to jitter, if you imagine

[00:51:38.100] sampling different examples.

[00:51:39.660] Because the statistics of the mean and the standard deviation

[00:51:42.220] are going to be impacted.

[00:51:44.140] And so you'll get a jitter for h,

[00:51:45.820] and you'll get a jitter for logits.

[00:51:48.740] And you'd think that this would be a bug or something

[00:51:51.420] undesirable.

[00:51:52.540] But in a very strange way, this actually

[00:51:55.060] turns out to be good in neural network training

[00:51:58.620] as a side effect.

[00:51:59.740] And the reason for that is that you

[00:52:01.140] can think of this as kind of like a regularizer.

[00:52:03.540] Because what's happening is you have your input,

[00:52:05.500] and you get your h.

[00:52:06.500] And then depending on the other examples,

[00:52:08.420] this is jittering a bit.

[00:52:10.020] And so what that does is that it's effectively padding out

[00:52:12.820] any one of these input examples.

[00:52:14.380] And it's introducing a little bit of entropy.

[00:52:16.500] And because of the padding out, it's

[00:52:18.900] actually kind of like a form of a data augmentation, which

[00:52:21.620] we'll cover in the future.

[00:52:23.100] And it's kind of like augmenting the input a little bit,

[00:52:25.820] and it's jittering it.

[00:52:26.860] And that makes it harder for the neural net

[00:52:28.660] to overfit these concrete specific examples.

[00:52:32.100] So by introducing all this noise,

[00:52:33.740] it actually like pads out the examples,

[00:52:35.700] and it regularizes the neural net.

[00:52:37.780] And that's one of the reasons why, deceivingly,

[00:52:40.660] as a second-order effect, this is actually a regularizer.

[00:52:43.700] And that has made it harder for us

[00:52:45.740] to remove the use of batch normalization.

[00:52:48.740] Because basically, no one likes this property that the examples

[00:52:52.300] in the batch are coupled mathematically

[00:52:54.140] and in the forward pass.

[00:52:55.660] And it leads to all kinds of strange results.

[00:52:58.740] We'll go into some of that in a second as well.

[00:53:01.740] And it leads to a lot of bugs and so on.

[00:53:04.900] And so no one likes this property.

[00:53:06.980] And so people have tried to deprecate

[00:53:09.900] the use of batch normalization and move to other normalization

[00:53:12.380] techniques that do not couple the examples of a batch.

[00:53:14.780] Examples are linear normalization,

[00:53:16.780] instance normalization, group normalization, and so on.

[00:53:19.980] And we'll cover some of these later.

[00:53:24.180] But basically, long story short, batch normalization

[00:53:26.340] was the first kind of normalization layer

[00:53:28.140] to be introduced.

[00:53:29.100] It worked extremely well.

[00:53:30.860] It happened to have this regularizing effect.

[00:53:33.420] It stabilized training.

[00:53:35.860] And people have been trying to remove it and move

[00:53:38.540] to some of the other normalization techniques.

[00:53:40.860] But it's been hard because it just works quite well.

[00:53:44.220] And some of the reason that it works quite well

[00:53:46.220] is, again, because of this regularizing effect

[00:53:48.100] and because it is quite effective at controlling

[00:53:51.780] the activations and their distributions.

[00:53:54.500] So that's kind of like the brief story of batch normalization.

[00:53:57.380] And I'd like to show you one of the other weird sort

[00:54:00.900] of outcomes of this coupling.

[00:54:03.460] So here's one of the strange outcomes

[00:54:05.020] that I only glossed over previously

[00:54:07.620] when I was evaluating the loss on the validation set.

[00:54:10.820] Basically, once we've trained a neural net,

[00:54:13.220] we'd like to deploy it in some kind of a setting.

[00:54:15.580] And we'd like to be able to feed in a single individual

[00:54:17.940] example and get a prediction out from our neural net.

[00:54:21.380] But how do we do that when our neural net now

[00:54:23.420] in the forward pass estimates the statistics

[00:54:25.780] of the mean and standard deviation of a batch?

[00:54:27.900] The neural net expects batches as an input now.

[00:54:30.460] So how do we feed in a single example

[00:54:32.060] and get sensible results out?

[00:54:34.420] And so the proposal in the batch normalization paper

[00:54:37.300] is the following.

[00:54:38.860] What we would like to do here is we

[00:54:40.660] would like to basically have a step after training that

[00:54:45.620] calculates and sets the batch norm mean and standard

[00:54:49.020] deviation a single time over the training set.

[00:54:52.180] And so I wrote this code here in interest of time.

[00:54:55.260] And we're going to call what's called calibrate

[00:54:57.180] the batch norm statistics.

[00:54:59.060] And basically, what we do is torsnot no grad,

[00:55:02.460] telling PyTorch that none of this

[00:55:04.540] we will call the dot backward on.

[00:55:06.460] And it's going to be a bit more efficient.

[00:55:08.860] We're going to take the training set,

[00:55:10.660] get the preactivations for every single training example,

[00:55:13.540] and then one single time estimate the mean and standard

[00:55:15.740] deviation over the entire training set.

[00:55:18.140] And then we're going to get bn mean and bn standard deviation.

[00:55:20.860] And now these are fixed numbers estimated

[00:55:23.260] over the entire training set.

[00:55:25.180] And here, instead of estimating it dynamically,

[00:55:29.820] we are going to instead here use bn mean.

[00:55:34.220] And here, we're just going to use bn standard deviation.

[00:55:38.020] And so at test time, we are going

[00:55:39.660] to fix these, clamp them, and use them during inference.

[00:55:43.060] And now you see that we get basically identical result.

[00:55:48.900] But the benefit that we've gained

[00:55:50.620] is that we can now also forward a single example

[00:55:53.180] because the mean and standard deviation are now fixed

[00:55:55.740] sort of tensors.

[00:55:57.340] That said, nobody actually wants to estimate

[00:55:59.380] this mean and standard deviation as a second stage

[00:56:02.420] after neural network training because everyone is lazy.

[00:56:05.660] And so this batch normalization paper

[00:56:07.740] actually introduced one more idea,

[00:56:09.500] which is that we can estimate the mean and standard

[00:56:11.940] deviation in a running manner during training

[00:56:15.780] of the neural net.

[00:56:17.060] And then we can simply just have a single stage of training.

[00:56:20.100] And on the side of that training,

[00:56:21.700] we are estimating the running mean and standard deviation.

[00:56:24.540] So let's see what that would look like.

[00:56:26.700] Let me basically take the mean here

[00:56:28.700] that we are estimating on the batch.

[00:56:30.100] And let me call this bn mean on the ith iteration.

[00:56:35.420] And then here, this is bn std at i.

[00:56:47.060] And the mean comes here, and the std comes here.

[00:56:53.020] So so far, I've done nothing.

[00:56:54.180] I've just moved around, and I created these extra variables

[00:56:56.820] for the mean and standard deviation.

[00:56:58.460] And I've put them here.

[00:56:59.860] So so far, nothing has changed.

[00:57:01.380] But what we're going to do now is

[00:57:02.780] we're going to keep a running mean of both of these values

[00:57:05.420] during training.

[00:57:06.580] So let me swing up here.

[00:57:07.620] And let me create a bn mean underscore running.

[00:57:11.940] And I'm going to initialize it at zeros.

[00:57:16.060] And then bn std running, which I'll initialize at once.

[00:57:23.180] Because in the beginning, because of the way

[00:57:25.940] we initialized w1 and b1, each preact

[00:57:29.780] will be roughly unit Gaussian.

[00:57:31.180] So the mean will be roughly 0, and the standard deviation

[00:57:33.420] roughly 1.

[00:57:34.540] So I'm going to initialize these that way.

[00:57:37.180] But then here, I'm going to update these.

[00:57:39.460] And in PyTorch, these mean and standard deviation

[00:57:44.180] that are running, they're not actually

[00:57:46.020] part of the gradient-based optimization.

[00:57:47.740] We're never going to derive gradients with respect to them.

[00:57:50.220] They're updated on the side of training.

[00:57:53.420] And so what we're going to do here

[00:57:54.740] is we're going to say with torch.nograd telling PyTorch

[00:57:58.820] that the update here is not supposed

[00:58:01.580] to be building out a graph, because there

[00:58:03.220] will be no dot backward.

[00:58:05.340] But this running mean is basically

[00:58:07.420] going to be 0.999 times the current value

[00:58:13.580] plus 0.001 times this value, this new mean.

[00:58:20.380] And in the same way, bn std running

[00:58:23.020] will be mostly what it used to be.

[00:58:25.820] But it will receive a small update

[00:58:27.500] in the direction of what the current standard deviation is.

[00:58:32.180] And as you're seeing here, this update

[00:58:33.940] is outside and on the side of the gradient-based optimization.

[00:58:38.420] And it's simply being updated not using gradient descent.

[00:58:40.940] It's just being updated using a janky, smooth running mean

[00:58:48.780] manner.

[00:58:50.460] And so while the network is training,

[00:58:52.500] and these pre-activations are sort of changing and shifting

[00:58:55.660] around during backpropagation, we

[00:58:58.180] are keeping track of the typical mean and standard deviation,

[00:59:00.980] and we're estimating them once.

[00:59:02.620] And when I run this, now I'm keeping track

[00:59:07.500] of this in the running matter.

[00:59:09.060] And what we're hoping for, of course,

[00:59:10.660] is that the bn mean underscore running and bn mean underscore

[00:59:13.900] std are going to be very similar to the ones that we calculated

[00:59:17.940] here before.

[00:59:19.580] And that way, we don't need a second stage, because we've

[00:59:22.580] sort of combined the two stages, and we've

[00:59:24.500] put them on the side of each other,

[00:59:26.020] if you want to look at it that way.

[00:59:28.060] And this is how this is also implemented

[00:59:29.620] in the batch normalization layer in PyTorch.

[00:59:32.300] So during training, the exact same thing will happen.

[00:59:36.460] And then later, when you're using inference,

[00:59:38.500] it will use the estimated running

[00:59:40.420] mean of both the mean and standard deviation

[00:59:43.180] of those hidden states.

[00:59:45.260] So let's wait for the optimization

[00:59:46.700] to complete, and then we'll go ahead

[00:59:48.260] and let's wait for the optimization to converge.

[00:59:50.260] And hopefully, the running mean and standard deviation

[00:59:52.420] are roughly equal to these two.

[00:59:53.980] And then we can simply use it here.

[00:59:55.940] And we don't need this stage of explicit calibration

[00:59:58.500] at the end.

[00:59:59.300] OK, so the optimization finished.

[01:00:01.460] I'll rerun the explicit estimation.

[01:00:03.980] And then the bn mean from the explicit estimation is here.

[01:00:07.860] And bn mean from the running estimation

[01:00:11.140] during the optimization you can see is very, very similar.

[01:00:16.300] It's not identical, but it's pretty close.

[01:00:19.620] And in the same way, bnstd is this.

[01:00:22.620] And bnstd running is this.

[01:00:26.420] As you can see that, once again, they are fairly similar values.

[01:00:29.460] Not identical, but pretty close.

[01:00:31.900] And so then here, instead of bn mean,

[01:00:33.700] we can use the bn mean running.

[01:00:35.980] Instead of bnstd, we can use bnstd running.

[01:00:39.820] And hopefully, the validation loss

[01:00:42.020] will not be impacted too much.

[01:00:44.460] OK, so it's basically identical.

[01:00:46.700] And this way, we've eliminated the need

[01:00:49.260] for this explicit stage of calibration

[01:00:51.620] because we are doing it inline over here.

[01:00:53.940] OK, so we're almost done with batch normalization.

[01:00:56.060] There are only two more notes that I'd like to make.

[01:00:58.460] Number one, I've skipped a discussion

[01:00:59.980] over what is this plus epsilon doing here.

[01:01:02.180] This epsilon is usually like some small fixed number.

[01:01:04.900] For example, 1e negative 5 by default.

[01:01:06.980] And what it's doing is that it's basically

[01:01:08.700] preventing a division by 0.

[01:01:10.700] In the case that the variance over your batch

[01:01:14.420] is exactly 0.

[01:01:15.900] In that case, here, we normally have a division by 0.

[01:01:19.020] But because of the plus epsilon, this

[01:01:20.980] is going to become a small number in the denominator

[01:01:23.020] instead.

[01:01:23.580] And things will be more well-behaved.

[01:01:25.620] So feel free to also add a plus epsilon here

[01:01:27.940] of a very small number.

[01:01:29.140] It doesn't actually substantially change the result.

[01:01:31.220] I'm going to skip it in our case just

[01:01:32.540] because this is unlikely to happen

[01:01:34.140] in our very simple example here.

[01:01:36.300] And the second thing I want you to notice

[01:01:38.140] is that we're being wasteful here.

[01:01:39.540] And it's very subtle.

[01:01:41.220] But right here, where we are adding

[01:01:42.700] the bias into H preact, these biases now

[01:01:46.620] are actually useless because we're adding them

[01:01:48.900] to the H preact.

[01:01:50.460] But then we are calculating the mean

[01:01:52.820] for every one of these neurons and subtracting it.

[01:01:55.940] So whatever bias you add here is going

[01:01:58.340] to get subtracted right here.

[01:02:00.820] And so these biases are not doing anything.

[01:02:02.820] In fact, they're being subtracted out.

[01:02:04.580] And they don't impact the rest of the calculation.

[01:02:07.340] So if you look at B1.grad, it's actually

[01:02:09.140] going to be 0 because it's being subtracted out

[01:02:11.620] and doesn't actually have any effect.

[01:02:13.580] And so whenever you're using batch normalization layers,

[01:02:16.060] then if you have any weight layers before,

[01:02:18.020] like a linear or a comb or something like that,

[01:02:20.580] you're better off coming here and just not using bias.

[01:02:24.220] So you don't want to use bias.

[01:02:26.220] And then here, you don't want to add it

[01:02:28.940] because it's that spurious.

[01:02:30.580] Instead, we have this batch normalization bias here.

[01:02:33.700] And that batch normalization bias

[01:02:35.220] is now in charge of the biasing of this distribution

[01:02:38.860] instead of this B1 that we had here originally.

[01:02:42.220] And so basically, the batch normalization layer

[01:02:44.740] has its own bias.

[01:02:45.860] And there's no need to have a bias in the layer

[01:02:48.420] before it because that bias is going

[01:02:50.180] to be subtracted out anyway.

[01:02:51.980] So that's the other small detail to be careful with sometimes.

[01:02:54.580] It's not going to do anything catastrophic.

[01:02:56.660] This B1 will just be useless.

[01:02:58.500] It will never get any gradient.

[01:03:00.340] It will not learn.

[01:03:01.100] It will stay constant.

[01:03:02.060] And it's just wasteful.

[01:03:03.100] But it doesn't actually really impact anything otherwise.

[01:03:07.140] OK, so I rearranged the code a little bit with comments.

[01:03:09.780] And I just wanted to give a very quick summary

[01:03:11.660] of the batch normalization layer.

[01:03:13.740] We are using batch normalization to control

[01:03:15.860] the statistics of activations in the neural net.

[01:03:19.660] It is common to sprinkle batch normalization

[01:03:21.500] layer across the neural net.

[01:03:23.180] And usually, we will place it after layers

[01:03:26.060] that have multiplications, like, for example,

[01:03:28.460] a linear layer or a convolutional layer,

[01:03:30.660] which we may cover in the future.

[01:03:33.260] Now, the batch normalization internally has parameters

[01:03:37.700] for the gain and the bias.

[01:03:39.500] And these are trained using backpropagation.

[01:03:41.820] It also has two buffers.

[01:03:44.500] The buffers are the mean and the standard deviation,

[01:03:47.140] the running mean and the running mean of the standard deviation.

[01:03:51.020] And these are not trained using backpropagation.

[01:03:53.020] These are trained using this janky update of kind

[01:03:56.580] of like a running mean update.

[01:03:58.980] So these are sort of the parameters and the buffers

[01:04:03.660] of batch normalization.

[01:04:05.260] And then really what it's doing is

[01:04:06.700] it's calculating the mean and the standard deviation

[01:04:08.940] of the activations that are feeding into the batch normalization

[01:04:12.980] over that batch.

[01:04:14.940] Then it's centering that batch to be unit Gaussian.

[01:04:18.580] And then it's offsetting and scaling it

[01:04:20.460] by the learned bias and gain.

[01:04:24.180] And then on top of that, it's keeping

[01:04:25.700] track of the mean and standard deviation of the inputs.

[01:04:28.940] And it's maintaining this running mean and standard

[01:04:31.420] deviation.

[01:04:32.780] And this will later be used at inference

[01:04:34.980] so that we don't have to re-estimate the mean

[01:04:36.940] and standard deviation all the time.

[01:04:38.980] And in addition, that allows us to basically forward

[01:04:41.460] individual examples at test time.

[01:04:44.300] So that's the batch normalization layer.

[01:04:45.940] It's a fairly complicated layer.

[01:04:48.420] But this is what it's doing internally.

[01:04:50.460] Now, I wanted to show you a little bit of a real example.

[01:04:53.300] So you can search ResNet, which is a residual neural network.

[01:04:57.780] And these are context of neural networks

[01:04:59.860] used for image classification.

[01:05:02.140] And of course, we haven't come to ResNets in detail.

[01:05:04.700] So I'm not going to explain all the pieces of it.

[01:05:07.780] But for now, just note that the image feeds into a ResNet

[01:05:11.220] on the top here.

[01:05:12.220] And there's many, many layers with repeating structure

[01:05:15.180] all the way to predictions of what's inside that image.

[01:05:18.380] This repeating structure is made up of these blocks.

[01:05:20.860] And these blocks are just sequentially stacked up

[01:05:23.140] in this deep neural network.

[01:05:25.660] Now, the code for this, the block basically that's used

[01:05:29.700] and repeated sequentially in series,

[01:05:32.420] is called this bottleneck block.

[01:05:36.180] And there's a lot here.

[01:05:37.460] This is all PyTorch.

[01:05:38.500] And of course, we haven't covered all of it.

[01:05:40.340] But I want to point out some small pieces of it.

[01:05:43.220] Here in the init is where we initialize the neural net.

[01:05:45.660] So this code of block here is basically the kind of stuff

[01:05:48.260] we're doing here.

[01:05:48.940] We're initializing all the layers.

[01:05:51.060] And in the forward, we are specifying

[01:05:53.020] how the neural net acts once you actually have the input.

[01:05:55.860] So this code here is along the lines

[01:05:57.900] of what we're doing here.

[01:06:01.700] And now these blocks are replicated and stacked up

[01:06:04.780] serially.

[01:06:05.780] And that's what a residual network would be.

[01:06:09.020] And so notice what's happening here.

[01:06:10.980] Conv1, these are convolutional layers.

[01:06:14.980] And these convolutional layers, basically,

[01:06:16.740] they're the same thing as a linear layer,

[01:06:19.580] except convolutional layers don't

[01:06:21.020] apply convolutional layers are used for images.

[01:06:24.860] And so they have spatial structure.

[01:06:26.620] And basically, this linear multiplication and bias offset

[01:06:29.620] are done on patches instead of the full input.

[01:06:34.780] So because these images have structure, spatial structure,

[01:06:37.940] convolutions just basically do wx plus b.

[01:06:40.820] But they do it on overlapping patches of the input.

[01:06:43.940] But otherwise, it's wx plus b.

[01:06:46.740] Then we have the norm layer, which by default

[01:06:48.820] here is initialized to be a batch norm in 2D,

[01:06:51.300] so two-dimensional batch normalization layer.

[01:06:54.260] And then we have a nonlinearity like ReLU.

[01:06:56.660] So instead of here they use ReLU,

[01:06:59.620] we are using 10H in this case.

[01:07:02.580] But both are just nonlinearities,

[01:07:04.500] and you can just use them relatively interchangeably.

[01:07:07.340] For very deep networks, ReLUs typically empirically

[01:07:09.980] work a bit better.

[01:07:11.860] So see the motif that's being repeated here.

[01:07:14.140] We have convolution, batch normalization, ReLU.

[01:07:16.540] Convolution, batch normalization, ReLU, et cetera.

[01:07:19.180] And then here, this is a residual connection

[01:07:21.060] that we haven't covered yet.

[01:07:23.020] But basically, that's the exact same pattern we have here.

[01:07:25.380] We have a weight layer, like a convolution

[01:07:28.660] or like a linear layer, batch normalization,

[01:07:32.500] and then 10H, which is a nonlinearity.

[01:07:35.580] But basically, a weight layer, a normalization layer,

[01:07:38.340] and nonlinearity.

[01:07:39.380] And that's the motif that you would be stacking up

[01:07:41.540] when you create these deep neural networks, exactly

[01:07:44.060] as is done here.

[01:07:45.220] And one more thing I'd like you to notice

[01:07:46.980] is that here, when they are initializing the conv layers,

[01:07:50.220] like conv one by one, the depth for that is right here.

[01:07:54.540] And so it's initializing an nn.conf2d,

[01:07:56.780] which is a convolution layer in PyTorch.

[01:07:58.660] And there's a bunch of keyword arguments here

[01:08:00.420] that I'm not going to explain yet.

[01:08:02.260] But you see how there's bias equals false.

[01:08:04.780] The bias equals false is exactly for the same reason

[01:08:07.100] as bias is not used in our case.

[01:08:10.100] You see how I erase the use of bias.

[01:08:12.140] And the use of bias is spurious, because after this weight

[01:08:14.820] layer, there's a bastion normalization.

[01:08:16.820] And the bastion normalization subtracts that bias

[01:08:19.220] and then has its own bias.

[01:08:20.660] So there's no need to introduce these spurious parameters.

[01:08:23.220] It wouldn't hurt performance, it's just useless.

[01:08:25.860] And so because they have this motif of conv bastion

[01:08:28.980] and relu, they don't need a bias here,

[01:08:31.100] because there's a bias inside here.

[01:08:33.460] So by the way, this example here is very easy to find.

[01:08:36.940] Just do a resnet PyTorch, and it's this example here.

[01:08:41.660] So this is kind of like the stock implementation

[01:08:43.660] of a residual neural network in PyTorch.

[01:08:46.340] And you can find that here.

[01:08:48.180] But of course, I haven't covered many of these parts yet.

[01:08:50.740] And I would also like to briefly descend

[01:08:52.420] into the definitions of these PyTorch layers

[01:08:55.180] and the parameters that they take.

[01:08:56.820] Now, instead of a convolutional layer,

[01:08:58.340] we're going to look at a linear layer,

[01:09:01.020] because that's the one that we're using here.

[01:09:02.900] This is a linear layer, and I haven't covered convolutions

[01:09:05.500] yet.

[01:09:06.180] But as I mentioned, convolutions are basically linear layers

[01:09:08.740] except on patches.

[01:09:11.260] So a linear layer performs a wx plus b,

[01:09:14.500] except here they're calling the wa transpose.

[01:09:18.820] So it's called wx plus b, very much like we did here.

[01:09:21.500] To initialize this layer, you need

[01:09:22.780] to know the fan in, the fan out.

[01:09:25.660] And that's so that they can initialize this w.

[01:09:29.460] This is the fan in and the fan out.

[01:09:32.020] So they know how big the weight matrix should be.

[01:09:35.620] You need to also pass in whether or not you want a bias.

[01:09:39.020] And if you set it to false, then no bias

[01:09:41.020] will be inside this layer.

[01:09:44.420] And you may want to do that exactly like in our case,

[01:09:47.140] if your layer is followed by a normalization

[01:09:49.180] layer such as batch norm.

[01:09:51.780] So this allows you to basically disable bias.

[01:09:54.260] Now, in terms of the initialization,

[01:09:55.720] if we swing down here, this is reporting the variables used

[01:09:58.780] inside this linear layer.

[01:10:01.020] And our linear layer here has two parameters, the weight

[01:10:04.780] and the bias.

[01:10:05.900] In the same way, they have a weight and a bias.

[01:10:08.660] And they're talking about how they initialize it by default.

[01:10:11.900] So by default, PyTorch will initialize your weights

[01:10:14.340] by taking the fan in and then doing 1 over fan in square

[01:10:19.500] root.

[01:10:20.900] And then instead of a normal distribution,

[01:10:23.660] they are using a uniform distribution.

[01:10:25.900] So it's very much the same thing.

[01:10:27.980] But they are using a 1 instead of 5 over 3.

[01:10:30.540] So there's no gain being calculated here.

[01:10:32.600] The gain is just 1.

[01:10:33.700] But otherwise, it's exactly 1 over the square root of fan in

[01:10:37.860] exactly as we have here.

[01:10:40.580] So 1 over the square root of k is the scale of the weights.

[01:10:45.140] But when they are drawing the numbers,

[01:10:46.660] they're not using a Gaussian by default.

[01:10:48.820] They're using a uniform distribution by default.

[01:10:51.500] And so they draw uniformly from negative square root of k

[01:10:54.340] to square root of k.

[01:10:56.140] But it's the exact same thing and the same motivation

[01:11:00.180] with respect to what we've seen in this lecture.

[01:11:03.140] And the reason they're doing this is,

[01:11:04.700] if you have a roughly Gaussian input,

[01:11:06.740] this will ensure that out of this layer,

[01:11:09.500] you will have a roughly Gaussian output.

[01:11:11.860] And you basically achieve that by scaling the weights

[01:11:15.300] by 1 over the square root of fan in.

[01:11:17.840] So that's what this is doing.

[01:11:20.100] And then the second thing is the batch normalization layer.

[01:11:23.260] So let's look at what that looks like in PyTorch.

[01:11:26.220] So here we have a one-dimensional batch

[01:11:27.720] normalization layer exactly as we are using here.

[01:11:30.580] And there are a number of keyword arguments going into it

[01:11:33.120] as well.

[01:11:33.700] So we need to know the number of features.

[01:11:35.740] For us, that is 200.

[01:11:37.460] And that is needed so that we can initialize

[01:11:39.420] these parameters here, the gain, the bias,

[01:11:42.400] and the buffers for the running mean and standard deviation.

[01:11:47.060] Then they need to know the value of epsilon here.

[01:11:49.980] And by default, this is 1 negative 5.

[01:11:51.780] You don't typically change this too much.

[01:11:54.000] Then they need to know the momentum.

[01:11:55.980] And the momentum here, as they explain,

[01:11:58.220] is basically used for these running mean and running

[01:12:01.420] standard deviation.

[01:12:02.820] So by default, the momentum here is 0.1.

[01:12:05.100] The momentum we are using here in this example is 0.001.

[01:12:09.780] And basically, you may want to change this sometimes.

[01:12:13.740] And roughly speaking, if you have a very large batch size,

[01:12:17.340] then typically what you'll see is

[01:12:18.980] that when you estimate the mean and the standard deviation,

[01:12:21.660] for every single batch size, if it's large enough,

[01:12:23.740] you're going to get roughly the same result.

[01:12:26.140] And so therefore, you can use slightly higher momentum,

[01:12:29.500] like 0.1.

[01:12:31.080] But for a batch size as small as 32,

[01:12:34.720] the mean and standard deviation here

[01:12:36.060] might take on slightly different numbers,

[01:12:37.820] because there's only 32 examples we

[01:12:39.300] are using to estimate the mean and standard deviation.

[01:12:41.980] So the value is changing around a lot.

[01:12:44.300] And if your momentum is 0.1, that

[01:12:46.580] might not be good enough for this value

[01:12:48.380] to settle and converge to the actual mean and standard

[01:12:52.620] deviation over the entire training set.

[01:12:55.220] And so basically, if your batch size is very small,

[01:12:57.540] momentum of 0.1 is potentially dangerous.

[01:12:59.820] And it might make it so that the running mean and standard

[01:13:02.580] deviation is thrashing too much during training,

[01:13:05.260] and it's not actually converging properly.

[01:13:09.500] Affine equals true determines whether this batch normalization

[01:13:12.700] layer has these learnable affine parameters, the gain

[01:13:17.300] and the bias.

[01:13:18.540] And this is almost always kept to true.

[01:13:20.780] I'm not actually sure why you would

[01:13:22.660] want to change this to false.

[01:13:26.580] Then track running stats is determining whether or not

[01:13:29.420] batch normalization layer of PyTorch will be doing this.

[01:13:32.860] And one reason you may want to skip the running stats

[01:13:37.680] is because you may want to, for example, estimate them

[01:13:40.060] at the end as a stage 2, like this.

[01:13:42.860] And in that case, you don't want the batch normalization

[01:13:45.060] layer to be doing all this extra compute

[01:13:46.760] that you're not going to use.

[01:13:48.940] And finally, we need to know which device

[01:13:51.260] we're going to run this batch normalization on, a CPU

[01:13:53.900] or a GPU, and what the data type should

[01:13:56.100] be, half precision, single precision, double precision,

[01:13:58.980] and so on.

[01:14:01.100] So that's the batch normalization layer.

[01:14:02.620] Otherwise, they link to the paper.

[01:14:04.020] It's the same formula we've implemented.

[01:14:06.220] And everything is the same, exactly as we've done here.

[01:14:10.860] So that's everything that I wanted to cover for this lecture.

[01:14:13.620] Really, what I wanted to talk about

[01:14:15.020] is the importance of understanding

[01:14:16.580] the activations and the gradients

[01:14:18.200] and their statistics in neural networks.

[01:14:20.420] And this becomes increasingly important,

[01:14:22.140] especially as you make your neural networks bigger, larger,

[01:14:24.540] and deeper.

[01:14:25.980] We looked at the distributions basically at the output layer.

[01:14:28.500] And we saw that if you have too confident mispredictions

[01:14:31.900] because the activations are too messed up at the last layer,

[01:14:35.060] you can end up with these hockey stick losses.

[01:14:37.580] And if you fix this, you get a better loss

[01:14:39.620] at the end of training because your training is not

[01:14:41.900] doing wasteful work.

[01:14:43.740] Then we also saw that we need to control the activations.

[01:14:46.040] We don't want them to squash to zero or explode to infinity.

[01:14:50.780] And because that, you can run into a lot of trouble

[01:14:52.940] with all of these nonlinearities in these neural nets.

[01:14:55.620] And basically, you want everything

[01:14:56.960] to be fairly homogeneous throughout the neural net.

[01:14:58.960] You want roughly Gaussian activations

[01:15:00.540] throughout the neural net.

[01:15:02.580] Then we talked about, OK, if we want roughly Gaussian

[01:15:05.740] activations, how do we scale these weight matrices

[01:15:08.700] and biases during initialization of the neural net

[01:15:11.340] so that we don't get so everything

[01:15:13.860] is as controlled as possible?

[01:15:17.460] So that gave us a large boost in improvement.

[01:15:20.140] And then I talked about how that strategy is not actually

[01:15:24.700] possible for much, much deeper neural nets

[01:15:27.380] because when you have much deeper neural nets with lots

[01:15:30.580] of different types of layers, it becomes really, really hard

[01:15:33.860] to precisely set the weights and the biases in such a way

[01:15:37.420] that the activations are roughly uniform

[01:15:39.820] throughout the neural net.

[01:15:41.340] So then I introduced the notion of a normalization layer.

[01:15:44.540] Now, there are many normalization layers

[01:15:46.180] that people use in practice, batch normalization, layer

[01:15:49.220] normalization, instance normalization,

[01:15:51.460] group normalization.

[01:15:52.700] We haven't covered most of them, but I've

[01:15:54.260] introduced the first one and also the one

[01:15:56.620] that I believe came out first.

[01:15:58.180] And that's called batch normalization.

[01:16:00.740] And we saw how batch normalization works.

[01:16:02.980] This is a layer that you can sprinkle throughout your deep

[01:16:05.300] neural net.

[01:16:06.300] And the basic idea is if you want roughly Gaussian

[01:16:09.300] activations, well, then take your activations

[01:16:11.820] and take the mean and the standard deviation

[01:16:14.620] and center your data.

[01:16:16.620] And you can do that because the centering operation

[01:16:19.620] is differentiable.

[01:16:21.620] And on top of that, we actually had

[01:16:23.060] to add a lot of bells and whistles.

[01:16:25.140] And that gave you a sense of the complexities

[01:16:26.980] of the batch normalization layer because now we're

[01:16:29.500] centering the data.

[01:16:30.260] That's great.

[01:16:30.940] But suddenly, we need the gain and the bias.

[01:16:33.420] And now those are trainable.

[01:16:35.780] And then because we are coupling all the training examples,

[01:16:38.620] now suddenly the question is, how do you do the inference?

[01:16:41.140] Well, to do the inference, we need

[01:16:43.100] to now estimate these mean and standard deviation

[01:16:47.300] once over the entire training set

[01:16:49.860] and then use those at inference.

[01:16:51.820] But then no one likes to do stage two.

[01:16:53.660] So instead, we fold everything into the batch normalization

[01:16:56.700] layer during training and try to estimate these

[01:16:59.420] in the running manner so that everything is a bit simpler.

[01:17:02.620] And that gives us the batch normalization layer.

[01:17:06.220] And as I mentioned, no one likes this layer.

[01:17:09.340] It causes a huge amount of bugs.

[01:17:12.420] And intuitively, it's because it is coupling examples

[01:17:16.900] in the forward pass of the neural net.

[01:17:18.660] And I've shot myself in the foot with this layer

[01:17:23.180] over and over again in my life.

[01:17:25.060] And I don't want you to suffer the same.

[01:17:28.300] So basically, try to avoid it as much as possible.

[01:17:31.820] Some of the other alternatives to these layers

[01:17:33.740] are, for example, group normalization

[01:17:35.180] or layer normalization.

[01:17:36.500] And those have become more common in more recent deep

[01:17:39.940] learning.

[01:17:40.900] But we haven't covered those yet.

[01:17:43.180] But definitely, batch normalization was very

[01:17:45.180] influential at the time when it came out in roughly 2015.

[01:17:48.620] Because it was kind of the first time

[01:17:50.300] that you could train reliably much deeper neural nets.

[01:17:55.180] And fundamentally, the reason for that

[01:17:56.580] is because this layer was very effective at controlling

[01:17:59.740] the statistics of the activations in the neural net.

[01:18:03.060] So that's the story so far.

[01:18:05.180] And that's all I wanted to cover.

[01:18:07.660] And in the future lectures, hopefully, we

[01:18:09.340] can start going into recurrent neural nets.

[01:18:11.420] And recurrent neural nets, as we'll see,

[01:18:14.140] are just very, very deep networks.

[01:18:16.300] Because you unroll the loop when you actually

[01:18:19.820] optimize these neural nets.

[01:18:21.340] And that's where a lot of this analysis

[01:18:25.180] around the activation statistics and all these normalization

[01:18:28.620] layers will become very, very important for good performance.

[01:18:32.620] So we'll see that next time.

[01:18:34.100] Bye.

[01:18:35.220] OK, so I lied.

[01:18:36.340] I would like us to do one more summary here as a bonus.

[01:18:39.100] And I think it's useful as to have

[01:18:40.980] one more summary of everything I've

[01:18:42.460] presented in this lecture.

[01:18:43.820] But also, I would like us to start PyTorchifying our code

[01:18:46.540] a little bit.

[01:18:47.140] So it looks much more like what you would encounter in PyTorch.

[01:18:50.300] So you'll see that I will structure our code

[01:18:52.100] into these modules, like a linear module and a batch

[01:18:57.460] form module.

[01:18:58.580] And I'm putting the code inside these modules

[01:19:01.140] so that we can construct neural networks very

[01:19:02.980] much like we would construct them in PyTorch.

[01:19:04.940] And I will go through this in detail.

[01:19:06.740] So we'll create our neural net.

[01:19:08.700] Then we will do the optimization loop, as we did before.

[01:19:12.460] And then one more thing that I want to do here

[01:19:14.220] is I want to look at the activation statistics

[01:19:16.180] both in the forward pass and in the backward pass.

[01:19:19.380] And then here we have the evaluation and sampling

[01:19:21.460] just like before.

[01:19:22.980] So let me rewind all the way up here and go a little bit

[01:19:25.780] slower.

[01:19:26.940] So here I am creating a linear layer.

[01:19:29.340] You'll notice that torch.nn has lots

[01:19:31.380] of different types of layers.

[01:19:32.780] And one of those layers is the linear layer.

[01:19:35.140] torch.nn.linear takes a number of input features,

[01:19:37.420] output features, whether or not we should have bias,

[01:19:39.980] and then the device that we want to place this layer on,

[01:19:42.580] and the data type.

[01:19:43.940] So I will omit these two.

[01:19:45.740] But otherwise, we have the exact same thing.

[01:19:48.380] We have the fan in, which is the number of inputs,

[01:19:50.700] fan out, the number of outputs, and whether or not

[01:19:53.900] we want to use a bias.

[01:19:55.340] And internally, inside this layer,

[01:19:56.860] there's a weight and a bias, if you'd like it.

[01:19:59.900] It is typical to initialize the weight using, say,

[01:20:04.140] random numbers drawn from a Gaussian.

[01:20:05.980] And then here's the timing initialization

[01:20:08.620] that we discussed already in this lecture.

[01:20:10.660] And that's a good default, and also the default

[01:20:12.860] that I believe PyTorch uses.

[01:20:14.780] And by default, the bias is usually initialized to zeros.

[01:20:18.380] Now, when you call this module, this

[01:20:20.980] will basically calculate w times x plus b, if you have nb.

[01:20:24.900] And then when you also call the parameters on this module,

[01:20:27.460] it will return the tensors that are

[01:20:30.220] the parameters of this layer.

[01:20:32.220] Now, next, we have the batch normalization layer.

[01:20:34.540] So I've written that here.

[01:20:37.040] And this is very similar to PyTorch's nn.batchnormal1d

[01:20:42.020] layer, as shown here.

[01:20:44.540] So I'm kind of taking these three parameters here,

[01:20:47.980] the dimensionality, the epsilon that we'll use in the division,

[01:20:51.500] and the momentum that we will use

[01:20:53.260] in keeping track of these running stats, the running mean

[01:20:55.700] and the running variance.

[01:20:58.140] Now, PyTorch actually takes quite a few more things,

[01:21:00.440] but I'm assuming some of their settings.

[01:21:02.300] So for us, I find it will be true.

[01:21:03.920] That means that we will be using a gamma and beta

[01:21:06.140] after the normalization.

[01:21:08.060] The track running stats will be true.

[01:21:09.620] So we will be keeping track of the running mean

[01:21:11.620] and the running variance in the batch norm.

[01:21:14.660] Our device, by default, is the CPU.

[01:21:17.140] And the data type, by default, is float, float32.

[01:21:22.260] So those are the defaults.

[01:21:23.500] Otherwise, we are taking all the same parameters

[01:21:26.180] in this batch norm layer.

[01:21:27.500] So first, I'm just saving them.

[01:21:30.020] Now, here's something new.

[01:21:31.140] There's a dot training, which by default is true.

[01:21:33.620] And PyTorch NN modules also have this attribute, dot training.

[01:21:37.180] And that's because many modules, and batch norm

[01:21:39.820] is included in that, have a different behavior

[01:21:43.060] whether you are training your neural net

[01:21:45.100] or whether you are running it in an evaluation mode

[01:21:47.620] and calculating your evaluation loss

[01:21:49.600] or using it for inference on some test examples.

[01:21:52.900] And batch norm is an example of this,

[01:21:54.760] because when we are training, we are

[01:21:56.300] going to be using the mean and the variance estimated

[01:21:58.460] from the current batch.

[01:21:59.700] But during inference, we are using the running mean

[01:22:02.300] and running variance.

[01:22:04.060] And so also, if we are training, we

[01:22:06.220] are updating mean and variance.

[01:22:07.780] But if we are testing, then these are not being updated.

[01:22:10.060] They are kept fixed.

[01:22:11.820] And so this flag is necessary and by default true,

[01:22:14.380] just like in PyTorch.

[01:22:16.420] Now, the parameters of batch norm 1D

[01:22:17.980] are the gamma and the beta here.

[01:22:21.820] And then the running mean and running variance

[01:22:23.820] are called buffers in PyTorch nomenclature.

[01:22:27.620] And these buffers are trained using exponential moving

[01:22:31.500] average.

[01:22:32.060] Here explicitly.

[01:22:33.460] And they are not part of the back propagation

[01:22:35.580] and stochastic gradient descent.

[01:22:37.060] So they are not sort of like parameters of this layer.

[01:22:39.900] And that's why when we have parameters here,

[01:22:42.980] we only return gamma and beta.

[01:22:44.660] We do not return the mean and the variance.

[01:22:46.660] This is trained sort of like internally

[01:22:48.780] here every forward pass using exponential moving average.

[01:22:54.620] So that's the initialization.

[01:22:56.940] Now, in a forward pass, if we are training,

[01:22:59.540] then we use the mean and the variance estimated by the batch.

[01:23:03.420] Let me pull up the paper here.

[01:23:05.940] We calculate the mean and the variance.

[01:23:08.900] Now, up above, I was estimating the standard deviation

[01:23:12.260] and keeping track of the standard deviation

[01:23:14.540] here in the running standard deviation

[01:23:16.460] instead of running variance.

[01:23:18.140] But let's follow the paper exactly.

[01:23:20.220] Here they calculate the variance, which

[01:23:22.460] is the standard deviation squared.

[01:23:23.940] And that's what's kept track of in the running variance

[01:23:26.700] instead of a running standard deviation.

[01:23:29.780] But those two would be very, very similar, I believe.

[01:23:33.980] If we are not training, then we use running mean and variance.

[01:23:36.860] We normalize.

[01:23:39.100] And then here, I am calculating the output of this layer.

[01:23:42.140] And I'm also assigning it to an attribute called dot out.

[01:23:45.540] Now, dot out is something that I'm using in our modules here.

[01:23:49.620] This is not what you would find in PyTorch.

[01:23:51.420] We are slightly deviating from it.

[01:23:53.300] I'm creating a dot out because I would

[01:23:54.900] have to very easily maintain all those variables so

[01:23:58.860] that we can create statistics of them and plot them.

[01:24:01.380] But PyTorch and modules will not have a dot out attribute.

[01:24:05.380] And finally, here we are updating the buffers using,

[01:24:07.860] again, as I mentioned, exponential moving average

[01:24:11.260] given the provided momentum.

[01:24:12.980] And importantly, you'll notice that I'm

[01:24:14.620] using the torch.nograd context manager.

[01:24:17.300] And I'm doing this because if we don't use this,

[01:24:19.860] then PyTorch will start building out

[01:24:21.580] an entire computational graph out of these tensors

[01:24:24.620] because it is expecting that we will eventually call dot

[01:24:26.900] backward.

[01:24:27.860] But we are never going to be calling dot backward

[01:24:29.820] on anything that includes running mean and running

[01:24:31.820] variance.

[01:24:32.620] So that's why we need to use this context manager,

[01:24:35.140] so that we are not maintaining and using

[01:24:38.420] all this additional memory.

[01:24:40.460] So this will make it more efficient.

[01:24:42.020] And it's just telling PyTorch that it will be no backward.

[01:24:44.180] We just have a bunch of tensors.

[01:24:45.480] We want to update them.

[01:24:46.420] That's it.

[01:24:47.900] And then we return.

[01:24:50.300] OK, now scrolling down, we have the 10H layer.

[01:24:52.860] This is very, very similar to torch.10H.

[01:24:56.020] And it doesn't do too much.

[01:24:57.820] It just calculates 10H, as you might expect.

[01:25:00.620] So that's torch.10H.

[01:25:02.660] And there's no parameters in this layer.

[01:25:05.300] But because these are layers, it now

[01:25:07.820] becomes very easy to stack them up into basically just a list.

[01:25:13.340] And we can do all the initializations

[01:25:15.660] that we're used to.

[01:25:16.460] So we have the initial embedding matrix.

[01:25:19.580] We have our layers, and we can call them sequentially.

[01:25:22.340] And then, again, with torch.nograd,

[01:25:24.580] there's some initializations here.

[01:25:26.340] So we want to make the output softmax a bit less confident,

[01:25:29.220] like we saw.

[01:25:30.460] And in addition to that, because we are using a six-layer

[01:25:33.360] multilayer perceptron here, so you

[01:25:35.060] see how I'm stacking linear 10H, linear 10H, et cetera,

[01:25:39.220] I'm going to be using the game here.

[01:25:41.020] And I'm going to play with this in a second.

[01:25:42.900] So you'll see how, when we change this,

[01:25:44.940] what happens to the statistics.

[01:25:47.340] Finally, the parameters are basically the embedding matrix

[01:25:49.740] and all the parameters in all the layers.

[01:25:52.500] And notice here, I'm using a double list comprehension,

[01:25:55.140] if you want to call it that.

[01:25:56.260] But for every layer in layers and for every parameter

[01:25:59.460] in each of those layers, we are just stacking up

[01:26:02.120] all those pieces, all those parameters.

[01:26:05.100] Now, in total, we have 46,000 parameters.

[01:26:09.480] And I'm telling PyTorch that all of them require gradient.

[01:26:16.060] Then here, we have everything here

[01:26:18.820] we are actually mostly used to.

[01:26:20.740] We are sampling batch.

[01:26:22.140] We are doing a forward pass.

[01:26:23.580] The forward pass now is just the linear application

[01:26:25.620] of all the layers in order, followed by the cross entropy.

[01:26:29.180] And then in the backward pass, you'll

[01:26:30.640] notice that for every single layer,

[01:26:32.300] I now iterate over all the outputs.

[01:26:34.220] And I'm telling PyTorch to retain the gradient of them.

[01:26:37.540] And then here, we are already used to all the gradients

[01:26:40.740] set to none, do the backward to fill in the gradients,

[01:26:43.940] do an update using stochastic gradient send,

[01:26:46.420] and then track some statistics.

[01:26:48.820] And then I am going to break after a single iteration.

[01:26:52.100] Now, here in this cell, in this diagram,

[01:26:54.220] I'm visualizing the histograms of the forward pass activations.

[01:26:58.760] And I'm specifically doing it at the 10H layers.

[01:27:01.860] So iterating over all the layers,

[01:27:04.340] except for the very last one, which is basically just the soft

[01:27:07.540] max layer, if it is a 10H layer, and I'm using a 10H layer

[01:27:12.820] just because they have a finite output, negative 1 to 1,

[01:27:15.500] and so it's very easy to visualize here.

[01:27:17.420] So you see negative 1 to 1, and it's a finite range,

[01:27:19.820] and easy to work with.

[01:27:21.780] I take the out tensor from that layer into t.

[01:27:25.580] And then I'm calculating the mean, the standard deviation,

[01:27:28.180] and the percent saturation of t.

[01:27:30.700] And the way I define the percent saturation

[01:27:32.340] is that t dot absolute value is greater than 0.97.

[01:27:35.500] So that means we are here at the tails of the 10H.

[01:27:38.700] And remember that when we are in the tails of the 10H,

[01:27:40.980] that will actually stop gradients.

[01:27:42.820] So we don't want this to be too high.

[01:27:45.620] Now, here I'm calling torch dot histogram,

[01:27:49.020] and then I am plotting this histogram.

[01:27:50.940] So basically what this is doing is

[01:27:52.360] that every different type of layer,

[01:27:53.940] and they all have a different color,

[01:27:55.400] we are looking at how many values in these tensors

[01:27:59.700] take on any of the values below on this axis here.

[01:28:04.220] So the first layer is fairly saturated here at 20%.

[01:28:08.020] So you can see that it's got tails here.

[01:28:10.520] But then everything sort of stabilizes.

[01:28:12.560] And if we had more layers here, it

[01:28:13.940] would actually just stabilize at around the standard deviation

[01:28:16.440] of about 0.65, and the saturation would be roughly 5%.

[01:28:20.740] And the reason that this stabilizes and gives us

[01:28:22.820] a nice distribution here is because gain

[01:28:25.340] is set to 5 over 3.

[01:28:27.780] Now, here, this gain, you see that by default, we

[01:28:32.100] initialize with 1 over square root of fan in.

[01:28:35.380] But then here, during initialization,

[01:28:36.940] I come in and I iterate over all the layers.

[01:28:38.800] And if it's a linear layer, I boost that by the gain.

[01:28:42.440] Now, we saw that 1.

[01:28:44.620] So basically, if we just do not use a gain, then what happens?

[01:28:48.860] If I redraw this, you will see that the standard deviation

[01:28:53.060] is shrinking, and the saturation is coming to 0.

[01:28:56.800] And basically, what's happening is

[01:28:58.140] the first layer is pretty decent,

[01:29:00.940] but then further layers are just kind of like shrinking down

[01:29:03.900] to 0.

[01:29:05.000] And it's happening slowly, but it's shrinking to 0.

[01:29:07.700] And the reason for that is when you just

[01:29:10.140] have a sandwich of linear layers alone,

[01:29:14.820] then initializing our weights in this manner we saw previously

[01:29:19.140] would have conserved the standard deviation of 1.

[01:29:22.140] But because we have this interspersed tanh layers

[01:29:24.940] in there, these tanh layers are squashing functions.

[01:29:29.580] And so they take your distribution,

[01:29:31.300] and they slightly squash it.

[01:29:32.940] And so some gain is necessary to keep expanding it

[01:29:37.180] to fight the squashing.

[01:29:39.980] So it just turns out that 5 over 3 is a good value.

[01:29:43.620] So if we have something too small, like 1,

[01:29:45.740] we saw that things will come towards 0.

[01:29:49.060] But if it's something too high, let's do 2.

[01:29:52.460] Then here we see that, well, let me

[01:29:57.460] do something a bit more extreme so it's a bit more visible.

[01:30:00.460] Let's try 3.

[01:30:02.220] OK, so we see here that the saturations are

[01:30:04.100] trying to be way too large.

[01:30:07.020] So 3 would create way too saturated activations.

[01:30:10.860] So 5 over 3 is a good setting for a sandwich of linear layers

[01:30:16.180] with tanh activations.

[01:30:17.820] And it roughly stabilizes the standard deviation

[01:30:20.420] at a reasonable point.

[01:30:22.580] Now, honestly, I have no idea where 5 over 3

[01:30:24.900] came from in PyTorch when we were looking

[01:30:27.860] at the counting initialization.

[01:30:30.060] I see empirically that it stabilizes

[01:30:31.980] this sandwich of linear and tanh,

[01:30:34.340] and that the saturation is in a good range.

[01:30:36.700] But I don't actually know if this came out of some math

[01:30:38.940] formula.

[01:30:39.580] I tried searching briefly for where this comes from,

[01:30:42.780] but I wasn't able to find anything.

[01:30:44.460] But certainly, we see that empirically,

[01:30:46.100] these are very nice ranges.

[01:30:47.420] Our saturation is roughly 5%, which is a pretty good number.

[01:30:50.860] And this is a good setting of the gain in this context.

[01:30:55.140] Similarly, we can do the exact same thing with the gradients.

[01:30:58.260] So here is a very same loop if it's a tanh.

[01:31:01.420] But instead of taking the layer that out, I'm taking the grad.

[01:31:04.420] And then I'm also showing the mean and the standard deviation.

[01:31:07.140] And I'm plotting the histogram of these values.

[01:31:09.740] And so you'll see that the gradient distribution

[01:31:11.740] is fairly reasonable.

[01:31:13.100] And in particular, what we're looking for

[01:31:14.860] is that all the different layers in this sandwich

[01:31:17.740] has roughly the same gradient.

[01:31:19.580] Things are not shrinking or exploding.

[01:31:21.940] So we can, for example, come here,

[01:31:23.980] and we can take a look at what happens if this gain was way

[01:31:26.420] too small.

[01:31:27.540] So this was 0.5.

[01:31:30.580] Then you see, first of all, the activations

[01:31:33.140] are shrinking to 0.

[01:31:34.300] But also, the gradients are doing something weird.

[01:31:36.420] The gradients started out here, and then now they're

[01:31:38.820] like expanding out.

[01:31:41.180] And similarly, if we, for example,

[01:31:43.580] have a too high of a gain, so like 3,

[01:31:46.460] then we see that also the gradients have some asymmetry

[01:31:49.100] going on, where as you go into deeper and deeper layers,

[01:31:52.100] the activations are also changing.

[01:31:54.140] And so that's not what we want.

[01:31:55.540] And in this case, we saw that without the use of batch norm,

[01:31:58.380] as we are going through right now,

[01:32:00.340] we have to very carefully set those gains

[01:32:03.220] to get nice activations in both the forward pass

[01:32:06.420] and the backward pass.

[01:32:07.540] Now, before we move on to batch normalization,

[01:32:10.140] I would also like to take a look at what happens when we have no

[01:32:12.540] 10H units here.

[01:32:13.940] So erasing all the 10H nonlinearities,

[01:32:16.780] but keeping the gain at 5 over 3,

[01:32:19.380] we now have just a giant linear sandwich.

[01:32:22.100] So let's see what happens to the activations.

[01:32:24.340] As we saw before, the correct gain here

[01:32:26.700] is 1, that is the standard deviation preserving gain.

[01:32:29.660] So 1.667 is too high.

[01:32:33.660] And so what's going to happen now is the following.

[01:32:37.340] I have to change this to be linear,

[01:32:40.380] because there's no more 10H layers.

[01:32:43.020] And let me change this to linear as well.

[01:32:46.180] So what we're seeing is the activations started out

[01:32:50.300] on the blue and have, by layer 4, become very diffuse.

[01:32:55.220] So what's happening to the activations is this.

[01:32:57.780] And with the gradients on the top layer,

[01:33:01.780] the gradient statistics are the purple,

[01:33:04.380] and then they diminish as you go down deeper in the layers.

[01:33:07.700] And so basically, you have an asymmetry in the neural net.

[01:33:10.500] And you might imagine that if you

[01:33:11.820] have very deep neural networks, say like 50 layers

[01:33:13.940] or something like that, this is not a good place to be.

[01:33:18.820] So that's why, before batch normalization,

[01:33:21.260] this was incredibly tricky to set.

[01:33:24.260] In particular, if this is too large of a gain, this happens.

[01:33:27.260] And if it's too little of a gain, then this happens.

[01:33:31.460] So the opposite of that basically happens.

[01:33:33.460] Here we have a shrinking and a diffusion,

[01:33:39.660] depending on which direction you look at it from.

[01:33:42.380] And so certainly, this is not what you want.

[01:33:44.180] And in this case, the correct setting of the gain

[01:33:46.180] is exactly 1, just like we're doing at initialization.

[01:33:50.220] And then we see that the statistics

[01:33:53.060] for the forward and the backward paths are well-behaved.

[01:33:56.260] And so the reason I want to show you this

[01:33:58.780] is that basically, getting neural nets to train

[01:34:02.580] before these normalization layers

[01:34:04.260] and before the use of advanced optimizers like Adam,

[01:34:06.980] which we still have to cover, and residual connections

[01:34:09.380] and so on, training neural nets basically look like this.

[01:34:13.380] It's like a total balancing act.

[01:34:14.980] You have to make sure that everything is precisely

[01:34:17.460] orchestrated.

[01:34:18.220] And you have to care about the activations and the gradients

[01:34:20.140] and their statistics.

[01:34:21.300] And then maybe you can train something.

[01:34:23.220] But it was basically impossible to train very deep networks.

[01:34:25.720] And this is fundamentally the reason for that.

[01:34:27.980] You'd have to be very, very careful

[01:34:29.500] with your initialization.

[01:34:32.220] The other point here is you might be asking yourself,

[01:34:35.300] by the way, I'm not sure if I covered this,

[01:34:37.100] why do we need these 10H layers at all?

[01:34:40.740] Why do we include them and then have to worry about the gain?

[01:34:43.420] And the reason for that, of course,

[01:34:45.020] is that if you just have a stack of linear layers,

[01:34:47.740] then certainly, we're getting very easily nice activations

[01:34:51.100] and so on.

[01:34:52.220] But this is just a massive linear sandwich.

[01:34:54.540] And it turns out that it collapses

[01:34:55.900] to a single linear layer in terms

[01:34:57.660] of its representation power.

[01:34:59.780] So if you were to plot the output

[01:35:01.660] as a function of the input, you're

[01:35:02.980] just getting a linear function.

[01:35:04.340] No matter how many linear layers you stack up,

[01:35:06.500] you still just end up with a linear transformation.

[01:35:09.020] All the wx plus bs just collapse into a large wx plus b

[01:35:13.940] with slightly different ws and slightly different b.

[01:35:17.540] But interestingly, even though the forward pass collapses

[01:35:19.780] to just a linear layer, because of back propagation

[01:35:22.740] and the dynamics of the backward pass,

[01:35:26.060] the optimization actually is not identical.

[01:35:28.640] You actually end up with all kinds

[01:35:30.300] of interesting dynamics in the backward pass

[01:35:34.700] because of the way the chain rule is calculating it.

[01:35:37.900] And so optimizing a linear layer by itself

[01:35:40.900] and optimizing a sandwich of 10 linear layers, in both cases,

[01:35:44.220] those are just a linear transformation

[01:35:45.760] in the forward pass, but the training dynamics

[01:35:47.660] will be different.

[01:35:48.580] And there's entire papers that analyze, in fact,

[01:35:51.300] infinitely linear layers and so on.

[01:35:54.540] And so there's a lot of things, too,

[01:35:56.220] that you can play with there.

[01:35:58.700] But basically, the 10H nonlinearities

[01:36:00.260] allow us to turn this sandwich from just a linear function

[01:36:09.140] into a neural network that can, in principle,

[01:36:13.060] approximate any arbitrary function.

[01:36:15.540] OK, so now I've reset the code to use the linear 10H

[01:36:18.740] sandwich, like before.

[01:36:20.540] And I reset everything so the gain is 5 over 3.

[01:36:23.940] We can run a single step of optimization,

[01:36:26.340] and we can look at the activation statistics

[01:36:28.220] of the forward pass and the backward pass.

[01:36:30.500] But I've added one more plot here

[01:36:31.900] that I think is really important to look

[01:36:33.580] at when you're training your neural nets and to consider.

[01:36:36.220] And ultimately, what we're doing is

[01:36:37.820] we're updating the parameters of the neural net.

[01:36:40.140] So we care about the parameters and their values

[01:36:42.940] and their gradients.

[01:36:44.180] So here, what I'm doing is I'm actually

[01:36:45.860] iterating over all the parameters available,

[01:36:48.060] and then I'm only restricting it to the two-dimensional

[01:36:51.860] parameters, which are basically the weights of these linear

[01:36:54.020] layers.

[01:36:54.820] And I'm skipping the biases, and I'm

[01:36:56.580] skipping the gammas and the betas and the bastroom

[01:37:00.300] just for simplicity.

[01:37:02.420] But you can also take a look at those as well.

[01:37:04.220] But what's happening with the weights

[01:37:05.620] is instructive by itself.

[01:37:09.100] So here we have all the different weights,

[01:37:11.060] their shapes.

[01:37:12.940] So this is the embedding layer, the first linear layer,

[01:37:15.620] all the way to the very last linear layer.

[01:37:17.660] And then we have the mean, the standard deviation

[01:37:19.700] of all these parameters.

[01:37:22.060] The histogram, and you can see that it actually

[01:37:23.900] doesn't look that amazing.

[01:37:25.060] So there's some trouble in paradise.

[01:37:26.700] Even though these gradients look OK,

[01:37:28.860] there's something weird going on here.

[01:37:30.540] I'll get to that in a second.

[01:37:32.220] And the last thing here is the gradient-to-data ratio.

[01:37:35.820] So sometimes I like to visualize this as well

[01:37:37.660] because what this gives you a sense of

[01:37:39.780] is what is the scale of the gradient compared

[01:37:42.820] to the scale of the actual values?

[01:37:45.460] And this is important because we're

[01:37:46.880] going to end up taking a step update that

[01:37:50.860] is the learning rate times the gradient onto the data.

[01:37:54.220] And so if the gradient has too large of a magnitude,

[01:37:56.780] if the numbers in there are too large

[01:37:58.340] compared to the numbers in data, then you'd be in trouble.

[01:38:01.860] But in this case, the gradient-to-data

[01:38:03.660] is our low numbers.

[01:38:05.500] So the values inside grad are 1,000 times

[01:38:08.580] smaller than the values inside data in these weights, most

[01:38:12.660] of them.

[01:38:13.980] Now, notably, that is not true about the last layer.

[01:38:17.220] And so the last layer actually here, the output layer,

[01:38:19.460] is a bit of a troublemaker in the way

[01:38:21.140] that this is currently arranged.

[01:38:22.620] Because you can see that the last layer here in pink

[01:38:28.620] takes on values that are much larger than some

[01:38:31.020] of the values inside the neural net.

[01:38:35.940] So the standard deviations are roughly 1 and negative 3

[01:38:38.240] throughout, except for the last layer, which actually has

[01:38:42.380] roughly 1 and negative 2 standard deviation

[01:38:44.780] of gradients.

[01:38:45.940] And so the gradients on the last layer

[01:38:47.620] are currently about 10 times greater

[01:38:52.420] than all the other weights inside the neural net.

[01:38:56.020] And so that's problematic, because in the simple

[01:38:58.500] stochastic gradedness setup, you would

[01:39:00.660] be training this last layer about 10 times faster

[01:39:03.860] than you would be training the other layers

[01:39:05.620] at initialization.

[01:39:07.260] Now, this actually kind of fixes itself a little bit

[01:39:09.940] if you train for a bit longer.

[01:39:11.180] So for example, if I greater than 1,000,

[01:39:14.140] only then do a break.

[01:39:16.260] Let me reinitialize.

[01:39:17.460] And then let me do it 1,000 steps.

[01:39:20.060] And after 1,000 steps, we can look at the forward pass.

[01:39:24.380] So you see how the neurons are saturating a bit.

[01:39:27.820] And we can also look at the backward pass.

[01:39:30.020] But otherwise, they look good.

[01:39:31.260] They're about equal.

[01:39:32.460] And there's no shrinking to 0 or exploding to infinities.

[01:39:36.220] And you can see that here in the weights,

[01:39:38.720] things are also stabilizing a little bit.

[01:39:40.380] So the tails of the last pink layer

[01:39:42.900] are actually coming in during the optimization.

[01:39:46.380] But certainly, this is a little bit troubling,

[01:39:48.900] especially if you are using a very simple update rule,

[01:39:51.140] like stochastic gradient descent,

[01:39:52.700] instead of a modern optimizer like Atom.

[01:39:54.900] Now, I'd like to show you one more plot that I usually

[01:39:57.220] look at when I train neural networks.

[01:39:59.080] And basically, the gradient to data ratio

[01:40:01.820] is not actually that informative,

[01:40:03.380] because what matters at the end is not the gradient to data

[01:40:05.840] ratio, but the update to the data ratio,

[01:40:08.260] because that is the amount by which we will actually

[01:40:10.340] change the data in these tensors.

[01:40:13.020] So coming up here, what I'd like to do

[01:40:15.020] is I'd like to introduce a new update to data ratio.

[01:40:19.340] It's going to be list, and we're going to build it out

[01:40:21.380] every single iteration.

[01:40:23.260] And here, I'd like to keep track of basically the ratio

[01:40:27.740] every single iteration.

[01:40:30.100] So without any gradients, I'm comparing the update,

[01:40:35.060] which is learning rate times the gradient.

[01:40:38.540] That is the update that we're going to apply

[01:40:40.340] to every parameter.

[01:40:42.580] So see, I'm iterating over all the parameters.

[01:40:44.580] And then I'm taking the standard deviation of the update

[01:40:47.060] we're going to apply and divide it

[01:40:49.820] by the actual content, the data of that parameter,

[01:40:54.140] and its standard deviation.

[01:40:56.140] So this is the ratio of basically how great

[01:40:58.780] are the updates to the values in these tensors.

[01:41:01.920] Then we're going to take a log of it,

[01:41:03.540] and actually, I'd like to take a log 10,

[01:41:07.540] just so it's a nicer visualization.

[01:41:10.020] So we're going to be basically looking

[01:41:11.420] at the exponents of this division here,

[01:41:16.860] and then dot item to pop out the float.

[01:41:18.900] And we're going to be keeping track of this

[01:41:20.700] for all the parameters and adding it to this UD tensor.

[01:41:24.340] So now let me re-initialize and run 1,000 iterations.

[01:41:27.660] We can look at the activations, the gradients,

[01:41:32.020] and the parameter gradients as we did before.

[01:41:34.260] But now I have one more plot here to introduce.

[01:41:37.620] What's happening here is where every interval load parameters,

[01:41:40.580] and I'm constraining it again, like I did here,

[01:41:42.580] to just the weights.

[01:41:44.700] So the number of dimensions in these sensors is 2.

[01:41:47.900] And then I'm basically plotting all of these update ratios

[01:41:52.220] over time.

[01:41:54.620] So when I plot this, I plot those ratios,

[01:41:57.580] and you can see that they evolve over time during initialization

[01:42:00.340] to take on certain values.

[01:42:02.060] And then these updates start stabilizing

[01:42:04.140] usually during training.

[01:42:05.500] Then the other thing that I'm plotting here

[01:42:07.300] is I'm plotting here an approximate value that

[01:42:09.860] is a rough guide for what it roughly should be.

[01:42:12.940] And it should be roughly 1 in negative 3.

[01:42:15.540] And so that means that basically there's

[01:42:17.540] some values in this tensor, and they take on certain values.

[01:42:21.900] And the updates to them at every single iteration

[01:42:24.300] are no more than roughly 1,000 of the actual magnitude

[01:42:29.060] in those tensors.

[01:42:30.980] If this was much larger, like for example,

[01:42:33.580] if the log of this was, like, say, negative 1,

[01:42:37.740] this is actually updating those values quite a lot.

[01:42:40.100] They're undergoing a lot of change.

[01:42:42.340] But the reason that the final layer here is an outlier

[01:42:46.660] is because this layer was artificially shrunk down

[01:42:50.180] to keep the softmax unconfident.

[01:42:54.500] So here, you see how we multiplied the weight by 0.1

[01:42:59.380] in the initialization to make the last layer

[01:43:01.940] prediction less confident.

[01:43:04.140] That artificially made the values inside that tensor way

[01:43:08.460] too low, and that's why we're getting temporarily

[01:43:10.780] a very high ratio.

[01:43:12.140] But you see that that stabilizes over time

[01:43:14.220] once that weight starts to learn.

[01:43:17.980] But basically, I like to look at the evolution of this update

[01:43:20.940] ratio for all my parameters, usually.

[01:43:23.260] And I like to make sure that it's not too much

[01:43:26.300] above 1 in negative 3, roughly.

[01:43:29.620] So around negative 3 on this log plot.

[01:43:33.060] If it's below negative 3, usually that

[01:43:34.620] means that the parameters are not training fast enough.

[01:43:37.500] So if our learning rate was very low, let's do that experiment.

[01:43:41.740] Let's initialize.

[01:43:42.900] And then let's actually do a learning rate

[01:43:44.580] of, say, 1 in negative 3 here, so 0.001.

[01:43:49.660] If your learning rate is way too low,

[01:43:53.860] this plot will typically reveal it.

[01:43:56.420] So you see how all of these updates are way too small.

[01:44:00.460] So the size of the update is basically 10,000 times

[01:44:06.420] in magnitude to the size of the numbers

[01:44:09.140] in that tensor in the first place.

[01:44:10.780] So this is a symptom of training way too slow.

[01:44:14.500] So this is another way to sometimes set the learning

[01:44:16.660] rate and to get a sense of what that learning rate should be.

[01:44:19.220] And ultimately, this is something

[01:44:20.220] that you would keep track of.

[01:44:21.500] If anything, the learning rate here

[01:44:23.460] is a little bit on the higher side

[01:44:25.940] because you see that we're above the black line of negative 3.

[01:44:30.420] We're somewhere around negative 2.5.

[01:44:32.420] It's like, OK.

[01:44:34.260] But everything is somewhat stabilizing.

[01:44:36.180] And so this looks like a pretty decent setting

[01:44:37.940] of learning rates and so on.

[01:44:40.540] But this is something to look at.

[01:44:41.940] And when things are miscalibrated,

[01:44:43.260] you will see very quickly.

[01:44:44.980] So for example, everything looks pretty well-behaved.

[01:44:48.340] But just as a comparison, when things are not properly

[01:44:51.460] calibrated, what does that look like?

[01:44:53.220] Let me come up here.

[01:44:54.580] And let's say that, for example, what do we do?

[01:44:58.460] Let's say that we forgot to apply this fan-in normalization.

[01:45:02.300] So the weights inside the linear layers

[01:45:03.820] are just a sample from a Gaussian in all the stages.

[01:45:07.660] What happens to our, how do we notice that something's off?

[01:45:11.100] Well, the activation plot will tell you, whoa,

[01:45:13.420] your neurons are way too saturated.

[01:45:15.220] The gradients are going to be all messed up.

[01:45:18.340] The histogram for these weights are going to be all messed up

[01:45:21.100] as well.

[01:45:22.020] And there's a lot of asymmetry.

[01:45:23.780] And then if we look here, I suspect

[01:45:25.340] it's all going to be also pretty messed up.

[01:45:27.420] So you see there's a lot of discrepancy

[01:45:30.700] in how fast these layers are learning.

[01:45:33.220] And some of them are learning way too fast.

[01:45:35.260] So negative 1, negative 1.5, those

[01:45:38.300] are very large numbers in terms of this ratio.

[01:45:41.020] Again, you should be somewhere around the range of 1.5

[01:45:43.620] or, again, you should be somewhere around negative 3

[01:45:45.900] and not much more above that.

[01:45:48.660] So this is how miscalibrations of your neural nets

[01:45:51.540] are going to manifest.

[01:45:52.940] And these kinds of plots here are

[01:45:54.460] a good way of sort of bringing those miscalibrations sort

[01:45:59.660] of to your attention and so you can address them.

[01:46:04.460] OK, so far we've seen that when we have this linear 10H

[01:46:07.220] sandwich, we can actually precisely calibrate the gains

[01:46:10.260] and make the activations, the gradients, and the parameters,

[01:46:13.380] and the updates all look pretty decent.

[01:46:15.860] But it definitely feels a little bit like balancing

[01:46:19.100] of a pencil on your finger.

[01:46:21.260] And that's because this gain has to be very precisely

[01:46:24.540] calibrated.

[01:46:25.580] So now let's introduce batch normalization layers

[01:46:27.620] into the mix.

[01:46:30.020] And let's see how that helps fix the problem.

[01:46:34.060] So here, I'm going to take the batch from 1D class.

[01:46:38.420] And I'm going to start placing it inside.

[01:46:41.140] And as I mentioned before, the standard typical place

[01:46:44.260] you would place it is between the linear layer,

[01:46:47.060] so right after it, but before the nonlinearity.

[01:46:49.260] But people have definitely played with that.

[01:46:51.260] And in fact, you can get very similar results

[01:46:54.180] even if you place it after the nonlinearity.

[01:46:57.320] And the other thing that I wanted to mention

[01:46:59.240] is it's totally fine to also place it

[01:47:00.820] at the end after the last linear layer

[01:47:03.060] and before the loss function.

[01:47:04.900] So this is potentially fine as well.

[01:47:08.820] And in this case, this would be output, would be vocab size.

[01:47:14.180] Now, because the last layer is batch room,

[01:47:16.980] we would not be changing the weight

[01:47:18.660] to make the softmax less confident.

[01:47:20.620] We'd be changing the gamma.

[01:47:23.020] Because gamma, remember, in the batch room

[01:47:25.500] is the variable that multiplicatively

[01:47:27.660] interacts with the output of that normalization.

[01:47:32.660] So we can initialize this sandwich now.

[01:47:35.780] We can train.

[01:47:37.140] And we can see that the activations are going

[01:47:39.620] to, of course, look very good.

[01:47:41.660] And they are going to necessarily look good.

[01:47:43.940] Because now before every single 10H layer,

[01:47:46.780] there is a normalization in the batch room.

[01:47:49.260] So this is, unsurprisingly, all looks pretty good.

[01:47:52.980] It's going to be standard deviation of roughly 0.65%,

[01:47:55.740] 2%, and roughly equal standard deviation

[01:47:58.100] throughout the entire layers.

[01:47:59.740] So everything looks very homogeneous.

[01:48:02.740] The gradients look good.

[01:48:04.700] The weights look good.

[01:48:06.860] And their distributions.

[01:48:09.260] And then the updates also look pretty reasonable.

[01:48:14.180] We are going above negative 3 a little bit, but not by too much.

[01:48:18.020] So all the parameters are training at roughly the same

[01:48:20.820] rate here.

[01:48:24.740] But now what we've gained is we are

[01:48:26.820] going to be slightly less brittle with respect

[01:48:32.460] to the gain of these.

[01:48:34.220] So for example, I can make the gain be, say, 0.2 here,

[01:48:39.220] which is much slower than what we had with the 10H.

[01:48:42.940] But as we'll see, the activations

[01:48:44.380] will actually be exactly unaffected.

[01:48:46.860] And that's because of, again, this explicit normalization.

[01:48:49.700] The gradients are going to look OK.

[01:48:51.460] The weight gradients are going to look OK.

[01:48:53.900] But actually, the updates will change.

[01:48:56.980] And so even though the forward and backward paths,

[01:48:59.940] to a very large extent, look OK because of the backward paths

[01:49:02.740] of the batch norm and how the scale of the incoming

[01:49:05.340] activations interacts in the batch norm

[01:49:07.980] and its backward paths, this is actually

[01:49:10.820] changing the scale of the updates on these parameters.

[01:49:16.300] So the gradients of these weights are affected.

[01:49:19.620] So we still don't get a completely free pass

[01:49:21.940] to pass in arbitrary weights here.

[01:49:24.980] But everything else is significantly more robust

[01:49:28.660] in terms of the forward, backward, and the weight gradients.

[01:49:32.980] It's just that you may have to retune your learning rate

[01:49:35.460] if you are changing sufficiently the scale of the activations

[01:49:39.420] that are coming into the batch norms.

[01:49:41.460] So here, for example, we changed the gains

[01:49:45.100] of these linear layers to be greater.

[01:49:46.900] And we're seeing that the updates are coming out lower

[01:49:49.140] as a result.

[01:49:51.740] And then finally, we can also, if we are using batch norms,

[01:49:54.580] we don't actually need to necessarily,

[01:49:56.700] let me reset this to 1 so there's no gain.

[01:49:59.220] We don't necessarily even have to normalize by fan-in sometimes.

[01:50:03.540] So if I take out the fan-in, so these are just now

[01:50:06.420] random Gaussian, we'll see that because of batch norm,

[01:50:09.420] this will actually be relatively well-behaved.

[01:50:11.780] So this look, of course, in the forward pass look good.

[01:50:17.580] The gradients look good.

[01:50:19.900] The backward weight updates look OK.

[01:50:23.780] A little bit of fat tails on some of the layers.

[01:50:26.660] And this looks OK as well.

[01:50:29.300] But as you can see, we're significantly below negative 3.

[01:50:33.540] So we'd have to bump up the learning rate

[01:50:35.300] of this batch norm so that we are training more properly.

[01:50:39.100] And in particular, looking at this,

[01:50:40.700] roughly looks like we have to 10x the learning rate

[01:50:43.220] to get to about 1e negative 3.

[01:50:46.740] So we'd come here, and we would change this to be update of 1.0.

[01:50:51.420] And if I reinitialize, then we'll

[01:50:59.380] see that everything still, of course, looks good.

[01:51:02.420] And now we are roughly here.

[01:51:04.220] And we expect this to be an OK training run.

[01:51:07.180] So long story short, we are significantly more robust

[01:51:09.500] to the gain of these linear layers,

[01:51:11.740] whether or not we have to apply the fan-in.

[01:51:14.060] And then we can change the gain, but we actually

[01:51:16.860] do have to worry a little bit about the update scales

[01:51:20.580] and making sure that the learning rate is properly

[01:51:22.900] calibrated here.

[01:51:24.100] But the activations of the forward, backward pass

[01:51:26.740] and the updates are looking significantly more well

[01:51:29.740] behaved, except for the global scale that is potentially

[01:51:32.660] being adjusted here.

[01:51:34.700] OK, so now let me summarize.

[01:51:36.580] There are three things I was hoping

[01:51:37.980] to achieve with this section.

[01:51:39.460] Number one, I wanted to introduce you

[01:51:41.100] to batch normalization, which is one

[01:51:42.620] of the first modern innovations that we're

[01:51:44.540] looking into that helped stabilize

[01:51:46.820] very deep neural networks and their training.

[01:51:49.700] And I hope you understand how the batch normalization works

[01:51:52.620] and how it would be used in a neural network.

[01:51:56.140] Number two, I was hoping to PyTorchify some of our code

[01:51:59.340] and wrap it up into these modules,

[01:52:01.860] so like linear, batch norm 1D, 10H, et cetera.

[01:52:04.700] These are layers or modules.

[01:52:06.820] And they can be stacked up into neural nets

[01:52:09.060] like Lego building blocks.

[01:52:10.980] And these layers actually exist in PyTorch.

[01:52:14.980] And if you import torch nn, then you can actually,

[01:52:17.940] the way I've constructed it, you can

[01:52:19.580] simply just use PyTorch by prepending nn.

[01:52:22.900] to all these different layers.

[01:52:25.100] And actually, everything will just

[01:52:27.100] work because the API that I've developed here

[01:52:29.780] is identical to the API that PyTorch uses.

[01:52:32.660] And the implementation also is basically,

[01:52:34.900] as far as I'm aware, identical to the one in PyTorch.

[01:52:38.300] And number three, I tried to introduce you

[01:52:39.980] to the diagnostic tools that you would

[01:52:41.500] use to understand whether your neural network is

[01:52:44.180] in a good state dynamically.

[01:52:46.260] So we are looking at the statistics and histograms

[01:52:48.940] and activation of the forward pass activations,

[01:52:52.260] the backward pass gradients.

[01:52:54.100] And then also, we're looking at the weights

[01:52:55.740] that are going to be updated as part of stochastic gradient

[01:52:58.260] ascent.

[01:52:58.900] And we're looking at their means, standard deviations,

[01:53:01.100] and also the ratio of gradients to data,

[01:53:04.540] or even better, the updates to data.

[01:53:07.780] And we saw that typically, we don't actually

[01:53:09.860] look at it as a single snapshot frozen in time

[01:53:12.060] at some particular iteration.

[01:53:13.740] Typically, people look at this as over time,

[01:53:16.420] just like I've done here.

[01:53:17.780] And they look at these update-to-data ratios,

[01:53:19.700] and they make sure everything looks OK.

[01:53:21.620] And in particular, I said that running negative 3,

[01:53:25.380] or basically negative 3 on the log scale,

[01:53:27.580] is a good rough heuristic for what

[01:53:30.220] you want this ratio to be.

[01:53:31.860] And if it's way too high, then probably the learning rate

[01:53:34.420] or the updates are a little too big.

[01:53:36.220] And if it's way too small, then the learning rate

[01:53:38.220] is probably too small.

[01:53:39.520] So that's just some of the things

[01:53:40.860] that you may want to play with when

[01:53:42.520] you try to get your neural network to work very well.

[01:53:46.460] Now, there's a number of things I did not try to achieve.

[01:53:49.180] I did not try to beat our previous performance,

[01:53:51.300] as an example, by introducing the BashNorm layer.

[01:53:54.020] Actually, I did try.

[01:53:55.580] And I used the learning rate finding mechanism

[01:53:58.700] that I've described before.

[01:53:59.860] I tried to train the BashNorm layer, BashNorm neural net.

[01:54:03.140] And I actually ended up with results

[01:54:05.460] that are very, very similar to what we've obtained before.

[01:54:08.220] And that's because our performance now

[01:54:10.060] is not bottlenecked by the optimization, which

[01:54:13.260] is what BashNorm is helping with.

[01:54:15.140] The performance at this stage is bottlenecked by, what I

[01:54:17.660] suspect, is the context length of our context.

[01:54:21.820] So currently, we are taking three characters

[01:54:23.620] to predict the fourth one.

[01:54:24.780] And I think we need to go beyond that.

[01:54:26.180] And we need to look at more powerful architectures,

[01:54:28.740] like recurrent neural networks and transformers,

[01:54:30.940] in order to further push the log probabilities that we're

[01:54:34.020] achieving on this data set.

[01:54:36.500] And I also did not try to have a full explanation of all

[01:54:40.940] of these activations, the gradients,

[01:54:42.460] and the backward pass, and the statistics

[01:54:44.220] of all these gradients.

[01:54:45.420] And so you may have found some of the parts here unintuitive.

[01:54:47.940] And maybe you were slightly confused about, OK,

[01:54:49.940] if I change the game here, how come that we need

[01:54:52.940] a different learning rate?

[01:54:54.140] And I didn't go into the full detail

[01:54:55.540] because you'd have to actually look

[01:54:56.660] at the backward pass of all these different layers

[01:54:58.740] and get an intuitive understanding

[01:55:00.100] of how that works.

[01:55:01.380] And I did not go into that in this lecture.

[01:55:04.020] The purpose really was just to introduce you

[01:55:05.860] to the diagnostic tools and what they look like.

[01:55:08.260] But there's still a lot of work remaining on the intuitive

[01:55:10.500] level to understand the initialization,

[01:55:12.700] the backward pass, and how all of that interacts.

[01:55:15.780] But you shouldn't feel too bad because, honestly, we

[01:55:18.340] are getting to the cutting edge of where the field is.

[01:55:22.900] We certainly haven't, I would say, solved initialization.

[01:55:25.580] And we haven't solved backpropagation.

[01:55:28.140] And these are still very much an active area of research.

[01:55:30.740] People are still trying to figure out

[01:55:31.940] what is the best way to initialize these networks,

[01:55:33.900] what is the best update rule to use, and so on.

[01:55:37.460] So none of this is really solved.

[01:55:38.820] And we don't really have all the answers

[01:55:40.380] to all these cases.

[01:55:44.060] But at least we're making progress.

[01:55:46.220] And at least we have some tools to tell us

[01:55:48.340] whether or not things are on the right track for now.

[01:55:51.820] So I think we've made positive progress in this lecture.

[01:55:55.020] And I hope you enjoyed that.

[01:55:56.140] And I will see you next time.

