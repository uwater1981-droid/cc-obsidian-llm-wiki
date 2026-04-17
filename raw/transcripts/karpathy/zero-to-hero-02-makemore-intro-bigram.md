---
title: "The spelled-out intro to language modeling: building makemore"
video_id: PaCmpygFfXo
url: "https://www.youtube.com/watch?v=PaCmpygFfXo"
author: Andrej Karpathy
slug: zero-to-hero-02-makemore-intro-bigram
fetched_at: "2026-04-17T20:58:28+08:00"
type: youtube-transcript-whisper
transcript_source: "https://github.com/averkij/karcaps (002-large.html)"
segments: 2343
---

# The spelled-out intro to language modeling: building makemore

> Video: https://www.youtube.com/watch?v=PaCmpygFfXo
> Transcript: averkij/karcaps (Whisper-large) `002-large.html`

[00:00:00.000] Hi, everyone. Hope you're well.

[00:00:02.220] Next up, what I'd like to do is I'd like to build out Make More.

[00:00:05.860] Like microGrad before it,

[00:00:07.680] Make More is a repository that I have on my GitHub web page.

[00:00:11.100] You can look at it. But just like with microGrad,

[00:00:14.100] I'm going to build it out step-by-step,

[00:00:16.180] and I'm going to spell everything out.

[00:00:17.820] We're going to build it out slowly and together.

[00:00:19.900] Now, what is Make More?

[00:00:22.000] Make More, as the name suggests,

[00:00:24.460] makes more of things that you give it.

[00:00:27.500] Here's an example.

[00:00:28.940] Names.txt is an example dataset to Make More.

[00:00:32.420] When you look at Names.txt,

[00:00:34.220] you'll find that it's a very large dataset of names.

[00:00:39.260] Here's lots of different types of names.

[00:00:41.780] In fact, I believe there are 32,000 names

[00:00:44.340] that I found randomly on a government website.

[00:00:47.740] If you train Make More on this dataset,

[00:00:50.680] it will learn to make more of things like this.

[00:00:54.860] In particular, in this case,

[00:00:57.100] that will mean more things that sound name-like,

[00:01:00.380] but are actually unique names.

[00:01:02.300] Maybe if you have a baby and you're trying to assign a name,

[00:01:05.020] maybe you're looking for a cool new sounding unique name,

[00:01:07.620] Make More might help you.

[00:01:09.260] Here are some example generations from

[00:01:11.940] the neural network once we train it on our dataset.

[00:01:15.900] Here's some example unique names that it will generate.

[00:01:19.500] Dontel, iRot, Zendi, and so on.

[00:01:25.500] All these are sound name-like,

[00:01:27.820] but they're not, of course, names.

[00:01:30.220] Under the hood, Make More is a character level language model.

[00:01:34.780] What that means is that it is treating

[00:01:37.100] every single line here as an example,

[00:01:39.540] and within each example,

[00:01:41.000] it's treating them all as sequences of individual characters.

[00:01:45.140] R-E-E-S-E is this example,

[00:01:49.020] and that's the sequence of characters,

[00:01:50.620] and that's the level on which we are building out Make More.

[00:01:53.820] What it means to be a character level language model then,

[00:01:56.980] is that it's just modeling those sequences of

[00:01:59.500] characters and it knows how to

[00:02:00.700] predict the next character in the sequence.

[00:02:03.260] Now, we're actually going to implement

[00:02:05.300] a large number of character level language models

[00:02:08.100] in terms of the neural networks that are

[00:02:09.540] involved in predicting the next character in a sequence.

[00:02:12.380] Very simple bigram and bag of word models,

[00:02:15.300] multilayered perceptrons, recurrent neural networks,

[00:02:18.100] all the way to modern transformers.

[00:02:20.680] In fact, the transformer that we will build will be

[00:02:23.300] basically the equivalent transformer to

[00:02:25.100] GPT-2 if you have heard of GPT.

[00:02:27.940] That's a big deal. It's a model network,

[00:02:30.540] and by the end of the series,

[00:02:32.180] you will actually understand how that

[00:02:33.360] works on the level of characters.

[00:02:36.000] Now, to give you a sense of the extensions here,

[00:02:39.420] after characters, we will probably spend

[00:02:41.340] some time on the word level so that we can

[00:02:43.380] generate documents of words,

[00:02:44.780] not just little segments of characters,

[00:02:47.340] but we can generate entire much larger documents.

[00:02:50.900] Then we're probably going to go into images

[00:02:53.020] and image text networks such as Dolly,

[00:02:56.300] StableDiffusion, and so on.

[00:02:57.960] But for now, we have to start here,

[00:03:00.700] character level language modeling. Let's go.

[00:03:03.220] Like before, we are starting with

[00:03:04.820] a completely blank Jupyter Notebook page.

[00:03:06.980] The first thing is I would like to

[00:03:08.700] basically load up the dataset names.txt.

[00:03:11.500] We're going to open up names.txt for reading,

[00:03:14.940] and we're going to read in everything into a massive string.

[00:03:19.380] Then because it's a massive string,

[00:03:21.500] we only like the individual words and put them in the list.

[00:03:24.460] Let's call splitlines on that string to get

[00:03:28.220] all of our words as a Python list of strings.

[00:03:31.780] Basically, we can look at, for example,

[00:03:33.460] the first 10 words,

[00:03:35.500] and we have that it's a list of Emma,

[00:03:39.100] Olivia, Ava, and so on.

[00:03:41.380] If we look at the top of the page here,

[00:03:44.900] that is indeed what we see. That's good.

[00:03:49.460] This list actually makes me feel that

[00:03:52.020] this is probably sorted by frequency.

[00:03:54.940] But these are the words.

[00:03:58.420] Now, we'd like to actually learn

[00:04:00.140] a little bit more about this dataset.

[00:04:01.780] Let's look at the total number of words.

[00:04:03.380] We expect this to be roughly 32,000.

[00:04:06.140] Then what is the, for example, shortest word?

[00:04:08.940] So min of length of each word for w in words.

[00:04:13.580] So the shortest word will be length 2,

[00:04:17.980] and max of length w for w in words.

[00:04:20.900] So the longest word will be 15 characters.

[00:04:24.540] Let's now think through our very first language model.

[00:04:27.180] As I mentioned, a character level language model

[00:04:29.740] is predicting the next character in a sequence

[00:04:32.180] given already some concrete sequence of characters before it.

[00:04:36.260] Now, what we have to realize here is that every single word here,

[00:04:39.380] like Isabella, is actually

[00:04:41.500] quite a few examples packed in to that single word.

[00:04:45.340] Because what is an existence of a word like Isabella

[00:04:48.020] in the dataset telling us really?

[00:04:49.620] It's saying that the character i

[00:04:52.580] is a very likely character to come first in a sequence of a name.

[00:04:58.100] The character s is likely to come after i.

[00:05:03.540] The character a is likely to come after is.

[00:05:07.500] The character b is very likely to come after isa,

[00:05:10.620] and so on all the way to a following Isabelle.

[00:05:14.340] Then there's one more example actually packed in here,

[00:05:17.100] and that is that after there's Isabella,

[00:05:21.260] the word is very likely to end.

[00:05:23.540] So that's one more explicit piece of information that we have here,

[00:05:27.220] that we have to be careful with.

[00:05:29.500] So there's a lot packed into a single individual word in terms of

[00:05:33.220] the statistical structure of what's likely to follow in these character sequences.

[00:05:38.020] Then of course, we don't have just an individual word,

[00:05:40.220] we actually have 32,000 of these,

[00:05:42.260] and so there's a lot of structure here to model.

[00:05:44.580] Now in the beginning, what I'd like to start with,

[00:05:47.020] is I'd like to start with building a bigram language model.

[00:05:50.780] Now in a bigram language model,

[00:05:52.900] we're always working with just two characters at a time.

[00:05:56.460] So we're only looking at one character that we are given,

[00:06:00.420] and we're trying to predict the next character in the sequence.

[00:06:03.580] So what characters are likely to follow r,

[00:06:07.340] what characters are likely to follow a, and so on.

[00:06:09.780] We're just modeling that little local structure.

[00:06:12.860] We're forgetting the fact that we may have a lot more information,

[00:06:16.820] we're always just looking at the previous character to predict the next one.

[00:06:20.100] So it's a very simple and weak language model,

[00:06:22.220] but I think it's a great place to start.

[00:06:23.940] So now let's begin by looking at these bigrams in our dataset and what they look like.

[00:06:28.020] These bigrams again are just two characters in a row.

[00:06:30.740] So for w in words,

[00:06:32.900] each w here is an individual word string.

[00:06:36.100] We want to iterate this word with consecutive characters.

[00:06:43.540] So two characters at a time,

[00:06:45.180] sliding it through the word.

[00:06:46.620] Now, a interesting, nice way,

[00:06:49.180] cute way to do this in Python by the way,

[00:06:51.180] is doing something like this.

[00:06:52.740] For character1, character2 in zip of w and w at one, one column.

[00:07:01.260] Print character1, character2, and let's not do all the words,

[00:07:05.820] let's just do the first three words.

[00:07:07.380] I'm going to show you in a second how this works.

[00:07:09.860] But for now, basically as an example,

[00:07:12.020] let's just do the very first word alone, emma.

[00:07:14.820] You see how we have a emma and this will just print em,

[00:07:19.340] mm, ma.

[00:07:20.820] The reason this works is because w is the string emma,

[00:07:25.220] w at one column is the string mma,

[00:07:28.380] and zip takes two iterators and it pairs them up and then

[00:07:33.860] creates an iterator over the tuples of their consecutive entries.

[00:07:37.220] Any one of these lists is shorter than the other,

[00:07:40.380] then it will just halt and return.

[00:07:43.420] So basically, that's why we return em, mm, ma.

[00:07:49.820] But then because this iterator,

[00:07:51.900] the second one here, runs out of elements,

[00:07:53.980] zip just ends and that's why we only get these tuples.

[00:07:57.620] So pretty cute. These are the consecutive elements in the first word.

[00:08:02.980] Now we have to be careful because we actually have more information

[00:08:05.460] here than just these three examples.

[00:08:07.980] As I mentioned, we know that e is very likely to come first,

[00:08:12.340] and we know that a in this case is coming last.

[00:08:15.340] So one way to do this is basically we're going to

[00:08:18.260] create a special array here, characters,

[00:08:22.780] and we're going to hallucinate a special start token here.

[00:08:27.820] I'm going to call it special start.

[00:08:32.220] So this is a list of one element,

[00:08:34.340] plus w, and then plus a special end character.

[00:08:41.060] The reason I'm wrapping list of w here is because w is a string,

[00:08:45.100] emma, list of w will just have the individual characters in the list.

[00:08:50.500] Then doing this again now,

[00:08:53.180] but not iterating over w's,

[00:08:55.580] but over the characters will give us something like this.

[00:08:59.780] So this is a bigram of the start character and e,

[00:09:04.660] and this is a bigram of the a and the special end character.

[00:09:08.900] Now we can look at, for example,

[00:09:10.460] what this looks like for Olivia or Eva.

[00:09:14.100] Indeed, we can actually potentially do this for the entire dataset,

[00:09:18.100] but we won't print that, that's going to be too much.

[00:09:20.460] But these are the individual character bigrams and we can print them.

[00:09:24.660] Now, in order to learn the statistics about

[00:09:26.900] which characters are likely to follow other characters,

[00:09:29.780] the simplest way in the bigram language models is to simply do it by counting.

[00:09:34.220] So we're basically just going to count how often

[00:09:37.340] any one of these combinations occurs in the training set in these words.

[00:09:41.620] So we're going to need some kind of a dictionary that's going to

[00:09:44.140] maintain some counts for every one of these bigrams.

[00:09:47.220] So let's use a dictionary b,

[00:09:49.460] and this will map these bigrams.

[00:09:52.700] So bigram is a tuple of character one,

[00:09:54.540] character two, and then b at bigram will be b.get of bigram,

[00:10:00.900] which is basically the same as b at bigram.

[00:10:04.300] But in the case that bigram is not in the dictionary b,

[00:10:08.580] we would like to, by default, return a zero plus one.

[00:10:12.820] So this will basically add up all the bigrams and count how often they occur.

[00:10:18.060] Let's get rid of printing or rather,

[00:10:21.780] let's keep the printing and let's just inspect what b is in this case.

[00:10:26.820] We see that many bigrams occur just a single time.

[00:10:30.220] This one allegedly occurred three times.

[00:10:32.820] So a was an ending character three times,

[00:10:35.380] and that's true for all of these words.

[00:10:37.820] All of Emma, Olivia, and Eva end with a.

[00:10:41.420] So that's why this occurred three times.

[00:10:45.740] Now let's do it for all the words.

[00:10:49.780] Oops, I should not have printed.

[00:10:54.380] I meant to erase that. Let's kill this.

[00:10:58.380] Let's just run, and now b will have the statistics of the entire dataset.

[00:11:03.780] So these are the counts across all the words of the individual bigrams.

[00:11:08.260] We could, for example, look at some of the most common ones and least common ones.

[00:11:12.820] This grows in Python,

[00:11:14.780] but the way to do this,

[00:11:16.020] the simplest way I like is we just use b.items.

[00:11:19.380] b.items returns the tuples of key value.

[00:11:25.300] In this case, the keys are the character bigrams and the values are the counts.

[00:11:30.780] Then what we want to do is we want to do sorted of this.

[00:11:37.820] But by default, sort is on the first item of a tuple.

[00:11:45.580] But we want to sort by the values which are

[00:11:47.340] the second element of a tuple that is the key value.

[00:11:50.340] So we want to use the key equals lambda that takes

[00:11:55.860] the key value and returns the key value at one,

[00:12:01.580] not at zero but at one, which is the count.

[00:12:03.900] So we want to sort by the count of these elements.

[00:12:09.780] Actually, we want it to go backwards.

[00:12:12.540] So here what we have is the bigram QnR occurs only a single time.

[00:12:18.140] DZ occurred only a single time.

[00:12:20.620] When we sort this the other way around,

[00:12:22.980] we're going to see the most likely bigrams.

[00:12:26.220] So we see that n was very often an ending character many, many times.

[00:12:31.740] Apparently, n always follows an a,

[00:12:34.300] and that's a very likely combination as well.

[00:12:37.500] So this is the individual counts that we achieve over the entire dataset.

[00:12:44.380] Now, it's actually going to be significantly more convenient for us to

[00:12:48.220] keep this information in a two-dimensional array instead of a Python dictionary.

[00:12:53.220] So we're going to store this information in a 2D array,

[00:12:58.300] and the rows are going to be the first character of the bigram,

[00:13:02.900] and the columns are going to be the second character.

[00:13:05.140] Each entry in this two-dimensional array will tell us how

[00:13:08.020] often that first character follows the second character in the dataset.

[00:13:12.500] So in particular, the array representation that we're going to use,

[00:13:16.260] or the library, is that of PyTorch.

[00:13:18.820] PyTorch is a deep learning neural network framework,

[00:13:22.180] but part of it is also this torch.tensor,

[00:13:25.140] which allows us to create

[00:13:26.380] multi-dimensional arrays and manipulate them very efficiently.

[00:13:29.460] So let's import PyTorch,

[00:13:32.180] which you can do by import torch.

[00:13:34.260] Then we can create arrays.

[00:13:37.300] So let's create an array of zeros,

[00:13:40.380] and we give it a size of this array.

[00:13:43.900] Let's create a three by five array as an example,

[00:13:46.700] and this is a three by five array of zeros.

[00:13:51.380] By default, you'll notice a.dtype,

[00:13:53.980] which is short for data type, is float 32.

[00:13:56.620] So these are single precision floating point numbers.

[00:13:59.260] Because we are going to represent counts,

[00:14:01.780] let's actually use.dtype as torch.int32.

[00:14:05.580] So these are 32-bit integers.

[00:14:09.900] So now you see that we have integer data inside this tensor.

[00:14:14.220] Now, tensors allow us to really

[00:14:16.940] manipulate all the individual entries and do it very efficiently.

[00:14:20.500] So for example, if we want to change this bit,

[00:14:23.580] we have to index into the tensor.

[00:14:25.660] In particular, here, this is the first row,

[00:14:29.580] and because it's zero indexed.

[00:14:32.820] So this is row index one and column index zero, one, two, three.

[00:14:38.500] So a at one comma three,

[00:14:41.220] we can set that to one,

[00:14:43.180] and then a will have a one over there.

[00:14:46.500] We can, of course, also do things like this.

[00:14:49.220] So now a will be two over there, or three.

[00:14:53.580] Also, we can, for example, say a00 is five,

[00:14:56.980] and then a will have a five over here.

[00:14:59.860] So that's how we can index into the arrays.

[00:15:03.180] Now, of course, the array that we are interested in is much, much bigger.

[00:15:06.060] So for our purposes,

[00:15:07.460] we have 26 letters of the alphabet,

[00:15:09.740] and then we have two special characters, S and E.

[00:15:13.780] So we want 26 plus two,

[00:15:16.700] or 28 by 28 array,

[00:15:19.060] and let's call it the capital N,

[00:15:21.020] because it's going to represent the counts.

[00:15:23.940] Let me erase this stuff.

[00:15:26.220] So that's the array that starts at zeros, 28 by 28.

[00:15:30.300] Now, let's copy-paste this here.

[00:15:34.460] But instead of having a dictionary B,

[00:15:36.780] which we're going to erase,

[00:15:38.500] we now have an N.

[00:15:40.540] Now, the problem here is that we have these characters which are strings,

[00:15:44.180] but we have to now basically index into

[00:15:47.340] a array and we have to index using integers.

[00:15:51.300] So we need some kind of a lookup table from characters to integers.

[00:15:55.060] So let's construct such a character array.

[00:15:57.900] The way we're going to do this is we're going to take all the words,

[00:16:01.020] which is a list of strings.

[00:16:02.620] We're going to concatenate all of it into a massive string.

[00:16:05.540] So this is just simply the entire dataset as a single string.

[00:16:08.820] We're going to pass this to the set constructor,

[00:16:11.780] which takes this massive string and

[00:16:14.900] throws out duplicates because sets do not allow duplicates.

[00:16:18.700] So set of this will just be the set of all the lowercase characters.

[00:16:23.780] There should be a total of 26 of them.

[00:16:27.860] Now, we actually don't want a set,

[00:16:29.940] we want a list.

[00:16:31.860] But we don't want a list sorted in some weird arbitrary way,

[00:16:35.620] we want it to be sorted from A to Z.

[00:16:39.180] So sorted list.

[00:16:41.460] So those are our characters.

[00:16:44.660] Now, what we want is this lookup table as I mentioned.

[00:16:47.820] So let's create a special S2I, I will call it.

[00:16:52.740] S is string or character,

[00:16:55.460] and this will be an S2I mapping for IS in enumerate of these characters.

[00:17:03.820] So enumerate basically gives us this iterator over

[00:17:07.700] the integer, index, and the actual element of the list.

[00:17:11.980] Then we are mapping the character to the integer.

[00:17:14.820] So S2I is a mapping from A to 0,

[00:17:18.420] B to 1, etc, all the way from Z to 25.

[00:17:22.220] That's going to be useful here,

[00:17:25.340] but we actually also have to specifically set that S will be 26,

[00:17:29.580] and S2I at E will be 27 because Z was 25.

[00:17:35.300] So those are the lookups.

[00:17:37.580] Now, we can come here and we can map

[00:17:39.740] both character 1 and character 2 to their integers.

[00:17:42.580] So this will be S2I at character 1,

[00:17:44.980] and IX2 will be S2I of character 2.

[00:17:48.860] Now, we should be able to do this line,

[00:17:53.180] but using our array.

[00:17:54.420] So n at IX1, IX2,

[00:17:57.220] this is the two-dimensional array indexing I've shown you before,

[00:18:00.460] and honestly just plus equals 1 because everything starts at 0.

[00:18:05.740] So this should work and give us

[00:18:10.140] a large 28 by 28 array of all these counts.

[00:18:14.300] So if we print n, this is the array,

[00:18:17.860] but of course it looks ugly.

[00:18:19.700] So let's erase this ugly mess,

[00:18:21.860] and let's try to visualize it a bit more nicer.

[00:18:24.540] So for that, we're going to use a library called matplotlib.

[00:18:28.420] So matplotlib allows us to create figures.

[00:18:31.020] So we can do things like pltim show of the count array.

[00:18:35.340] So this is the 28 by 28 array,

[00:18:38.740] and this is the structure,

[00:18:40.860] but even this, I would say, is still pretty ugly.

[00:18:43.620] So we're going to try to create a much nicer visualization of it,

[00:18:47.020] and I wrote a bunch of code for that.

[00:18:49.220] The first thing we're going to need is,

[00:18:51.740] we're going to need to invert this array here, this dictionary.

[00:18:56.500] So S2I is a mapping from S to I,

[00:18:59.540] and in I2S, we're going to reverse this dictionary.

[00:19:02.980] So iterate over all the items and just reverse that array.

[00:19:06.300] So I2S maps inversely from 0 to A,

[00:19:10.580] 1 to B, etc. So we'll need that.

[00:19:14.020] Then here's the code that I came up with to try to

[00:19:16.900] make this a little bit nicer.

[00:19:19.660] We create a figure, we plot n,

[00:19:24.180] and then we visualize a bunch of things here.

[00:19:27.220] Let me just run it so you get a sense of what this is.

[00:19:32.100] So you see here that we have the array spaced out,

[00:19:36.980] and every one of these is basically like B follows G 0 times,

[00:19:42.140] B follows H 41 times.

[00:19:44.780] So A follows J 175 times.

[00:19:47.860] What you can see that I'm doing here is,

[00:19:49.940] first I show that entire array,

[00:19:52.820] and then I iterate over all the individual cells here,

[00:19:56.380] and I create a character string here,

[00:19:59.180] which is the inverse mapping I2S of the integer I and the integer J.

[00:20:04.580] So those are the bigrams in a character representation.

[00:20:08.260] Then I plot just the bigram text,

[00:20:12.020] and then I plot the number of times that this bigram occurs.

[00:20:15.820] Now, the reason that there's a dot item here is

[00:20:18.660] because when you index into these arrays,

[00:20:21.060] these are torch tensors,

[00:20:22.820] you see that we still get a tensor back.

[00:20:25.860] So the type of this thing,

[00:20:27.620] you'd think it would be just an integer, 149,

[00:20:29.820] but it's actually a torch dot tensor.

[00:20:31.900] If you do dot item,

[00:20:34.340] then it will pop out that individual integer.

[00:20:37.900] So it'll just be 149.

[00:20:40.340] So that's what's happening there.

[00:20:42.300] These are just some options to make it look nice.

[00:20:44.900] So what is the structure of this array?

[00:20:48.340] We have all these counts and we see that some of them occur often,

[00:20:51.940] and some of them do not occur often.

[00:20:53.780] Now, if you scrutinize this carefully,

[00:20:55.940] you will notice that we're not actually being very clever.

[00:20:58.380] That's because when you come over here,

[00:21:00.420] you'll notice that, for example,

[00:21:01.700] we have an entire row of completely zeros.

[00:21:04.580] That's because the end character is never

[00:21:07.540] possibly going to be the first character of a bigram,

[00:21:09.820] because we're always placing these end tokens at the end of a bigram.

[00:21:13.940] Similarly, we have an entire column of zeros here,

[00:21:17.340] because the S character will never possibly be the second element of a bigram,

[00:21:23.300] because we always start with S and we end with E,

[00:21:25.780] and we only have the words in between.

[00:21:27.500] So we have an entire column of zeros,

[00:21:29.980] an entire row of zeros,

[00:21:31.620] and in this little two-by-two matrix here as well,

[00:21:34.020] the only one that can possibly happen is if S directly follows E.

[00:21:38.340] That can be non-zero if we have a word that has no letters.

[00:21:42.940] So in that case, there's no letters in the word,

[00:21:44.700] it's an empty word, and we just have S follows E.

[00:21:47.260] But the other ones are just not possible.

[00:21:50.060] So we're basically wasting space,

[00:21:51.740] and not only that, but the S and the E are getting very crowded here.

[00:21:55.380] I was using these brackets because there's

[00:21:57.860] convention in natural language processing to use

[00:21:59.860] these kinds of brackets to denote special tokens,

[00:22:02.980] but we're going to use something else.

[00:22:04.900] So let's fix all this and make it prettier.

[00:22:07.980] We're not actually going to have two special tokens,

[00:22:10.420] we're only going to have one special token.

[00:22:12.700] So we're going to have n by n array of 27 by set 27 instead.

[00:22:18.300] Instead of having two,

[00:22:20.340] we will just have one and I will call it a dot.

[00:22:26.420] Let me swing this over here.

[00:22:29.700] Now, one more thing that I would like to do is I would actually

[00:22:32.700] like to make this special character half position zero,

[00:22:36.100] and I would like to offset all the other letters off.

[00:22:38.980] I find that a little bit more pleasing.

[00:22:41.900] So we need a plus one here so that the first character,

[00:22:47.220] which is A, will start at one.

[00:22:49.420] So S2i will now be A starts at one and dot is zero.

[00:22:55.980] I2S, of course, we're not changing this because I2S

[00:22:59.660] just creates a reverse mapping and this will work fine.

[00:23:02.140] So one is A, two is B,

[00:23:04.180] zero is dot. So we've reversed that.

[00:23:07.820] Here, we have a dot and a dot.

[00:23:12.180] This should work fine.

[00:23:14.460] Make sure I start at zeros, count.

[00:23:18.660] Then here, we don't go up to 28,

[00:23:20.380] we go up to 27,

[00:23:22.100] and this should just work.

[00:23:30.860] So we see that dot dot never happened,

[00:23:33.540] it's at zero because we don't have empty words.

[00:23:36.220] Then this row here now is just very simply

[00:23:39.460] the counts for all the first letters.

[00:23:43.420] So J starts a word,

[00:23:46.540] H starts a word, I starts a word, etc.

[00:23:49.460] Then these are all the ending characters.

[00:23:52.900] In between, we have the structure of

[00:23:54.740] what characters follow each other.

[00:23:56.700] So this is the counts array of our entire dataset.

[00:24:01.500] This array actually has all the information necessary for us to

[00:24:05.020] actually sample from this bigram character level language model.

[00:24:09.980] Roughly speaking, what we're going to do is we're just going to

[00:24:12.940] start following these probabilities and these counts,

[00:24:15.700] and we're going to start sampling from the model.

[00:24:18.660] In the beginning, of course,

[00:24:20.340] we start with the dot,

[00:24:21.820] the start token dot.

[00:24:24.420] So to sample the first character of a name,

[00:24:28.060] we're looking at this row here.

[00:24:30.060] So we see that we have the counts,

[00:24:32.700] and those counts externally are telling us how

[00:24:34.940] often any one of these characters is to start a word.

[00:24:39.140] So if we take this n and we grab the first row,

[00:24:44.340] we can do that by using just indexing at zero,

[00:24:48.220] and then using this notation colon for the rest of that row.

[00:24:53.260] So n zero colon is indexing into the zero row,

[00:24:59.380] and then it's grabbing all the columns.

[00:25:01.740] So this will give us a one-dimensional array of the first row.

[00:25:06.180] So 0, 4, 4, 10.

[00:25:08.220] You notice 0, 4, 4, 10,

[00:25:10.340] 1306, 1542, etc.

[00:25:12.980] It's just the first row.

[00:25:14.180] The shape of this is 27.

[00:25:17.140] It's just the row of 27.

[00:25:19.460] The other way that you can do this also is you

[00:25:21.700] just grab the zero row like this.

[00:25:26.220] This is equivalent.

[00:25:27.660] Now, these are the counts.

[00:25:29.940] Now, what we'd like to do is we'd like to

[00:25:32.300] basically sample from this.

[00:25:34.740] Since these are the raw counts,

[00:25:36.140] we actually have to convert this to probabilities.

[00:25:38.780] So we create a probability vector.

[00:25:42.340] So we'll take n of zero,

[00:25:44.740] and we'll actually convert this to float first.

[00:25:49.540] So these integers are converted to floating-point numbers.

[00:25:54.100] The reason we're creating floats is because we're

[00:25:56.340] about to normalize these counts.

[00:25:58.540] So to create a probability distribution here,

[00:26:01.380] we basically want to do p divide p dot sum.

[00:26:08.860] Now, we get a vector of smaller numbers,

[00:26:12.100] and these are now probabilities.

[00:26:13.660] So of course, because we divided by the sum,

[00:26:15.780] the sum of p now is one.

[00:26:18.580] So this is a nice proper probability distribution.

[00:26:21.100] It sums to one, and this is giving us the probability for

[00:26:23.820] any single character to be the first character of a word.

[00:26:27.460] So now we can try to sample from this distribution.

[00:26:30.540] To sample from these distributions,

[00:26:32.180] we're going to use Torch.Multinomial,

[00:26:34.140] which I've pulled up here.

[00:26:35.700] So Torch.Multinomial returns samples

[00:26:40.940] from the multinomial probability distribution,

[00:26:43.140] which is a complicated way of saying,

[00:26:45.060] you give me probabilities and I will give you integers,

[00:26:47.940] which are sampled according to the probability distribution.

[00:26:51.340] So this is the signature of the method.

[00:26:53.180] To make everything deterministic,

[00:26:54.780] we're going to use a generator object in PyTorch.

[00:26:58.660] So this makes everything deterministic.

[00:27:00.940] So when you run this on your computer,

[00:27:02.580] you're going to get the exact same results

[00:27:04.660] that I'm getting here on my computer.

[00:27:06.780] So let me show you how this works.

[00:27:11.740] Here's a deterministic way of creating

[00:27:15.420] a Torch generator object,

[00:27:17.700] seeding it with some number that we can agree on.

[00:27:20.940] So that seeds a generator, gives us an object g.

[00:27:24.820] Then we can pass that g to a function that creates here,

[00:27:30.620] random numbers, Torch.rand creates random numbers,

[00:27:33.780] three of them, and it's using

[00:27:35.860] this generator object as a source of randomness.

[00:27:39.620] So without normalizing it, I can just print.

[00:27:45.820] This is like numbers between 0 and 1 that are random,

[00:27:49.860] according to this thing.

[00:27:51.300] Whenever I run it again,

[00:27:52.980] I'm always going to get the same result because I

[00:27:55.060] keep using the same generator object which I'm seeding here.

[00:27:58.420] Then if I divide to normalize,

[00:28:02.780] I'm going to get a nice probability distribution

[00:28:05.220] of just three elements.

[00:28:07.140] Then we can use Torch.Multinomial to draw samples from it.

[00:28:10.980] So this is what that looks like.

[00:28:13.180] Torch.Multinomial will take

[00:28:16.020] the Torch tensor of probability distributions.

[00:28:20.460] Then we can ask for a number of samples, let's say 20.

[00:28:23.940] Replacement equals true means that when we draw an element,

[00:28:29.100] we can draw it and then we can put it back into

[00:28:32.060] the list of eligible indices to draw again.

[00:28:35.780] We have to specify replacement as true because by

[00:28:38.340] default for some reason it's false.

[00:28:41.380] I think it's just something to be careful with.

[00:28:45.580] The generator is passed in here,

[00:28:47.380] so we're going to always get

[00:28:48.700] deterministic results, the same results.

[00:28:51.260] If I run these two,

[00:28:53.380] we're going to get a bunch of samples from this distribution.

[00:28:56.780] Now, you'll notice here that the probability for

[00:28:59.700] the first element in this tensor is 60 percent.

[00:29:04.380] In these 20 samples,

[00:29:06.460] we'd expect 60 percent of them to be zero.

[00:29:10.140] We'd expect 30 percent of them to be one.

[00:29:13.900] Because the element index two has only 10 percent probability,

[00:29:19.340] very few of these samples should be two.

[00:29:22.100] Indeed, we only have a small number of twos.

[00:29:25.100] We can sample as many as we would like.

[00:29:28.260] The more we sample, the more

[00:29:30.940] these numbers should roughly have the distribution here.

[00:29:35.460] We should have lots of zeros,

[00:29:38.180] half as many ones,

[00:29:42.460] and we should have three times as few ones,

[00:29:48.020] and three times as few twos.

[00:29:51.500] You see that we have very few twos,

[00:29:53.380] we have some ones, and most of them are zero.

[00:29:55.740] That's what Torchline multinomial is doing.

[00:29:58.460] For us here, we are interested in this row.

[00:30:02.460] We've created this p here,

[00:30:06.700] and now we can sample from it.

[00:30:09.100] If we use the same seed,

[00:30:12.420] and then we sample from this distribution,

[00:30:15.180] let's just get one sample,

[00:30:17.260] then we see that the sample is say 13.

[00:30:22.060] This will be the index.

[00:30:25.300] You see how it's a tensor that wraps 13?

[00:30:28.620] We again have to use that item to pop out that integer.

[00:30:32.740] Now, index would be just the number 13.

[00:30:36.740] Of course, we can map

[00:30:40.900] the i2s of ix to figure out

[00:30:43.620] exactly which character we're sampling here.

[00:30:46.060] We're sampling m. We're saying that

[00:30:48.860] the first character is m in our generation.

[00:30:53.020] Just looking at the row here,

[00:30:54.980] m was drawn, and we can see that

[00:30:57.060] m actually starts a large number of words.

[00:30:59.820] m started 2,500 words out of 32,000 words,

[00:31:04.660] so almost a bit less than 10 percent

[00:31:07.420] of the words start with m.

[00:31:09.060] This was actually a fairly likely character to draw.

[00:31:14.460] That would be the first character of our word,

[00:31:17.020] and now we can continue to sample

[00:31:18.620] more characters because now we know that m is already sampled.

[00:31:24.540] Now to draw the next character,

[00:31:26.680] we will come back here and we will look

[00:31:29.340] for the row that starts with m.

[00:31:32.540] You see m and we have a row here.

[00:31:36.460] We see that mdot is 516,

[00:31:40.660] ma is this many, mb is this many, etc.

[00:31:43.860] These are the counts for the next row,

[00:31:45.500] and that's the next character that we are going to now generate.

[00:31:48.460] I think we are ready to actually just write out the loop

[00:31:51.140] because I think you're starting to get a sense

[00:31:52.580] of how this is going to go.

[00:31:55.340] We always begin at index 0 because that's the start token.

[00:32:01.620] Then while true, we're going to grab

[00:32:05.700] the row corresponding to index that we're currently on,

[00:32:09.700] so that's n array at ix,

[00:32:14.220] converted to float is rp.

[00:32:18.060] Then we normalize the speed to sum to one.

[00:32:24.500] I accidentally ran the infinite loop.

[00:32:27.900] We normalize p to sum to one.

[00:32:30.460] Then we need this generator object.

[00:32:33.260] We're going to initialize up here and we're going to

[00:32:36.220] draw a single sample from this distribution.

[00:32:40.100] Then this is going to tell us what index is going to be next.

[00:32:45.700] If the index sampled is zero,

[00:32:49.580] then that's now the end token, so we will break.

[00:32:54.620] Otherwise, we are going to print i2s of ix.

[00:33:04.620] That's pretty much it.

[00:33:07.260] We're just, this should work.

[00:33:09.940] More. That's the name that we've sampled.

[00:33:14.620] We started with m,

[00:33:16.500] the next step was o,

[00:33:17.700] then r, and then dot.

[00:33:20.700] This dot, we print it here as well.

[00:33:24.460] Let's now do this a few times.

[00:33:29.260] Let's actually create an out list here.

[00:33:36.380] Instead of printing, we're going to append,

[00:33:39.500] so out.append this character.

[00:33:43.660] Then here, let's just print it at the end.

[00:33:47.140] Let's just join up all the outs and we're just going to print more.

[00:33:51.540] Now, we're always getting the same result because of the generator.

[00:33:54.940] If we want to do this a few times,

[00:33:56.940] we can go for i in range 10.

[00:34:00.660] We can sample 10 names and we can just do that 10 times.

[00:34:04.980] These are the names that we're getting out. Let's do 20.

[00:34:13.420] I'll be honest with you, this doesn't look right.

[00:34:16.340] I stare at it a few minutes to convince myself that it actually is right.

[00:34:20.220] The reason these samples are so terrible is

[00:34:22.660] that bigram language model is actually just like really terrible.

[00:34:27.460] We can generate a few more here.

[00:34:29.700] You can see that they're name like a little bit like Yanu,

[00:34:33.900] Erily, etc, but they're just totally messed up.

[00:34:38.420] The reason that this is so bad,

[00:34:40.940] we're generating h as a name,

[00:34:42.940] but you have to think through it from the model's eyes.

[00:34:46.380] It doesn't know that this h is the very first h.

[00:34:49.460] All it knows is that h was previously,

[00:34:52.060] and now how likely is h the last character?

[00:34:55.580] Well, it's somewhat likely,

[00:34:57.740] and so it just makes it last character.

[00:34:59.340] It doesn't know that there were other things before it,

[00:35:01.700] or there were not other things before it.

[00:35:03.940] That's why it's generating all these nonsense names.

[00:35:07.980] Another way to do this is to

[00:35:12.260] convince yourself that it is actually doing something reasonable,

[00:35:14.420] even though it's so terrible,

[00:35:15.820] is these little p's here are 27,

[00:35:20.460] like 27. How about if we did something like this?

[00:35:25.580] Instead of p having any structure whatsoever,

[00:35:28.740] how about if p was just torch.ones of 27?

[00:35:36.620] By default, this is a float 32,

[00:35:39.020] so this is fine. Divide 27.

[00:35:42.140] What I'm doing here is this is

[00:35:44.940] the uniform distribution which will make everything equally likely,

[00:35:49.340] and we can sample from that.

[00:35:51.860] Let's see if that does any better.

[00:35:55.100] This is what you have from a model that is completely untrained,

[00:35:58.700] where everything is equally likely,

[00:36:00.460] so it's obviously garbage.

[00:36:02.380] Then if we have a trained model which is trained on just bigrams,

[00:36:06.900] this is what we get.

[00:36:08.380] You can see that it is more name-like, it is actually working.

[00:36:11.740] It's just bigram is so terrible and we have to do better.

[00:36:16.460] Now next, I would like to fix an inefficiency that we have going on here.

[00:36:20.020] Because what we're doing here is we're always fetching

[00:36:22.940] a row of n from the counts matrix up ahead,

[00:36:26.180] and then we're always doing the same things.

[00:36:27.980] We're converting to float and we're dividing,

[00:36:29.840] and we're doing this every single iteration of this loop,

[00:36:32.620] and we just keep renormalizing these rows over and over again,

[00:36:35.020] and it's extremely inefficient and wasteful.

[00:36:37.180] What I'd like to do is I'd like to actually prepare

[00:36:39.860] a matrix capital P that will just have the probabilities in it.

[00:36:43.860] In other words, it's going to be the same as the capital N matrix here of counts,

[00:36:47.980] but every single row will have the row of probabilities that is normalized to one,

[00:36:52.900] indicating the probability distribution for the next character,

[00:36:55.860] given the character before it as defined by which row we're in.

[00:37:01.220] Basically, what we'd like to do is we'd like to just do it up front here,

[00:37:04.940] and then we would like to just use that row here.

[00:37:07.940] Here, we would like to just do p equals p of ix instead.

[00:37:14.020] The other reason I want to do this is not just for efficiency,

[00:37:17.060] but also I would like us to practice

[00:37:19.300] these n-dimensional tensors and I'd like us to practice

[00:37:22.600] their manipulation and especially something that's called

[00:37:24.700] broadcasting that we'll go into in a second.

[00:37:26.740] We're actually going to have to become very good at

[00:37:28.940] these tensor manipulations because if we're

[00:37:31.140] going to build out all the way to transformers,

[00:37:33.180] we're going to be doing some pretty complicated array operations for

[00:37:36.500] efficiency and we need to really understand that and be very good at it.

[00:37:41.500] Intuitively, what we want to do is that we first want to grab

[00:37:44.700] the floating point copy of n,

[00:37:47.740] and I'm mimicking the line here basically,

[00:37:50.540] and then we want to divide all the rows so that they sum to one.

[00:37:55.660] We'd like to do something like this, p divide p dot sum.

[00:37:59.780] But now we have to be careful because p dot sum actually

[00:38:04.940] produces a sum, sorry,

[00:38:08.660] equals n dot float copy.

[00:38:10.420] p dot sum sums up all of the counts of this entire matrix n,

[00:38:18.100] and gives us a single number of just the summation of everything.

[00:38:21.140] That's not the way we want to divide.

[00:38:23.460] We want to simultaneously and in parallel,

[00:38:25.980] divide all the rows by their respective sums.

[00:38:30.380] What we have to do now is we have to go into documentation for

[00:38:34.180] tors.sum and we can scroll down here to a definition that is relevant to us,

[00:38:38.780] which is where we don't only provide an input array that we want to sum,

[00:38:43.340] but we also provide the dimension along which we want to sum.

[00:38:47.020] In particular, we want to sum up over rows.

[00:38:51.780] Now, one more argument that I want you to pay attention to here

[00:38:54.820] is the keepDim is false.

[00:38:57.460] If keepDim is true,

[00:38:59.740] then the output tensor is of the same size as input,

[00:39:02.420] except of course the dimension along which you summed,

[00:39:04.780] which will become just one.

[00:39:06.820] But if you pass in keepDim as false,

[00:39:11.460] then this dimension is squeezed out.

[00:39:14.500] Torch.sum not only does the sum and collapses dimension to be of size one,

[00:39:18.860] but in addition, it does what's called a squeeze,

[00:39:21.100] where it squeezes out that dimension.

[00:39:24.900] Basically, what we want here is we instead want to do p dot sum of sum axis.

[00:39:30.420] In particular, notice that p dot shape is 27 by 27.

[00:39:35.140] When we sum up across axis zero,

[00:39:37.780] then we would be taking the zeroth dimension and we would be summing across it.

[00:39:42.260] When keepDim is true,

[00:39:44.580] then this thing will not only give us the counts along the columns,

[00:39:51.660] but notice that basically the shape of this is one by 27.

[00:39:55.500] We just get a row vector.

[00:39:57.220] The reason we get a row vector here again is because we pass in zero dimension.

[00:40:01.020] This zeroth dimension becomes one and we've done a sum and we get a row.

[00:40:05.980] Basically, we've done the sum this way,

[00:40:09.140] vertically, and arrived at just a single one by 27 vector of counts.

[00:40:14.700] What happens when you take out keepDim is that we just get 27.

[00:40:19.580] It squeezes out that dimension and we just get a one-dimensional vector of size 27.

[00:40:27.860] Now, we don't actually want one by 27 row vector,

[00:40:32.980] because that gives us the counts or the sums across the columns.

[00:40:39.180] We actually want to sum the other way along dimension one.

[00:40:42.620] You'll see that the shape of this is 27 by one,

[00:40:45.620] so it's a column vector.

[00:40:47.260] It's a 27 by one vector of counts.

[00:40:53.100] That's because what's happened here is that we're going horizontally,

[00:40:56.780] and this 27 by 27 matrix becomes a 27 by one array.

[00:41:02.820] Now, you'll notice, by the way,

[00:41:04.580] that the actual numbers of these counts are identical.

[00:41:10.380] That's because this special array of counts here comes from bigram statistics.

[00:41:14.860] Actually, it just so happens by chance,

[00:41:17.620] or because of the way this array is constructed,

[00:41:20.100] that the sums along the columns or along the rows,

[00:41:22.940] horizontally or vertically, is identical.

[00:41:25.740] But actually, what we want to do in this case is we want to sum across the rows horizontally.

[00:41:32.340] What we want here is to be that sum of one with keepDim true,

[00:41:36.700] 27 by one column vector.

[00:41:39.300] Now, what we want to do is we want to divide by that.

[00:41:42.860] Now, we have to be careful here again.

[00:41:46.180] Is it possible to take what's a p.shape you see here as 27 by 27?

[00:41:52.660] Is it possible to take a 27 by 27 array and divide it by what is a 27 by one array?

[00:42:00.540] Is that an operation that you can do?

[00:42:03.660] Whether or not you can perform this operation is determined by what's called broadcasting rules.

[00:42:08.220] If you just search broadcasting semantics in Torch,

[00:42:11.700] you'll notice that there's a special definition for what's called broadcasting,

[00:42:15.740] that for whether or not these two arrays can be combined in a binary operation like division.

[00:42:23.500] The first condition is each tensor has at least one dimension,

[00:42:26.740] which is the case for us.

[00:42:28.420] Then when iterating over the dimension sizes,

[00:42:30.500] starting at the trailing dimension,

[00:42:32.220] the dimension sizes must either be equal,

[00:42:34.540] one of them is one, or one of them does not exist.

[00:42:38.460] Let's do that.

[00:42:40.260] We need to align the two arrays and their shapes,

[00:42:44.100] which is very easy because both of these shapes have two elements, so they're aligned.

[00:42:48.140] Then we iterate over from the right and going to the left.

[00:42:52.340] Each dimension must be either equal, one of them is a one, or one of them does not exist.

[00:42:57.940] In this case, they're not equal, but one of them is a one, so this is fine.

[00:43:01.940] Then this dimension, they're both equal, so this is fine.

[00:43:05.900] All the dimensions are fine and therefore this operation is broadcastable.

[00:43:11.580] That means that this operation is allowed.

[00:43:14.420] What is it that these arrays do when you divide 27 by 27 by 27 by one?

[00:43:19.580] What it does is that it takes this dimension one and it stretches it out,

[00:43:24.020] it copies it to match 27 here in this case.

[00:43:28.980] In our case, it takes this column vector,

[00:43:31.220] which is 27 by one and it copies it 27 times to make these both be 27 by 27 internally.

[00:43:40.420] You can think of it that way.

[00:43:41.780] It copies those counts and then it does an element-wise division,

[00:43:46.780] which is what we want because these counts,

[00:43:49.340] we want to divide by them on every single one of these columns in this matrix.

[00:43:54.380] This actually we expect will normalize every single row.

[00:43:59.300] We can check that this is true by taking the first row,

[00:44:02.460] for example, and taking its sum.

[00:44:05.180] We expect this to be one because it's now normalized.

[00:44:09.940] Then we expect this now because if we actually correctly normalize all the rows,

[00:44:15.420] we expect to get the exact same result here.

[00:44:17.700] Let's run this. It's the exact same result. This is correct.

[00:44:22.860] Now I would like to scare you a little bit.

[00:44:25.660] I basically encourage you very strongly to read through broadcasting semantics,

[00:44:30.300] and I encourage you to treat this with respect.

[00:44:32.700] It's not something to play fast and loose with,

[00:44:35.220] it's something to really respect,

[00:44:36.660] really understand, and look up maybe some tutorials for broadcasting and practice it,

[00:44:40.340] and be careful with it because you can very quickly run into bugs.

[00:44:43.740] Let me show you what I mean.

[00:44:46.300] You see how here we have p.sum of one, keep them as true.

[00:44:50.300] The shape of this is 27 by one.

[00:44:52.820] Let me take out this line just so we have the n and then we can see the counts.

[00:44:58.340] We can see that this is all the counts across all the rows,

[00:45:03.420] and it's 27 by one column vector.

[00:45:06.340] Now, suppose that I tried to do the following,

[00:45:10.340] but I erase keep them as true here. What does that do?

[00:45:14.900] If keep them is not true, it's false.

[00:45:17.180] Then remember, according to documentation,

[00:45:19.220] it gets rid of this dimension one,

[00:45:21.380] it squeezes it out.

[00:45:22.860] Basically, we just get all the same counts,

[00:45:25.380] the same result, except the shape of it is not 27 by one,

[00:45:28.980] it's just 27, the one that disappears.

[00:45:31.420] But all the counts are the same.

[00:45:33.940] You'd think that this divide that would work.

[00:45:39.740] First of all, can we even write this and is it even expected to run?

[00:45:45.100] Is it broadcastable? Let's determine if this result is broadcastable.

[00:45:48.860] p.summit1 is shape, is 27.

[00:45:52.700] This is 27 by 27,

[00:45:54.420] so 27 by 27 broadcasting into 27.

[00:45:59.940] Now, rules of broadcasting, number 1,

[00:46:03.340] align all the dimensions on the right, done.

[00:46:06.140] Now, iteration over all the dimensions starting from the right,

[00:46:08.740] going to the left.

[00:46:09.900] All the dimensions must either be equal,

[00:46:12.460] one of them must be one or one of them does not exist.

[00:46:15.940] Here, they are all equal.

[00:46:17.580] Here, the dimension does not exist.

[00:46:19.820] Internally, what broadcasting will do is it will create a one here,

[00:46:23.980] and then we see that one of them is a one,

[00:46:27.580] and this will get copied,

[00:46:29.100] and this will run, this will broadcast.

[00:46:32.180] You'd expect this to work because we can divide this.

[00:46:43.460] Now, if I run this,

[00:46:44.660] you'd expect it to work, but it doesn't.

[00:46:47.780] You actually get garbage.

[00:46:49.340] You get a wrong result because this is actually a bug.

[00:46:52.260] This keep them equals true makes it work.

[00:46:59.740] This is a bug. In both cases,

[00:47:03.620] we are doing the correct counts.

[00:47:06.340] We are summing up across the rows,

[00:47:08.940] but keep them as saving us and making it work.

[00:47:11.460] In this case, I'd like to encourage you to potentially

[00:47:14.540] pause this video at this point and try to think about

[00:47:16.700] why this is buggy and why the keep them was necessary here.

[00:47:22.340] The reason for this is I'm trying to hint it here

[00:47:26.700] when I was giving you a bit of a hint on how this works.

[00:47:29.260] This 27 vector internally inside the broadcasting,

[00:47:34.100] this becomes a one by 27,

[00:47:36.220] and one by 27 is a row vector.

[00:47:39.140] Now, we are dividing 27 by 27 by one by 27,

[00:47:42.900] and torch will replicate this dimension.

[00:47:45.900] Basically, it will take this row vector,

[00:47:51.900] and it will copy it vertically now 27 times.

[00:47:56.340] The 27 by 27 aligns exactly,

[00:47:58.060] and element-wise divides.

[00:48:00.060] Basically, what's happening here is we're

[00:48:04.660] actually normalizing the columns instead of normalizing the rows.

[00:48:08.780] You can check that what's happening here is that

[00:48:12.420] p at zero, which is the first row of p,

[00:48:15.020] that sum is not one, it's seven.

[00:48:18.220] It is the first column as an example that sums to one.

[00:48:23.580] To summarize, where does the issue come from?

[00:48:26.660] The issue comes from the silent adding of a dimension here,

[00:48:29.580] because in broadcasting rules,

[00:48:31.260] you align on the right and go from right to left,

[00:48:33.740] and if dimension doesn't exist, you create it.

[00:48:36.020] That's where the problem happens.

[00:48:37.780] We still did the counts correctly.

[00:48:39.420] We did the counts across the rows,

[00:48:41.260] and we got the counts on the right here as a column vector.

[00:48:45.460] But because the keepdence was true,

[00:48:47.220] this dimension was discarded,

[00:48:49.380] and now we just have a vector of 27.

[00:48:51.300] Because of broadcasting the way it works,

[00:48:53.780] this vector of 27 suddenly becomes a row vector,

[00:48:56.820] and then this row vector gets replicated vertically,

[00:48:59.780] and that every single point we are dividing by

[00:49:01.940] the count in the opposite direction.

[00:49:07.500] This thing just doesn't work.

[00:49:11.380] This needs to be keepdence equals true in this case.

[00:49:15.460] Then we have that p at zero is normalized.

[00:49:19.380] Conversely, the first column,

[00:49:21.620] you'd expect to potentially not be normalized,

[00:49:23.940] and this is what makes it work.

[00:49:27.420] Pretty subtle, and hopefully this helps to scare you,

[00:49:32.260] that you should have respect for broadcasting,

[00:49:34.300] be careful, check your work,

[00:49:36.140] and understand how it works under the hood,

[00:49:38.820] and make sure that it's broadcasting

[00:49:39.980] in the direction that you like.

[00:49:41.260] Otherwise, you're going to introduce very subtle bugs,

[00:49:43.260] very hard to find bugs, and just be careful.

[00:49:46.580] One more note on efficiency.

[00:49:48.180] We don't want to be doing this here because this

[00:49:51.220] creates a completely new tensor that we store into p.

[00:49:54.220] We prefer to use in-place operations if possible.

[00:49:57.620] This would be an in-place operation,

[00:50:00.140] has the potential to be faster.

[00:50:01.700] It doesn't create new memory under the hood.

[00:50:04.420] Then let's erase this, we don't need it.

[00:50:07.660] Let's also just do fewer,

[00:50:12.740] just so I'm not wasting space.

[00:50:14.540] We're actually in a pretty good spot now.

[00:50:16.780] We trained a bigram language model,

[00:50:19.180] and we trained it really just by counting

[00:50:21.780] how frequently any pairing occurs and then

[00:50:24.700] normalizing so that we get a nice property distribution.

[00:50:27.780] Really these elements of this array p are

[00:50:31.380] really the parameters of our bigram language model,

[00:50:33.520] giving us and summarizing the statistics of these bigrams.

[00:50:36.740] We trained the model and then we know how to sample from the model.

[00:50:40.100] We just iteratively sampled

[00:50:42.460] the next character and feed it in each time and get a next character.

[00:50:46.600] Now what I'd like to do is I'd like to somehow

[00:50:48.940] evaluate the quality of this model.

[00:50:50.940] We'd like to somehow summarize

[00:50:52.780] the quality of this model into a single number.

[00:50:55.260] How good is it at predicting the training set as an example?

[00:51:00.260] In the training set, we can evaluate now

[00:51:02.380] the training loss and this training loss is telling us

[00:51:06.020] about the quality of this model in

[00:51:08.220] a single number just like we saw in micrograd.

[00:51:11.540] Let's try to think through the quality of

[00:51:13.780] the model and how we would evaluate it.

[00:51:16.140] Basically, what we're going to do is we're going to copy-paste

[00:51:19.460] this code that we previously used for counting.

[00:51:23.420] Let me just print these bigrams first.

[00:51:26.020] We're going to use fstrings and I'm going to

[00:51:28.340] print character 1 followed by character 2,

[00:51:30.680] these are the bigrams and then I don't want

[00:51:32.660] to do it for all the words, just the first three words.

[00:51:35.500] Here we have Emma, Olivia, and Ava bigrams.

[00:51:39.620] Now what we'd like to do is we'd like to basically look at

[00:51:43.380] the probability that the model

[00:51:45.140] assigns to every one of these bigrams.

[00:51:47.940] In other words, we can look at the probability,

[00:51:50.220] which is summarized in the matrix P of ix1,

[00:51:53.660] ix2 and then we can print it here as probability.

[00:51:59.980] Because these probabilities are way too large,

[00:52:02.380] let me percent or column 0.4f to truncate it a bit.

[00:52:08.660] What do we have here? We're looking at the probabilities that

[00:52:11.740] the model assigns to every one of these bigrams in the dataset.

[00:52:15.060] We can see some of them are 4 percent,

[00:52:17.140] 3 percent, etc, just to have a measuring stick in our mind,

[00:52:20.300] by the way, with 27 possible characters or tokens.

[00:52:24.740] If everything was equally likely,

[00:52:26.600] then you'd expect all these probabilities to

[00:52:29.100] be 4 percent roughly.

[00:52:32.220] Anything above 4 percent means that we've learned

[00:52:34.940] something useful from these bigram statistics.

[00:52:37.500] You see that roughly some of these are 4 percent,

[00:52:39.580] but some of them are as high as 40 percent,

[00:52:41.660] 35 percent, and so on.

[00:52:43.900] You see that the model actually assigned

[00:52:45.340] a pretty high probability to whatever's in the training set,

[00:52:48.200] and so that's a good thing.

[00:52:50.100] Basically, if you have a very good model,

[00:52:52.380] you'd expect that these probabilities should be near one,

[00:52:54.800] because that means that your model

[00:52:56.620] is correctly predicting what's going to come next,

[00:52:58.700] especially on the training set where you train your model.

[00:53:02.700] Now we'd like to think about how can we summarize

[00:53:05.900] these probabilities into a single number

[00:53:08.420] that measures the quality of this model.

[00:53:11.020] Now, when you look at the literature into

[00:53:13.300] maximum likelihood estimation and statistical modeling and so on,

[00:53:17.020] you'll see that what's typically used

[00:53:18.780] here is something called the likelihood,

[00:53:21.180] and the likelihood is the product of all of these probabilities.

[00:53:25.660] The product of all of these probabilities is

[00:53:28.300] the likelihood and it's really telling us about

[00:53:30.980] the probability of the entire data set

[00:53:33.340] assigned by the model that we've trained,

[00:53:37.500] and that is a measure of quality.

[00:53:39.300] The product of these should be as high as possible.

[00:53:43.020] When you are training the model and when you have a good model,

[00:53:45.860] your product of these probabilities should be very high.

[00:53:49.420] Now, because the product of these probabilities

[00:53:52.260] is an unwieldy thing to work with,

[00:53:54.220] you can see that all of them are between zero and one,

[00:53:56.220] so your product of these probabilities will be a very tiny number.

[00:54:00.140] So for convenience,

[00:54:02.740] what people work with usually is not the likelihood,

[00:54:04.900] but they work with what's called the log likelihood.

[00:54:07.420] So the product of these is the likelihood.

[00:54:10.780] To get the log likelihood,

[00:54:12.380] we just have to take the log of the probability.

[00:54:14.900] The log of the probability here,

[00:54:17.140] the log of x from zero to one,

[00:54:19.300] the log is a, you see here,

[00:54:21.540] monotonic transformation of the probability,

[00:54:24.340] where if you pass in one, you get zero.

[00:54:28.620] So probability one gets you log probability of zero.

[00:54:32.020] And then as you go lower and lower probability,

[00:54:34.380] the log will grow more and more negative

[00:54:36.300] until all the way to negative infinity at zero.

[00:54:41.660] So here we have a lock prob,

[00:54:43.580] which is really just a torch dot log of probability.

[00:54:46.700] Let's print it out to get a sense of what that looks like.

[00:54:49.780] Lock prob, also 0.4f.

[00:54:54.860] So as you can see,

[00:54:57.820] when we plug in numbers that are very close,

[00:55:00.340] some of our higher numbers,

[00:55:01.420] we get closer and closer to zero.

[00:55:03.260] And then if we plug in very bad probabilities,

[00:55:05.820] we get more and more negative number.

[00:55:07.540] That's bad.

[00:55:09.020] So, and the reason we work with this is

[00:55:12.380] for large extent convenience, right?

[00:55:15.060] Because we have mathematically that if you have

[00:55:17.260] some product a times b times c of all these probabilities,

[00:55:20.340] right, the likelihood is the product

[00:55:23.540] of all these probabilities.

[00:55:25.420] Then the log of these is just log of a plus log of b

[00:55:33.860] plus log of c.

[00:55:35.300] If you remember your logs from your high school

[00:55:38.100] or undergrad and so on.

[00:55:39.860] So we have that basically,

[00:55:41.620] the likelihood is the product of probabilities.

[00:55:43.380] The log likelihood is just the sum of the logs

[00:55:46.300] of the individual probabilities.

[00:55:48.860] So log likelihood starts at zero.

[00:55:54.700] And then log likelihood here,

[00:55:56.860] we can just accumulate simply.

[00:56:00.500] And then at the end, we can print this.

[00:56:05.500] Print the log likelihood.

[00:56:09.700] F strings.

[00:56:11.940] Maybe you're familiar with this.

[00:56:14.020] So log likelihood is negative 38.

[00:56:20.060] Okay.

[00:56:21.380] Now we actually want,

[00:56:25.260] so how high can log likelihood get?

[00:56:27.860] It can go to zero.

[00:56:29.900] So when all the probabilities are one,

[00:56:31.460] log likelihood will be zero.

[00:56:32.940] And then when all the probabilities are lower,

[00:56:34.980] this will grow more and more negative.

[00:56:37.540] Now, we don't actually like this

[00:56:39.260] because what we'd like is a loss function.

[00:56:41.740] And a loss function has the semantics

[00:56:43.700] that low is good

[00:56:46.300] because we're trying to minimize the loss.

[00:56:48.220] So we actually need to invert this.

[00:56:50.380] And that's what gives us something called

[00:56:52.340] the negative log likelihood.

[00:56:56.020] Negative log likelihood is just negative

[00:56:58.660] of the log likelihood.

[00:57:03.980] These are F strings, by the way,

[00:57:04.980] if you'd like to look this up.

[00:57:06.580] Negative log likelihood equals.

[00:57:09.460] So negative log likelihood now is just the negative of it.

[00:57:12.140] And so the negative log likelihood is a very nice

[00:57:14.540] loss function because the lowest it can get is zero.

[00:57:19.820] And the higher it is,

[00:57:21.100] the worse off the predictions are that you're making.

[00:57:24.780] And then one more modification to this

[00:57:26.180] that sometimes people do is that for convenience,

[00:57:29.260] they actually like to normalize by,

[00:57:31.260] they like to make it an average instead of a sum.

[00:57:34.460] And so here, let's just keep some counts as well.

[00:57:39.300] So n plus equals one starts at zero.

[00:57:42.860] And then here we can have sort of like

[00:57:45.420] a normalized log likelihood.

[00:57:50.580] If we just normalize it by the count,

[00:57:52.500] then we will sort of get the average log likelihood.

[00:57:55.820] So this would be usually our loss function here

[00:57:58.900] is what this is what we would use.

[00:58:02.340] So our loss function for the training set

[00:58:04.300] assigned by the model is 2.4.

[00:58:06.500] That's the quality of this model.

[00:58:08.580] And the lower it is, the better off we are.

[00:58:10.780] And the higher it is, the worse off we are.

[00:58:13.380] And the job of our training is to find the parameters

[00:58:18.420] that minimize the negative log likelihood loss.

[00:58:22.820] And that would be like a high quality model.

[00:58:25.460] Okay, so to summarize, I actually wrote it out here.

[00:58:28.060] So our goal is to maximize likelihood,

[00:58:30.860] which is the product of all the probabilities

[00:58:34.140] assigned by the model.

[00:58:35.700] And we want to maximize this likelihood

[00:58:37.740] with respect to the model parameters.

[00:58:39.780] And in our case, the model parameters here

[00:58:42.020] are defined in the table.

[00:58:43.540] These numbers, the probabilities are the model parameters

[00:58:47.460] sort of in our Barygram language model so far.

[00:58:50.140] But you have to keep in mind that here we are storing

[00:58:52.420] everything in a table format, the probabilities.

[00:58:54.860] But what's coming up as a brief preview

[00:58:57.020] is that these numbers will not be kept explicitly,

[00:59:00.220] but these numbers will be calculated by a neural network.

[00:59:03.180] So that's coming up.

[00:59:04.660] And we want to change and tune the parameters

[00:59:06.900] of these neural networks.

[00:59:08.180] We want to change these parameters

[00:59:09.580] to maximize the likelihood,

[00:59:11.020] the product of the probabilities.

[00:59:13.420] Now, maximizing the likelihood

[00:59:15.100] is equivalent to maximizing the log likelihood

[00:59:17.300] because log is a monotonic function.

[00:59:19.980] Here's the graph of log.

[00:59:22.140] And basically all it is doing is it's just scaling your,

[00:59:26.820] you can look at it as just a scaling of the loss function.

[00:59:29.460] And so the optimization problem here and here

[00:59:33.100] are actually equivalent because this is just scaling.

[00:59:35.620] You can look at it that way.

[00:59:37.100] And so these are two identical optimization problems.

[00:59:41.980] Maximizing the log likelihood is equivalent

[00:59:43.660] to minimizing the negative log likelihood.

[00:59:46.260] And then in practice, people actually minimize

[00:59:48.020] the average negative log likelihood

[00:59:50.300] to get numbers like 2.4.

[00:59:53.020] And then this summarizes the quality of your model.

[00:59:56.300] And we'd like to minimize it and make it

[00:59:57.940] as small as possible.

[00:59:59.700] And the lowest it can get is zero.

[01:00:02.420] And the lower it is, the better off your model is

[01:00:05.780] because it's assigning high probabilities to your data.

[01:00:09.580] Now let's estimate the probability

[01:00:10.820] over the entire training set,

[01:00:11.860] just to make sure that we get something around 2.4.

[01:00:14.940] Let's run this over the entire, oops.

[01:00:17.380] Let's take out the print segment as well.

[01:00:20.820] Okay, 2.45 over the entire training set.

[01:00:24.540] Now what I'd like to show you is that you can actually

[01:00:26.220] evaluate the probability for any word that you want.

[01:00:28.340] Like for example, if we just test a single word, Andre,

[01:00:32.860] and bring back the print statement,

[01:00:35.900] then you see that Andre is actually kind of like

[01:00:37.460] an unlikely word, or like on average,

[01:00:40.740] we take three log probability to represent it.

[01:00:44.420] And roughly that's because Ej apparently

[01:00:46.460] is very uncommon as an example.

[01:00:50.060] Now think through this.

[01:00:53.900] When I take Andre and I append q,

[01:00:55.740] and I test the probability of it under aq,

[01:01:00.300] we actually get infinity.

[01:01:03.060] And that's because jq has a 0% probability

[01:01:06.340] according to our model.

[01:01:07.620] So the log likelihood,

[01:01:09.380] so the log of zero will be negative infinity.

[01:01:12.180] We get infinite loss.

[01:01:14.580] So this is kind of undesirable, right?

[01:01:15.780] Because we plugged in a string

[01:01:16.820] that could be like a somewhat reasonable name.

[01:01:19.220] But basically what this is saying is that this model

[01:01:21.220] is exactly 0% likely to predict this name.

[01:01:26.500] And our loss is infinity on this example.

[01:01:29.740] And really what the reason for that is that j

[01:01:32.860] is followed by q zero times.

[01:01:37.180] Where's q?

[01:01:38.020] jq is zero.

[01:01:39.340] And so jq is 0% likely.

[01:01:42.260] So it's actually kind of gross

[01:01:43.620] and people don't like this too much.

[01:01:45.260] To fix this, there's a very simple fix

[01:01:46.980] that people like to do to sort of like

[01:01:49.020] smooth out your model a little bit.

[01:01:50.300] And it's called model smoothing.

[01:01:52.140] And roughly what's happening is that

[01:01:53.420] we will add some fake counts.

[01:01:56.260] So imagine adding a count of one to everything.

[01:02:00.980] So we add a count of one like this,

[01:02:04.660] and then we recalculate the probabilities.

[01:02:07.820] And that's model smoothing.

[01:02:08.980] And you can add as much as you like.

[01:02:10.180] You can add five and that will give you a smoother model.

[01:02:12.900] And the more you add here,

[01:02:14.780] the more uniform model you're gonna have.

[01:02:17.780] And the less you add, the more peaked model

[01:02:20.820] you are gonna have, of course.

[01:02:22.340] So one is like a pretty decent count to add.

[01:02:25.780] And that will ensure that there will be no zeros

[01:02:28.380] in our probability matrix P.

[01:02:30.860] And so this will of course change the generations

[01:02:32.620] a little bit.

[01:02:33.700] In this case it didn't, but in principle it could.

[01:02:36.580] But what that's gonna do now is that

[01:02:38.380] nothing will be infinity unlikely.

[01:02:41.180] So now our model will predict some other probability.

[01:02:44.820] And we see that jq now has a very small probability

[01:02:47.740] so the model still finds it very surprising

[01:02:49.380] that this was a word or a bigram,

[01:02:51.660] but we don't get negative infinity.

[01:02:53.500] So it's kind of like a nice fix

[01:02:54.540] that people like to apply sometimes

[01:02:55.820] and it's called model smoothing.

[01:02:57.140] Okay, so we've now trained a respectable bigram

[01:03:00.060] character level language model.

[01:03:01.540] And we saw that we both sort of trained the model

[01:03:05.540] by looking at the counts of all the bigrams

[01:03:07.860] and normalizing the rows to get probability distributions.

[01:03:11.580] We saw that we can also then use those parameters

[01:03:14.820] of this model to perform sampling of new words.

[01:03:19.580] So we sample new names according to those distributions.

[01:03:22.300] And we also saw that we can evaluate

[01:03:23.860] the quality of this model.

[01:03:25.540] And the quality of this model is summarized

[01:03:27.060] in a single number, which is the negative log likelihood.

[01:03:30.020] And the lower this number is, the better the model is

[01:03:33.380] because it is giving high probabilities

[01:03:35.660] to the actual next characters

[01:03:37.420] in all the bigrams in our training set.

[01:03:40.220] So that's all well and good,

[01:03:42.100] but we've arrived at this model explicitly

[01:03:44.420] by doing something that felt sensible.

[01:03:46.220] We were just performing counts

[01:03:48.140] and then we were normalizing those counts.

[01:03:51.100] Now, what I would like to do is I would like

[01:03:52.380] to take an alternative approach.

[01:03:54.220] We will end up in a very, very similar position,

[01:03:56.580] but the approach will look very different

[01:03:58.300] because I would like to cast the problem

[01:03:59.980] of bigram character level language modeling

[01:04:01.980] into the neural network framework.

[01:04:04.340] And in the neural network framework,

[01:04:05.460] we're going to approach things slightly differently,

[01:04:08.260] but again, end up in a very similar spot.

[01:04:10.380] I'll go into that later.

[01:04:12.100] Now, our neural network is going to be

[01:04:15.060] still a bigram character level language model.

[01:04:17.340] So it receives a single character as an input,

[01:04:20.500] then there's neural network with some weights

[01:04:22.380] or some parameters W,

[01:04:24.300] and it's going to output the probability distribution

[01:04:27.420] over the next character in a sequence.

[01:04:29.260] It's going to make guesses as to what is likely

[01:04:31.660] to follow this character that was input to the model.

[01:04:36.060] And then in addition to that,

[01:04:37.620] we're going to be able to evaluate any setting

[01:04:39.700] of the parameters of the neural net

[01:04:41.460] because we have the loss function,

[01:04:43.740] the negative log likelihood.

[01:04:45.140] So we're going to take a look at its probability distributions

[01:04:47.500] and we're going to use the labels,

[01:04:50.020] which are basically just the identity

[01:04:51.740] of the next character in that bigram, the second character.

[01:04:54.820] So knowing what second character actually comes next

[01:04:57.380] in the bigram allows us to then look at

[01:05:00.340] how high of probability the model assigns

[01:05:02.620] to that character.

[01:05:03.980] And then we of course want the probability to be very high.

[01:05:07.100] And that is another way of saying that the loss is low.

[01:05:10.020] So we're going to use gradient-based optimization then

[01:05:12.780] to tune the parameters of this network

[01:05:14.820] because we have the loss function

[01:05:16.460] and we're going to minimize it.

[01:05:17.860] So we're going to tune the weights

[01:05:19.620] so that the neural net is correctly predicting

[01:05:21.580] the probabilities for the next character.

[01:05:23.780] So let's get started.

[01:05:25.020] The first thing I want to do is I want to compile

[01:05:26.860] the training set of this neural network, right?

[01:05:28.940] So create the training set of all the bigrams, okay?

[01:05:33.940] And here, I'm going to copy-paste this code

[01:05:39.940] because this code iterates over all the bigrams.

[01:05:43.420] So here we start with the words,

[01:05:45.020] we iterate over all the bigrams,

[01:05:46.580] and previously, as you recall, we did the counts.

[01:05:49.180] But now we're not going to do counts,

[01:05:50.580] we're just creating a training set.

[01:05:52.860] Now this training set will be made up of two lists.

[01:05:58.100] We have the bigrams,

[01:06:00.380] we have the inputs and the targets, the labels.

[01:06:09.620] And these bigrams will denote x, y,

[01:06:11.460] those are the characters, right?

[01:06:13.220] And so we're given the first character of the bigram

[01:06:15.700] and then we're trying to predict the next one.

[01:06:17.820] Both of these are going to be integers.

[01:06:19.380] So here we'll take x's dot append is just x1,

[01:06:24.580] y's dot append is x2.

[01:06:26.260] And then here, we actually don't want lists of integers,

[01:06:30.220] we will create tensors out of these.

[01:06:32.380] So x's is torch dot tensor of x's,

[01:06:36.300] and y's is torch dot tensor of y's.

[01:06:40.260] And then we don't actually want to take all the words

[01:06:42.780] just yet because I want everything to be manageable.

[01:06:45.700] So let's just do the first word, which is emma.

[01:06:50.060] And then it's clear what these x's and y's would be.

[01:06:52.700] Here, let me print character one, character two,

[01:06:56.300] just so you see what's going on here.

[01:06:59.140] So the bigrams of these characters is.emmmma.

[01:07:06.140] So this single word, as I mentioned,

[01:07:07.860] has one, two, three, four, five examples

[01:07:10.300] for our neural network.

[01:07:12.140] There are five separate examples in emma.

[01:07:14.860] And those examples are summarized here.

[01:07:16.620] When the input to the neural network is integer zero,

[01:07:20.220] the desired label is integer five, which corresponds to E.

[01:07:25.220] When the input to the neural network is five,

[01:07:27.460] we want its weights to be arranged

[01:07:29.140] so that 13 gets a very high probability.

[01:07:32.260] When 13 is put in, we want 13 to have a high probability.

[01:07:36.340] When 13 is put in,

[01:07:37.380] we also want one to have a high probability.

[01:07:40.660] When one is input,

[01:07:41.820] we want zero to have a very high probability.

[01:07:44.620] So there are five separate input examples,

[01:07:46.820] to a neural net, in this data set.

[01:07:52.180] I wanted to add a tangent of a note of caution

[01:07:55.100] to be careful with a lot of the APIs

[01:07:56.700] of some of these frameworks.

[01:07:58.540] You saw me silently use torch.tensor with a lowercase T,

[01:08:02.940] and the output looked right.

[01:08:04.860] But you should be aware that there's actually two ways

[01:08:07.260] of constructing a tensor.

[01:08:08.860] There's a torch.lowercaseTensor,

[01:08:11.020] and there's also a torch.capitalTensor class.

[01:08:14.580] Which you can also construct.

[01:08:16.780] So you can actually call both.

[01:08:18.180] You can also do torch.capitalTensor,

[01:08:20.940] and you get an Xs and Ys as well.

[01:08:23.340] So that's not confusing at all.

[01:08:26.980] There are threads on what is the difference

[01:08:28.020] between these two.

[01:08:29.420] And unfortunately, the docs are just like not clear

[01:08:33.380] on the difference.

[01:08:34.220] And when you look at the docs of lowercaseTensor,

[01:08:37.180] constructs tensor with no autograd history by copying data.

[01:08:41.620] It's just like, it doesn't have a lot to do with this.

[01:08:44.380] It doesn't make sense.

[01:08:46.740] So the actual difference, as far as I can tell,

[01:08:48.500] is explained eventually in this random thread

[01:08:50.380] that you can Google.

[01:08:51.660] And really it comes down to, I believe,

[01:08:55.100] that, where is this?

[01:08:58.660] Torch.tensor infers the D type, the data type, automatically,

[01:09:01.820] while torch.tensor just returns a float tensor.

[01:09:04.500] I would recommend stick to torch.lowercaseTensor.

[01:09:07.780] So indeed, we see that when I construct this

[01:09:12.260] with a capital T, the data type here of Xs is float 32.

[01:09:18.260] But torch.lowercaseTensor,

[01:09:21.260] you see how it's now X.D type is now integer.

[01:09:26.820] So it's advised that you use lowercase T

[01:09:30.900] and you can read more about it if you like

[01:09:32.540] in some of these threads.

[01:09:34.380] But basically, I'm pointing out some of these things

[01:09:38.260] because I want to caution you

[01:09:39.340] and I want you to get used to reading

[01:09:41.580] a lot of documentation

[01:09:42.740] and reading through a lot of Q and As

[01:09:45.180] and threads like this.

[01:09:47.060] And some of this stuff is unfortunately not easy

[01:09:50.220] and not very well documented

[01:09:51.260] and you have to be careful out there.

[01:09:52.740] What we want here is integers

[01:09:54.740] because that's what makes sense.

[01:09:58.140] And so lowercaseTensor is what we are using.

[01:10:01.340] Okay, now we want to think through

[01:10:02.580] how we're going to feed in these examples

[01:10:04.660] into a neural network.

[01:10:06.380] Now, it's not quite as straightforward as plugging it in

[01:10:10.180] because these examples right now are integers.

[01:10:12.340] So there's like a zero, five, or 13.

[01:10:14.740] It gives us the index of the character

[01:10:16.620] and you can't just plug an integer index

[01:10:18.460] into a neural net.

[01:10:20.100] These neural nets are sort of made up of these neurons

[01:10:24.540] and these neurons have weights.

[01:10:26.940] And as you saw in microGrad,

[01:10:28.580] these weights act multiplicatively on the inputs,

[01:10:31.060] WX plus B, there's 10HS and so on.

[01:10:34.340] And so it doesn't really make sense

[01:10:35.380] to make an input neuron take on integer values

[01:10:37.860] that you feed in and then multiply on with weights.

[01:10:41.780] So instead, a common way of encoding integers

[01:10:44.660] is what's called one-hot encoding.

[01:10:47.100] In one-hot encoding, we take an integer like 13

[01:10:50.860] and we create a vector that is all zeros

[01:10:53.700] except for the 13th dimension, which we turn to a one.

[01:10:57.540] And then that vector can feed into a neural net.

[01:11:01.180] Now, conveniently, PyTorch actually has something

[01:11:04.780] called the one-hot function inside torch and in functional.

[01:11:10.420] It takes a tensor made up of integers.

[01:11:14.980] Long is an integer.

[01:11:19.300] And it also takes a number of classes,

[01:11:22.660] which is how large you want your vector to be.

[01:11:27.820] So here, let's import torch. and in.functional.sf.

[01:11:31.860] This is a common way of importing it.

[01:11:34.180] And then let's do f.one-hot.

[01:11:36.700] And we feed in the integers that we want to encode.

[01:11:39.980] So we can actually feed in the entire array of Xs.

[01:11:44.060] And we can tell it that num classes is 27.

[01:11:47.780] So it doesn't have to try to guess it.

[01:11:49.420] It may have guessed that it's only 13

[01:11:51.620] and would give us an incorrect result.

[01:11:54.700] So this is the one-hot.

[01:11:55.820] Let's call this xinc for xencoded.

[01:11:59.660] And then we see that xencoded.shape is five by 27.

[01:12:05.060] And we can also visualize it, plt.imshow of xinc,

[01:12:10.460] to make it a little bit more clear

[01:12:11.620] because this is a little messy.

[01:12:13.420] So we see that we've encoded

[01:12:14.780] all the five examples into vectors.

[01:12:18.500] We have five examples, so we have five rows.

[01:12:20.620] And each row here is now an example into a neural net.

[01:12:24.180] And we see that the appropriate bit is turned on as a one.

[01:12:27.900] And everything else is zero.

[01:12:29.820] So here, for example, the zeroth bit is turned on.

[01:12:34.540] The fifth bit is turned on.

[01:12:36.300] 13th bits are turned on for both of these examples.

[01:12:39.500] And the first bit here is turned on.

[01:12:42.780] So that's how we can encode integers into vectors.

[01:12:47.420] And then these vectors can feed in to neural nets.

[01:12:50.060] One more issue to be careful with here, by the way,

[01:12:51.940] is let's look at the data type of the encoding.

[01:12:55.020] We always want to be careful

[01:12:56.260] with data types. What would you expect

[01:12:58.900] xencoding's data type to be?

[01:13:01.060] When we're plugging numbers into neural nets,

[01:13:03.140] we don't want them to be integers.

[01:13:04.460] We want them to be floating-point numbers

[01:13:06.100] that can take on various values.

[01:13:08.660] But the dtype here is actually 64-bit integer.

[01:13:12.500] And the reason for that, I suspect,

[01:13:13.940] is that one-hot received a 64-bit integer here,

[01:13:17.820] and it returned the same data type.

[01:13:20.060] And when you look at the signature of one-hot,

[01:13:21.860] it doesn't even take a dtype, a desired data type,

[01:13:24.820] of the output tensor.

[01:13:26.620] And so we can't, in a lot of functions in Torch,

[01:13:29.140] we'd be able to do something like dtype equals

[01:13:30.940] torch.float32, which is what we want,

[01:13:33.820] but one-hot does not support that.

[01:13:36.060] So instead, we're going to want to cast this

[01:13:38.300] to float like this,

[01:13:41.500] so that these, everything is the same,

[01:13:44.700] everything looks the same, but the dtype is float32.

[01:13:48.180] And floats can feed into neural nets.

[01:13:51.500] So now let's construct our first neural net.

[01:13:53.580] So now let's construct our first neuron.

[01:13:56.260] This neuron will look at these input vectors.

[01:14:00.220] And as you remember from microGrad,

[01:14:02.140] these neurons basically perform a very simple function,

[01:14:04.460] wx plus b, where wx is a dot product, right?

[01:14:09.740] So we can achieve the same thing here.

[01:14:12.220] Let's first define the weights of this neuron, basically.

[01:14:15.140] What are the initial weights at initialization

[01:14:17.580] for this neuron?

[01:14:19.020] Let's initialize them with torch.randin.

[01:14:21.900] Torch.randin fills a tensor with random numbers

[01:14:27.260] drawn from a normal distribution.

[01:14:29.380] And a normal distribution has a probability density function

[01:14:33.660] like this.

[01:14:34.540] And so most of the numbers drawn from this distribution

[01:14:37.140] will be around zero, but some of them will be as high

[01:14:40.660] as almost three and so on.

[01:14:42.060] And very few numbers will be above three in magnitude.

[01:14:46.460] So we need to take a size as an input here.

[01:14:50.540] And I'm going to use size as to be 27 by one.

[01:14:54.660] So 27 by one, and then let's visualize w.

[01:14:58.660] So w is a column vector of 27 numbers.

[01:15:03.100] And these weights are then multiplied by the inputs.

[01:15:08.660] So now to perform this multiplication,

[01:15:10.740] we can take X encoding and we can multiply it with w.

[01:15:15.100] This is a matrix multiplication operator in PyTorch.

[01:15:18.420] And the output of this operation is five by one.

[01:15:22.420] The reason it's five by one is the following.

[01:15:24.660] We took X encoding, which is five by 27,

[01:15:27.940] and we multiplied it by 27 by one.

[01:15:32.420] And in matrix multiplication, you see that the output

[01:15:36.620] will become five by one because these 27 will multiply

[01:15:41.620] and add.

[01:15:43.700] So basically what we're seeing here out of this operation

[01:15:47.180] is we are seeing the five activations of this neuron

[01:15:55.140] on these five inputs.

[01:15:57.020] And we've evaluated all of them in parallel.

[01:15:59.220] We didn't feed in just a single input to the single neuron.

[01:16:02.100] We fed in simultaneously all the five inputs

[01:16:05.100] into the same neuron.

[01:16:06.820] And in parallel, PyTorch has evaluated the wx plus b,

[01:16:11.820] but here it's just wx, there's no bias.

[01:16:14.700] It has evaluated w times x for all of them independently.

[01:16:19.700] Now, instead of a single neuron though,

[01:16:21.180] I would like to have 27 neurons.

[01:16:23.180] And I'll show you in a second why I want 27 neurons.

[01:16:26.660] So instead of having just a one here,

[01:16:28.580] which is indicating this presence of one single neuron,

[01:16:31.620] we can use 27.

[01:16:33.740] And then when w is 27 by 27,

[01:16:37.300] this will, in parallel, evaluate all the 27 neurons

[01:16:42.300] on all the five inputs.

[01:16:44.220] Giving us a much better, much, much bigger result.

[01:16:47.700] So now what we've done is five by 27 multiplied 27 by 27.

[01:16:52.140] And the output of this is now five by 27.

[01:16:55.780] So we can see that the shape of this is five by 27.

[01:17:01.900] So what is every element here telling us, right?

[01:17:05.140] It's telling us for every one of 27 neurons that we created,

[01:17:09.140] what is the firing rate of those neurons

[01:17:12.620] on every one of those five examples?

[01:17:15.460] So the element, for example, three comma 13,

[01:17:21.260] is giving us the firing rate of the 13th neuron

[01:17:25.260] looking at the third input.

[01:17:27.940] And the way this was achieved is by a dot product

[01:17:32.380] between the third input and the third input.

[01:17:37.380] And the 13th column of this w matrix here.

[01:17:44.380] So using matrix multiplication,

[01:17:46.860] we can very efficiently evaluate the dot product

[01:17:50.780] between lots of input examples in a batch

[01:17:54.180] and lots of neurons where all of those neurons have weights

[01:17:57.820] in the columns of those w's.

[01:18:00.340] And in matrix multiplication,

[01:18:01.300] we're just doing those dot products in parallel.

[01:18:04.660] Just to show you that this is the case,

[01:18:06.500] we can take xnk and we can take the third row.

[01:18:11.540] And we can take the w and take its 13th column.

[01:18:16.740] And then we can do xnk at three,

[01:18:20.980] elementwise multiply with w at 13,

[01:18:26.140] and sum that up.

[01:18:27.260] That's wx plus b.

[01:18:29.100] Well, there's no plus b, it's just wx dot product.

[01:18:32.300] And that's this number.

[01:18:34.740] So you see that this is just being done efficiently

[01:18:37.540] by the matrix multiplication operation

[01:18:39.940] for all the input examples

[01:18:41.660] and for all the output neurons of this first layer.

[01:18:45.580] Okay, so we fed our 27 dimensional inputs

[01:18:48.820] into a first layer of a neural net that has 27 neurons.

[01:18:52.380] So we have 27 inputs and now we have 27 neurons.

[01:18:56.540] These neurons perform w times x.

[01:18:59.300] They don't have a bias

[01:19:00.460] and they don't have a non-linearity like 10h.

[01:19:02.740] We're going to leave them to be a linear layer.

[01:19:06.140] In addition to that,

[01:19:07.020] we're not going to have any other layers.

[01:19:08.660] This is going to be it.

[01:19:09.580] It's just going to be the dumbest, smallest,

[01:19:12.260] simplest neural net, which is just a single linear layer.

[01:19:16.020] And now I'd like to explain

[01:19:17.100] what I want those 27 outputs to be.

[01:19:20.820] Intuitively, what we're trying to produce here

[01:19:22.340] for every single input example

[01:19:24.180] is we're trying to produce

[01:19:25.060] some kind of a probability distribution

[01:19:26.780] for the next character in a sequence.

[01:19:28.860] And there's 27 of them.

[01:19:31.140] But we have to come up with precise semantics

[01:19:33.780] for exactly how we're going to interpret

[01:19:35.860] these 27 numbers that these neurons take on.

[01:19:39.780] Now intuitively, you see here that these numbers are negative

[01:19:43.180] and some of them are positive, et cetera.

[01:19:45.260] And that's because these are coming out

[01:19:46.580] of a neural net layer initialized

[01:19:48.820] with these normal distribution parameters.

[01:19:54.460] But what we want is we want something like we had here.

[01:19:57.340] Like each row here told us the counts

[01:20:00.940] and then we normalize the counts to get probabilities.

[01:20:03.580] And we want something similar to come out of a neural net.

[01:20:06.580] But what we just have right now

[01:20:07.780] is just some negative and positive numbers.

[01:20:10.660] Now we want those numbers to somehow represent

[01:20:12.860] the probabilities for the next character.

[01:20:15.420] But you see that probabilities,

[01:20:16.940] they have a special structure.

[01:20:18.860] They're positive numbers and they sum to one.

[01:20:22.940] And so that doesn't just come out of a neural net.

[01:20:25.860] And then they can't be counts

[01:20:27.860] because these counts are positive

[01:20:30.700] and counts are integers.

[01:20:32.820] So counts are also not really a good thing

[01:20:34.820] to output from a neural net.

[01:20:36.780] So instead what the neural net is going to output

[01:20:38.980] and how we are going to interpret the 27 numbers

[01:20:44.140] is that these 27 numbers are giving us log counts, basically.

[01:20:50.500] So instead of giving us counts directly,

[01:20:53.020] like in this table, they're giving us log counts.

[01:20:56.140] And to get the counts, we're going to take the log counts

[01:20:59.020] and we're going to exponentiate them.

[01:21:01.420] Now, exponentiation takes the following form.

[01:21:07.260] It takes numbers that are negative or they are positive.

[01:21:11.020] It takes the entire real line.

[01:21:12.940] And then if you plug in negative numbers,

[01:21:14.900] you're going to get e to the x, which is always below one.

[01:21:20.740] So you're getting numbers lower than one.

[01:21:23.580] And if you plug in numbers greater than zero,

[01:21:26.020] you're getting numbers greater than one

[01:21:28.420] all the way growing to the infinity.

[01:21:30.980] And this here grows to zero.

[01:21:33.380] So basically we're going to take these numbers here

[01:21:40.420] and instead of them being positive and negative

[01:21:45.060] and all over the place,

[01:21:46.220] we're going to interpret them as log counts.

[01:21:48.900] And then we're going to element-wise

[01:21:50.260] exponentiate these numbers.

[01:21:52.860] Exponentiating them now gives us something like this.

[01:21:56.580] And you see that these numbers now

[01:21:57.780] because they went through an exponent,

[01:21:59.420] all the negative numbers turned into numbers below one,

[01:22:02.260] like 0.338.

[01:22:04.340] And all the positive numbers originally turned into

[01:22:07.260] even more positive numbers, sort of greater than one.

[01:22:10.900] So like for example, seven is some positive number

[01:22:16.580] over here that is greater than zero.

[01:22:21.180] But exponentiated outputs here basically give us something

[01:22:26.020] that we can use and interpret as the equivalent

[01:22:28.860] of counts originally.

[01:22:31.060] So you see these counts here, 112, 751, 1, et cetera.

[01:22:36.500] The neural net is kind of now predicting counts.

[01:22:41.700] And these counts are positive numbers.

[01:22:44.060] They can never be below zero.

[01:22:45.500] So that makes sense.

[01:22:46.820] And they can now take on various values

[01:22:49.780] depending on the settings of W.

[01:22:52.860] So let me break this down.

[01:22:54.900] We're going to interpret these to be the log counts.

[01:22:59.940] In other words, for this,

[01:23:01.020] that is often used is so-called logits.

[01:23:03.940] These are logits, log counts.

[01:23:07.380] And these will be sort of the counts, logits exponentiated.

[01:23:12.100] And this is equivalent to the n matrix,

[01:23:14.980] sort of the n array that we used previously.

[01:23:18.540] Remember this was the n.

[01:23:20.380] This is the array of counts.

[01:23:22.940] And each row here are the counts for the next character.

[01:23:31.620] So those are the counts.

[01:23:33.180] And now the probabilities are just the counts normalized.

[01:23:38.540] And so I'm not going to find the same,

[01:23:41.780] but basically I'm not going to scroll all over the place.

[01:23:44.820] We've already done this.

[01:23:46.300] We want to counts.sum along the first dimension,

[01:23:50.660] and we want to keep them as true.

[01:23:53.780] We've went over this,

[01:23:55.140] and this is how we normalize the rows of our counts matrix

[01:23:58.900] to get our probabilities.

[01:24:02.260] Props.

[01:24:03.980] So now these are the probabilities.

[01:24:06.860] And these are the counts that we have currently.

[01:24:09.660] And now when I show the probabilities,

[01:24:11.660] you see that every row here, of course,

[01:24:17.540] will sum to one, because they're normalized.

[01:24:21.060] And the shape of this is five by 27.

[01:24:25.060] And so really what we've achieved is

[01:24:27.540] for every one of our five examples,

[01:24:29.660] we now have a row that came out of a neural net.

[01:24:32.980] And because of the transformations here,

[01:24:35.460] we made sure that this output of this neural net now

[01:24:38.340] are probabilities, or we can say that

[01:24:40.300] are probabilities, or we can interpret to be probabilities.

[01:24:43.940] So our WX here gave us logits,

[01:24:47.980] and then we interpret those to be log counts.

[01:24:50.660] We exponentiate to get something that looks like counts,

[01:24:53.900] and then we normalize those counts

[01:24:55.260] to get a probability distribution.

[01:24:57.420] And all of these are differentiable operations.

[01:25:00.300] So what we've done now is we are taking inputs.

[01:25:03.220] We have differentiable operations

[01:25:04.660] that we can back propagate through,

[01:25:06.860] and we're getting out probability distributions.

[01:25:09.740] So, for example, for the zeroth example that fed in,

[01:25:15.380] which was the zeroth example here,

[01:25:18.180] was a one-hot vector of zero.

[01:25:20.500] And it basically corresponded

[01:25:24.260] to feeding in this example here.

[01:25:27.740] So we're feeding in a dot into a neural net.

[01:25:30.300] And the way we fed the dot into a neural net

[01:25:32.180] is that we first got its index,

[01:25:34.260] then we one-hot encoded it,

[01:25:36.500] then it went into the neural net,

[01:25:38.460] and out came this distribution of probabilities.

[01:25:43.340] And its shape is 27.

[01:25:47.220] There's 27 numbers.

[01:25:48.660] And we're going to interpret this

[01:25:50.380] as the neural net's assignment

[01:25:52.300] for how likely every one of these characters,

[01:25:56.660] the 27 characters, are to come next.

[01:25:59.700] And as we tune the weights W,

[01:26:02.340] we're going to be, of course,

[01:26:03.180] getting different probabilities out

[01:26:04.940] for any character that you input.

[01:26:07.180] And so now the question is just,

[01:26:08.540] can we optimize and find a good W

[01:26:10.980] such that the probabilities coming out are pretty good?

[01:26:14.500] And the way we measure pretty good is by the loss function.

[01:26:17.180] Okay, so I organized everything into a single summary

[01:26:19.380] so that hopefully it's a bit more clear.

[01:26:21.220] So it starts here.

[01:26:22.580] We have an input dataset.

[01:26:24.380] We have some inputs to the neural net,

[01:26:26.380] and we have some labels

[01:26:27.700] for the correct next character in a sequence.

[01:26:30.460] These are integers.

[01:26:32.700] Here I'm using tors generators now

[01:26:35.300] so that you see the same numbers that I see.

[01:26:38.540] And I'm generating 27 neurons weights.

[01:26:42.740] And each neuron here receives 27 inputs.

[01:26:48.540] Then here, we're going to plug in

[01:26:49.780] all the input examples, Xs, into a neural net.

[01:26:52.540] So here, this is a forward pass.

[01:26:55.580] First, we have to encode all of the inputs

[01:26:57.940] into one-hot representations.

[01:27:00.340] So we have 27 classes.

[01:27:01.660] We pass in these integers.

[01:27:03.740] And Xinc becomes a array that is five by 27,

[01:27:09.500] zeros, except for a few ones.

[01:27:12.220] We then multiply this in the first layer of a neural net

[01:27:14.980] to get logits,

[01:27:16.660] exponentiate the logits to get fake counts, sort of,

[01:27:20.620] and normalize these counts to get probabilities.

[01:27:24.260] So these last two lines, by the way, here,

[01:27:27.420] are called the softmax, which I pulled up here.

[01:27:31.900] Softmax is a very often used layer in a neural net

[01:27:35.900] that takes these Zs, which are logits,

[01:27:38.860] exponentiates them, and divides and normalizes.

[01:27:43.020] It's a way of taking outputs of a neural net layer.

[01:27:46.260] And these outputs can be positive or negative.

[01:27:49.780] And it outputs probability distributions.

[01:27:52.380] It outputs something that is always sums to one

[01:27:55.900] and are positive numbers, just like probabilities.

[01:27:58.820] So this is kind of like a normalization function,

[01:28:00.540] if you want to think of it that way.

[01:28:02.140] And you can put it on top of any other linear layer

[01:28:04.460] inside a neural net.

[01:28:05.620] And it basically makes a neural net output probabilities

[01:28:08.580] that's very often used.

[01:28:10.420] And we used it as well here.

[01:28:13.340] So this is the forward pass,

[01:28:14.380] and that's how we made a neural net output probability.

[01:28:17.820] Now, you'll notice that all of these,

[01:28:24.340] this entire forward pass is made up of differentiable layers.

[01:28:27.820] Everything here, we can backpropagate through.

[01:28:30.140] And we saw some of the backpropagation in microGrad.

[01:28:33.100] This is just multiplication and addition.

[01:28:36.460] All that's happening here is just multiply and then add.

[01:28:38.700] And we know how to backpropagate through them.

[01:28:40.700] Exponentiation, we know how to backpropagate through.

[01:28:43.780] And then here we are summing,

[01:28:46.460] and sum is easily backpropagatable as well,

[01:28:50.060] and division as well.

[01:28:51.980] So everything here is a differentiable operation,

[01:28:54.500] and we can backpropagate through.

[01:28:57.500] Now, we achieve these probabilities,

[01:28:59.580] which are five by 27.

[01:29:01.540] For every single example,

[01:29:03.140] we have a vector of probabilities that sum to one.

[01:29:06.260] And then here I wrote a bunch of stuff

[01:29:08.500] to sort of like break down the examples.

[01:29:11.420] So we have five examples making up Emma, right?

[01:29:16.260] And there are five bigrams inside Emma.

[01:29:19.940] So bigram example one is that E

[01:29:24.220] is the beginning character right after dot.

[01:29:27.020] And the indexes for these are zero and five.

[01:29:30.140] So then we feed in a zero,

[01:29:33.140] that's the input to the neural net.

[01:29:34.820] We get probabilities from the neural net

[01:29:37.020] that are 27 numbers.

[01:29:40.220] And then the label is five,

[01:29:42.460] because E actually comes after dot.

[01:29:44.820] So that's the label.

[01:29:46.860] And then we use this label five

[01:29:50.260] to index into the probability distribution here.

[01:29:53.140] So this index five here is zero, one, two, three, four, five.

[01:29:58.660] It's this number here, which is here.

[01:30:03.060] So that's basically the probability assigned

[01:30:04.700] by the neural net to the actual correct character.

[01:30:07.740] You see that the network currently thinks

[01:30:09.420] that this next character, that E following dot,

[01:30:12.380] is only 1% likely,

[01:30:14.260] which is of course not very good, right?

[01:30:15.860] Because this actually is a training example,

[01:30:18.580] and the network thinks that this is currently

[01:30:20.060] very, very unlikely.

[01:30:21.380] But that's just because we didn't get very lucky

[01:30:23.620] in generating a good setting of W.

[01:30:25.980] So right now this network thinks this is unlikely,

[01:30:28.380] and 0.01 is not a good outcome.

[01:30:31.060] So the log likelihood then is very negative.

[01:30:34.820] And the negative log likelihood is very positive.

[01:30:38.340] And so four is a very high negative log likelihood,

[01:30:42.180] and that means we're going to have a high loss.

[01:30:44.340] Because what is the loss?

[01:30:45.780] The loss is just the average negative log likelihood.

[01:30:48.900] So the second character is E, M.

[01:30:50.820] And you see here that also the network thought

[01:30:52.860] that M following E is very unlikely, 1%.

[01:30:58.660] For M following M, it thought it was 2%.

[01:31:01.540] And for A following M,

[01:31:02.940] it actually thought it was 7% likely.

[01:31:05.020] So just by chance,

[01:31:07.100] this one actually has a pretty good probability,

[01:31:09.100] and therefore a pretty low negative log likelihood.

[01:31:12.540] And finally here, it thought this was 1% likely.

[01:31:15.540] So overall, this is a very good result.

[01:31:18.340] So overall, our average negative log likelihood,

[01:31:20.940] which is the loss, the total loss that summarizes

[01:31:24.660] basically how well this network currently works,

[01:31:27.340] at least on this one word,

[01:31:28.740] not on the full data set, just the one word,

[01:31:30.820] is 3.76, which is actually a fairly high loss.

[01:31:34.140] This is not a very good setting of Ws.

[01:31:36.900] Now here's what we can do.

[01:31:38.660] We're currently getting 3.76.

[01:31:41.260] We can actually come here and we can change our W.

[01:31:44.220] We can resample it.

[01:31:45.580] So let me just add one to have a different seed,

[01:31:48.660] and then we get a different W,

[01:31:50.500] and then we can rerun this.

[01:31:52.820] And with this different seed,

[01:31:54.180] with this different setting of Ws,

[01:31:56.260] we now get 3.37.

[01:31:58.580] So this is a much better W, right?

[01:32:00.620] And it's better because the probabilities

[01:32:02.860] just happen to come out higher

[01:32:05.420] for the characters that actually are next.

[01:32:08.820] And so you can imagine actually just resampling this.

[01:32:11.620] We can try two.

[01:32:12.820] So, okay, this was not very good.

[01:32:15.740] Let's try one more.

[01:32:16.940] We can try three.

[01:32:19.340] Okay, this was terrible setting

[01:32:20.980] because we have a very high loss.

[01:32:23.020] So anyway, I'm gonna erase this.

[01:32:28.060] What I'm doing here, which is just guess and check

[01:32:30.340] of randomly assigning parameters

[01:32:31.900] and seeing if the network is good,

[01:32:33.660] that is amateur hour.

[01:32:35.620] That's not how you optimize a neural net.

[01:32:37.580] The way you optimize a neural net

[01:32:38.940] is you start with some random guess,

[01:32:40.700] and we're gonna come up with a random guess

[01:32:42.500] and we're gonna commit to this one,

[01:32:43.580] even though it's not very good.

[01:32:45.340] But now the big deal is we have a loss function.

[01:32:48.420] So this loss is made up only of differentiable operations.

[01:32:54.300] And we can minimize the loss by tuning Ws,

[01:32:58.820] by computing the gradients of the loss

[01:33:01.420] with respect to these W matrices.

[01:33:05.140] And so then we can tune W to minimize the loss

[01:33:07.740] and find a good setting of W

[01:33:09.780] using gradient-based optimization.

[01:33:11.820] So let's see how that would work.

[01:33:13.180] Now, things are actually going to look almost identical

[01:33:15.340] to what we had with microGrad.

[01:33:17.220] So here I pulled up the lecture from microGrad,

[01:33:21.060] the notebook that's from this repository.

[01:33:23.980] And when I scroll all the way to the end

[01:33:25.180] where we left off with microGrad,

[01:33:26.820] we had something very, very similar.

[01:33:28.660] We had a number of input examples.

[01:33:31.020] In this case, we had four input examples inside Xs

[01:33:34.300] and we had their targets, desired targets.

[01:33:37.780] Just like here, we have our Xs now,

[01:33:39.740] but we have five of them.

[01:33:40.860] And they're now integers instead of vectors.

[01:33:44.300] But we're going to convert our integers to vectors,

[01:33:47.100] except our vectors will be 27 large instead of three large.

[01:33:52.060] And then here, what we did is,

[01:33:53.340] first we did a forward pass

[01:33:54.940] where we ran a neural net on all of the inputs

[01:33:58.500] to get predictions.

[01:34:00.460] Our neural net at the time, this NFX,

[01:34:02.740] was a multilayer perceptron.

[01:34:05.260] Our neural net is going to look different

[01:34:07.260] because our neural net is just a single layer,

[01:34:10.540] single linear layer, followed by a softmax.

[01:34:13.980] So that's our neural net.

[01:34:15.980] And the loss here was the mean squared error.

[01:34:18.500] So we simply subtracted the prediction

[01:34:20.100] from the ground truth and squared it and summed it all up.

[01:34:23.060] And that was the loss.

[01:34:24.260] And loss was the single number

[01:34:25.940] that summarized the quality of the neural net.

[01:34:28.620] And when loss is low, like almost zero,

[01:34:32.020] that means the neural net is predicting correctly.

[01:34:36.340] So we had a single number

[01:34:37.780] that summarized the performance of the neural net.

[01:34:42.180] And everything here was differentiable

[01:34:43.740] and was stored in a massive compute graph.

[01:34:46.980] And then we iterated over all the parameters.

[01:34:49.300] We made sure that the gradients are set to zero

[01:34:51.860] and we called loss.backward.

[01:34:54.300] And loss.backward initiated backpropagation

[01:34:56.980] at the final output node of loss, right?

[01:34:59.660] So, yeah, remember these expressions?

[01:35:02.300] We had loss all the way at the end.

[01:35:03.700] We start backpropagation and we went all the way back

[01:35:06.540] and we made sure that we populated

[01:35:08.340] all the parameters.grad.

[01:35:10.900] So.grad started at zero, but backpropagation filled it in.

[01:35:14.620] And then in the update, we iterated all the parameters

[01:35:17.540] and we simply did a parameter update

[01:35:19.900] where every single element of our parameters

[01:35:23.660] was notched in the opposite direction of the gradient.

[01:35:27.620] And so we're going to do the exact same thing here.

[01:35:31.860] So I'm going to pull this up on the side here,

[01:35:38.620] so that we have it available.

[01:35:39.980] And we're actually going to do the exact same thing.

[01:35:42.180] So this was the forward pass.

[01:35:44.180] So we did this.

[01:35:47.020] And props is our widespread.

[01:35:49.060] So now we have to evaluate the loss,

[01:35:50.620] but we're not using the mean squared error.

[01:35:52.500] We're using the negative log likelihood

[01:35:54.220] because we are doing classification.

[01:35:55.700] We're not doing regression as it's called.

[01:35:57.900] So here, we want to calculate loss.

[01:36:00.820] Now, the way we calculate it is,

[01:36:03.100] it's just this average negative log likelihood.

[01:36:06.060] Now this props here has a shape of five by 27.

[01:36:11.980] And so to get all the,

[01:36:13.820] we basically want to pluck out the probabilities

[01:36:16.460] at the correct indices here.

[01:36:18.860] So in particular, because the labels are stored here

[01:36:21.620] in the array wise,

[01:36:23.260] basically what we're after is for the first example,

[01:36:25.900] we're looking at probability of five, right?

[01:36:28.460] At index five.

[01:36:29.700] For the second example,

[01:36:31.420] at the second row or row index one,

[01:36:34.980] we are interested in the probability asides to index 13.

[01:36:39.140] At the second example, we also have 13.

[01:36:41.940] At the third row, we want one.

[01:36:46.180] And at the last row, which is four, we want zero.

[01:36:50.060] So these are the probabilities we're interested in, right?

[01:36:52.860] And you can see that they're not amazing.

[01:36:55.060] They're not amazing as we saw above.

[01:36:58.900] So these are the probabilities we want,

[01:37:00.340] but we want like a more efficient way

[01:37:02.020] to access these probabilities,

[01:37:04.740] not just listing them out in a tuple like this.

[01:37:07.300] So it turns out that the way to this in PyTorch,

[01:37:09.500] one of the ways at least,

[01:37:10.940] is we can basically pass in all of these,

[01:37:16.940] sorry about that,

[01:37:18.020] all of these integers in a vectors.

[01:37:22.260] So these ones, you see how they're just zero, one, two,

[01:37:26.260] three, four,

[01:37:27.340] we can actually create that using mp, not mp, sorry,

[01:37:30.380] torch.arrange of five, zero, one, two, three, four.

[01:37:34.620] So we can index here with torch.arrange of five,

[01:37:38.500] and here we index with ys.

[01:37:41.420] And you see that that gives us exactly these numbers.

[01:37:49.180] So that plucks out the probabilities

[01:37:51.660] that the neural network assigns

[01:37:53.140] to the correct next character.

[01:37:56.420] Now we take those probabilities and we don't,

[01:37:58.700] we actually look at the log probability.

[01:38:00.620] So we want to dot log,

[01:38:03.660] and then we want to just average that up.

[01:38:06.780] So take the mean of all of that.

[01:38:08.700] And then it's the negative average log likelihood

[01:38:11.980] that is the loss.

[01:38:14.380] So the loss here is 3.7 something.

[01:38:18.140] And you see that this loss 3.76, 3.76 is exactly

[01:38:22.140] as we've obtained before,

[01:38:23.460] but this is a vectorized form of that expression.

[01:38:26.540] So we get the same loss.

[01:38:29.620] And the same loss we can consider sort of

[01:38:31.820] as part of this forward pass,

[01:38:34.180] and we've achieved here now loss.

[01:38:36.460] Okay, so we made our way all the way to loss.

[01:38:38.420] We've defined the forward pass.

[01:38:40.140] We forwarded the network and the loss.

[01:38:42.180] Now we're ready to do the backward pass.

[01:38:44.420] So backward pass, we want to first make sure

[01:38:49.300] that all the gradients are reset.

[01:38:51.020] So they're at zero.

[01:38:52.500] Now in PyTorch, you can set the gradients to be zero,

[01:38:56.020] but you can also just set it to none.

[01:38:58.220] And setting it to none is more efficient.

[01:39:00.300] And PyTorch will interpret none as like a lack

[01:39:03.460] of a gradient and it's the same as zeros.

[01:39:05.860] So this is a way to set to zero the gradient.

[01:39:08.620] And now we do loss.backward.

[01:39:12.780] Before we do loss.backward, we need one more thing.

[01:39:16.940] If you remember from micrograd,

[01:39:18.940] PyTorch actually requires that we pass in requires grad

[01:39:23.340] is true so that we tell PyTorch that we are interested

[01:39:28.340] in calculating the gradient for this leaf tensor.

[01:39:31.420] By default, this is false.

[01:39:33.540] So let me recalculate with that.

[01:39:35.940] And then set to none and loss.backward.

[01:39:40.780] Now something magical happened when loss.backward was run

[01:39:44.460] because PyTorch, just like micrograd,

[01:39:47.140] when we did the forward pass here,

[01:39:49.140] it keeps track of all the operations under the hood.

[01:39:52.380] It builds a full computational graph.

[01:39:54.860] Just like the graphs we produced in micrograd,

[01:39:57.900] those graphs exist inside PyTorch.

[01:40:00.820] And so it knows all the dependencies

[01:40:02.740] and all the mathematical operations of everything.

[01:40:05.060] And when you then calculate the loss,

[01:40:07.140] we can call a.backward on it.

[01:40:09.620] And.backward then fills in the gradients

[01:40:12.740] of all the intermediates all the way back to w's,

[01:40:17.860] which are the parameters of our neural net.

[01:40:19.980] So now we can do w.grad

[01:40:22.420] and we see that it has structure.

[01:40:23.980] There's stuff inside it.

[01:40:29.180] And these gradients, every single element here,

[01:40:32.340] so w.shape is 27 by 27.

[01:40:35.900] w.grad.shape is the same, 27 by 27.

[01:40:39.780] And every element of w.grad is telling us

[01:40:43.540] the influence of that weight on the loss function.

[01:40:47.900] So for example, this number all the way here,

[01:40:51.100] if this element, the zero zero element of w,

[01:40:54.740] because the gradient is positive,

[01:40:56.660] it's telling us that this has a positive influence

[01:40:59.500] on the loss, slightly nudging w, slightly taking w zero zero

[01:41:05.900] and adding a small h to it would increase the loss mildly

[01:41:12.300] because this gradient is positive.

[01:41:14.860] Some of these gradients are also negative.

[01:41:17.460] So that's telling us about the gradient information

[01:41:20.300] and we can use this gradient information

[01:41:22.380] to update the weights of this neural network.

[01:41:25.780] So let's now do the update.

[01:41:27.420] It's going to be very similar to what we had in microGrad.

[01:41:29.980] We need no loop over all the parameters

[01:41:32.540] because we only have one parameter, tensor, and that is w.

[01:41:36.380] So we simply do w.data plus equals,

[01:41:41.460] we can actually copy this almost exactly,

[01:41:43.180] negative 0.1 times w.grad.

[01:41:48.700] And that would be the update to the tensor.

[01:41:52.140] So that updates the tensor and because the tensor is updated,

[01:41:58.380] we would expect that now the loss should decrease.

[01:42:01.660] So here, if I print loss.item, it was 3.76, right?

[01:42:10.500] So we've updated the w here.

[01:42:13.300] So if I recalculate forward pass,

[01:42:16.220] loss now should be slightly lower.

[01:42:18.860] So 3.76 goes to 3.74.

[01:42:23.220] And then we can again set grad to none and backward, update.

[01:42:30.100] And now the parameter is changed again.

[01:42:32.340] So if we recalculate the forward pass,

[01:42:34.540] we expect a lower loss again, 3.72, okay?

[01:42:40.340] And this is again doing the,

[01:42:41.900] we're now doing gradient descent.

[01:42:43.500] And when we achieve a low loss, that will mean that the network

[01:42:46.860] is assigning high probabilities

[01:42:48.700] to the correct next characters.

[01:42:50.260] Okay, so I rearranged everything

[01:42:51.500] and I put it all together from scratch.

[01:42:54.420] So here is where we construct our dataset of bigrams.

[01:42:58.220] You see that we are still iterating

[01:42:59.620] only over the first word, emma.

[01:43:01.940] I'm going to change that in a second.

[01:43:04.060] I added a number that counts the number of elements in Xs,

[01:43:08.740] and I'm going to change that in a second.

[01:43:11.020] I'm going to adjust the number of elements in Xs

[01:43:14.540] so that we explicitly see that the number of examples is five,

[01:43:17.900] because currently we're just working with emma,

[01:43:19.500] and there's five bigrams there.

[01:43:21.580] And here I added a loop of exactly what we had before.

[01:43:24.580] So we had 10 iterations of gradient descent

[01:43:27.060] of forward pass, backward pass, and an update.

[01:43:29.900] And so running these two cells, initialization

[01:43:32.060] and gradient descent,

[01:43:33.620] gives us some improvement on the loss function.

[01:43:38.100] But now I want to use all the words.

[01:43:41.580] And there's not five, but 228,000 bigrams now.

[01:43:46.580] However, this should require no modification whatsoever.

[01:43:49.300] Everything should just run,

[01:43:50.700] because all the code we wrote doesn't care

[01:43:52.460] if there's five bigrams or 228,000 bigrams,

[01:43:55.780] and with everything we should just work.

[01:43:57.140] So you see that this will just run.

[01:44:00.340] But now we are optimizing over the entire training set

[01:44:02.660] of all the bigrams.

[01:44:04.620] And you see now that we are decreasing very slightly.

[01:44:07.460] So actually we can probably afford a larger learning rate.

[01:44:12.340] We can probably afford even larger learning rate.

[01:44:20.620] Even 50 seems to work on this very, very simple example.

[01:44:23.900] So let me reinitialize and let's run 100 iterations.

[01:44:29.220] See what happens.

[01:44:30.220] We seem to be coming up to some pretty good losses here.

[01:44:35.220] 2.47, let me run 100 more.

[01:44:38.460] What is the number that we expect, by the way, in the loss?

[01:44:41.140] We expect to get something around

[01:44:42.980] what we had originally, actually.

[01:44:45.820] So all the way back, if you remember,

[01:44:47.380] in the beginning of this video,

[01:44:48.700] when we optimized just by counting,

[01:44:52.700] our loss was roughly 2.47.

[01:44:55.460] So that's the number that we expect to get,

[01:44:57.500] just by counting, our loss was roughly 2.47

[01:45:01.660] after we added smoothing.

[01:45:03.660] But before smoothing, we had roughly 2.45 likelihood,

[01:45:08.460] sorry, loss.

[01:45:09.820] And so that's actually roughly the vicinity

[01:45:11.660] of what we expect to achieve.

[01:45:13.940] But before we achieved it by counting,

[01:45:15.980] and here we are achieving roughly the same result,

[01:45:18.900] but with gradient-based optimization.

[01:45:21.060] So we come to about 2.46, 2.45, et cetera.

[01:45:26.060] And that makes sense, because fundamentally,

[01:45:27.500] we're not taking in any additional information.

[01:45:29.700] We're still just taking in the previous character

[01:45:31.580] and trying to predict the next one.

[01:45:33.380] But instead of doing it explicitly by counting

[01:45:35.980] and normalizing,

[01:45:37.940] we are doing it with gradient-based learning.

[01:45:39.820] And it just so happens that the explicit approach

[01:45:42.300] happens to very well optimize the loss function

[01:45:45.300] without any need for a gradient-based optimization,

[01:45:48.220] because the setup for bigram language models

[01:45:50.220] is so straightforward and so simple.

[01:45:52.620] We can just afford to estimate those probabilities directly

[01:45:55.620] and maintain them in a table.

[01:45:58.980] But the gradient-based approach

[01:46:00.700] is significantly more flexible.

[01:46:02.980] So we've actually gained a lot,

[01:46:04.860] because what we can do now is we can expand this approach

[01:46:11.060] and complexify the neural net.

[01:46:12.940] So currently, we're just taking a single character

[01:46:14.740] and feeding into a neural net,

[01:46:15.980] and the neural net is extremely simple.

[01:46:17.900] But we're about to iterate on this substantially.

[01:46:20.540] We're going to be taking multiple previous characters,

[01:46:23.460] and we're going to be feeding them into increasingly

[01:46:26.100] more complex neural nets.

[01:46:27.580] But fundamentally, the output of the neural net

[01:46:30.100] will always just be logits.

[01:46:32.740] And those logits will go through

[01:46:33.980] the exact same transformation.

[01:46:35.580] We are going to take them through a softmax,

[01:46:38.020] calculate the loss function and the negative log-likelihood,

[01:46:41.180] and do gradient-based optimization.

[01:46:43.660] And so actually, as we complexify the neural nets

[01:46:47.060] and work all the way up to transformers,

[01:46:49.820] none of this will really fundamentally change.

[01:46:52.020] None of this will fundamentally change.

[01:46:53.740] The only thing that will change is the way we do

[01:46:56.420] the forward pass, where we take in some previous characters

[01:46:59.140] and calculate logits for the next character in a sequence.

[01:47:02.940] That will become more complex,

[01:47:05.180] but we'll use the same machinery to optimize it.

[01:47:08.940] And it's not obvious how we would have extended

[01:47:13.260] this bigram approach into the case

[01:47:16.380] where there are many more characters at the input,

[01:47:19.300] because eventually these tables would get way too large

[01:47:22.300] because there's way too many combinations

[01:47:24.260] of what previous characters could be.

[01:47:27.940] If you only have one previous character,

[01:47:29.740] we can just keep everything in a table that counts.

[01:47:32.180] But if you have the last 10 characters that are input,

[01:47:35.180] we can't actually keep everything in the table anymore.

[01:47:37.500] So this is fundamentally an unscalable approach,

[01:47:39.940] and the neural network approach

[01:47:41.220] is significantly more scalable.

[01:47:43.140] And it's something that actually we can improve on over time.

[01:47:46.860] So that's where we will be digging next.

[01:47:48.620] I wanted to point out two more things.

[01:47:51.220] Number one, I want you to notice that this xinc here,

[01:47:56.860] this is made up of one-hot vectors.

[01:47:59.060] And then those one-hot vectors

[01:48:00.380] are multiplied by this W matrix.

[01:48:03.300] And we think of this as multiple neurons being forwarded

[01:48:06.740] in a fully connected manner.

[01:48:08.820] But actually what's happening here is that, for example,

[01:48:12.060] if you have a one-hot vector here that has a one

[01:48:15.700] at say the fifth dimension,

[01:48:17.860] then because of the way the matrix multiplication works,

[01:48:21.180] multiplying that one-hot vector with W

[01:48:23.500] actually ends up plucking out the fifth row of W.

[01:48:27.580] Logits would become just the fifth row of W.

[01:48:31.380] And that's because of the way the matrix multiplication works.

[01:48:36.900] So that's actually what ends up happening.

[01:48:40.060] So, but that's actually exactly what happened before.

[01:48:43.340] Because remember all the way up here,

[01:48:45.220] we have a bigram.

[01:48:46.420] We took the first character

[01:48:47.780] and then that first character indexed

[01:48:49.740] into a row of this array here.

[01:48:53.540] And that row gave us the probability distribution

[01:48:55.780] for the next character.

[01:48:57.260] So the first character was used as a lookup

[01:48:59.700] into a matrix here to get the probability distribution.

[01:49:05.020] Well, that's actually exactly what's happening here.

[01:49:07.220] Because we're taking the index,

[01:49:09.300] we're encoding it as one-hot and multiplying it by W.

[01:49:12.140] So Logits literally becomes the appropriate row of W.

[01:49:19.620] And that gets, just as before,

[01:49:21.500] exponentiated to create the counts

[01:49:23.700] and then normalized and becomes probability.

[01:49:26.260] So this W here is literally the same as this array here.

[01:49:33.740] But W remember is the log counts, not the counts.

[01:49:37.660] So it's more precise to say that W exponentiated

[01:49:41.140] W dot exp is this array.

[01:49:46.340] But this array was filled in by counting

[01:49:49.420] and by basically populating the counts of bigrams.

[01:49:54.020] Whereas in the gradient-based framework,

[01:49:55.980] we initialize it randomly and then we let loss guide us

[01:50:00.700] to arrive at the exact same array.

[01:50:03.340] So this array exactly here is basically the array W

[01:50:08.820] at the end of optimization,

[01:50:10.260] except we arrived at it piece by piece by following the loss.

[01:50:15.140] And that's why we also obtain

[01:50:16.300] the same loss function at the end.

[01:50:18.060] And the second notice, if I come here,

[01:50:20.620] remember the smoothing where we added fake counts

[01:50:23.780] to our counts in order to smooth out

[01:50:26.700] and make more uniform the distributions

[01:50:29.460] of these probabilities.

[01:50:31.180] And that prevented us from assigning zero probability

[01:50:33.460] to any one bigram.

[01:50:37.260] Now, if I increase the count here,

[01:50:40.340] what's happening to the probability?

[01:50:42.940] As I increase the count,

[01:50:44.860] probability becomes more and more uniform, right?

[01:50:48.220] Because these counts go only up to like 900 or whatever.

[01:50:51.580] So if I'm adding plus a million to every single number here,

[01:50:55.220] you can see how the row and its probability then

[01:50:58.260] when we divide is just going to become more and more close

[01:51:00.780] to exactly even probability, uniform distribution.

[01:51:04.260] It turns out that the gradient-based framework

[01:51:06.340] has an equivalent to smoothing.

[01:51:09.940] In particular, think through these Ws here,

[01:51:14.940] which we initialized randomly.

[01:51:17.740] We could also think about initializing Ws to be zero.

[01:51:21.220] If all the entries of W are zero,

[01:51:25.220] then you'll see that logits will become all zero

[01:51:28.020] and then exponentiating those logits becomes all one.

[01:51:31.300] And then the probabilities turn out to be exactly the same.

[01:51:34.220] Exactly uniform.

[01:51:35.980] So basically when Ws are all equal to each other,

[01:51:38.820] or say, especially zero,

[01:51:41.340] then the probabilities come out completely uniform.

[01:51:44.580] So trying to incentivize W to be near zero

[01:51:49.380] is basically equivalent to label smoothing.

[01:51:53.100] And the more you incentivize that in a loss function,

[01:51:55.700] the more smooth distribution you're going to achieve.

[01:51:59.100] So this brings us to something that's called regularization,

[01:52:02.140] where we can actually augment the loss function

[01:52:04.540] to have a small component

[01:52:06.100] that we call a regularization loss.

[01:52:09.260] In particular, what we're going to do is we can take W

[01:52:11.820] and we can, for example, square all of its entries.

[01:52:14.820] And then we can, oops, sorry about that.

[01:52:19.260] We can take all the entries of W and we can sum them.

[01:52:23.820] And because we're squaring, there will be no signs anymore.

[01:52:28.500] Negatives and positives all get squashed

[01:52:30.260] to be positive numbers.

[01:52:31.700] And then the way this works is you achieve zero loss

[01:52:35.220] if W is exactly or zero,

[01:52:37.380] but if W has non-zero numbers, you accumulate loss.

[01:52:41.380] And so we can actually take this and we can add it on here.

[01:52:45.020] So we can do something like loss plus W square dot sum.

[01:52:52.180] Or let's actually, instead of sum, let's take a mean,

[01:52:54.660] because otherwise the sum gets too large.

[01:52:57.700] So mean is like a little bit more manageable.

[01:53:00.060] And then we have a regularization loss here,

[01:53:02.660] let's say 0.01 times or something like that.

[01:53:05.420] You can choose the regularization strength.

[01:53:08.580] And then we can just optimize this.

[01:53:11.340] And now this optimization actually has two components.

[01:53:14.500] Not only is it trying to make

[01:53:15.620] all the probabilities work out,

[01:53:17.580] but in addition to that, there's an additional component

[01:53:19.540] that simultaneously tries to make all Ws be zero,

[01:53:23.100] because if Ws are non-zero, you feel a loss.

[01:53:25.500] And so minimizing this, the only way to achieve that

[01:53:27.980] is for W to be zero.

[01:53:30.180] And so you can think of this as adding like a spring force

[01:53:33.380] or like a gravity force that pushes W to be zero.

[01:53:37.380] So W wants to be zero and the probabilities

[01:53:39.540] want to be uniform, but they also simultaneously

[01:53:42.260] want to match up your probabilities

[01:53:45.140] as indicated by the data.

[01:53:46.940] And so the strength of this regularization

[01:53:50.220] is exactly controlling the amount of counts

[01:53:54.020] that you add here.

[01:53:55.140] Adding a lot more counts here

[01:53:58.660] corresponds to increasing this number,

[01:54:02.660] because the more you increase it,

[01:54:04.300] the more this part of the loss function dominates this part,

[01:54:07.500] and the more these weights will be unable to grow,

[01:54:11.620] because as they grow, they accumulate way too much loss.

[01:54:16.140] And so if this is strong enough,

[01:54:19.100] then we are not able to overcome the force of this loss.

[01:54:22.740] And we will never, and basically everything

[01:54:25.500] will be uniform predictions.

[01:54:27.340] So I thought that's kind of cool.

[01:54:28.660] Okay, and lastly, before we wrap up,

[01:54:31.060] I wanted to show you how you would sample

[01:54:32.580] from this neural net model.

[01:54:34.700] And I copy pasted the sampling code from before,

[01:54:38.500] where remember that we sampled five times.

[01:54:42.140] And all we did, we started at zero,

[01:54:44.140] we grabbed the current IX row of P,

[01:54:48.220] and that was our probability row

[01:54:50.220] from which we sampled the next index

[01:54:52.820] and just accumulated that and break when zero.

[01:54:56.820] And running this gave us these results.

[01:55:01.780] I still have the P in memory, so this is fine.

[01:55:05.580] Now, this P doesn't come from the row of P,

[01:55:09.940] instead it comes from this neural net.

[01:55:12.940] First, we take IX and we encode it into a one-hot row.

[01:55:17.940] Of X-sync, this X-sync multiplies our W,

[01:55:22.460] which really just plucks out the row of W

[01:55:24.620] corresponding to IX, really that's what's happening.

[01:55:27.780] And that gets our logits,

[01:55:29.620] and then we normalize those logits,

[01:55:32.220] exponentiate to get counts,

[01:55:33.620] and then normalize to get the distribution.

[01:55:36.420] And then we can sample from the distribution.

[01:55:38.740] So if I run this, kind of anti-climatic or climatic,

[01:55:44.740] depending on how you look at it, but we get the exact same result.

[01:55:50.140] And that's because this is the identical model.

[01:55:52.540] Not only does it achieve the same loss,

[01:55:54.740] but as I mentioned, these are identical models,

[01:55:57.740] and this W is the log counts of what we've estimated before.

[01:56:02.140] But we came to this answer in a very different way,

[01:56:05.340] and it's got a very different interpretation.

[01:56:07.340] But fundamentally, this is basically the same model

[01:56:09.340] and gives the same samples here.

[01:56:10.940] And so that's kind of cool. Okay, so we've actually covered a lot of ground.

[01:56:15.940] We introduced the bigram character level language model.

[01:56:19.940] We saw how we can train the model,

[01:56:21.940] how we can sample from the model,

[01:56:23.540] and how we can evaluate the quality of the model

[01:56:25.940] using the negative log likelihood loss.

[01:56:28.340] And then we actually trained the model in two completely different ways

[01:56:31.340] that actually get the same result and the same model.

[01:56:34.340] In the first way, we just counted up the frequency of all the bigrams

[01:56:38.340] and normalized. In the second way,

[01:56:41.340] we used the negative log likelihood loss as a guide

[01:56:46.340] to optimizing the counts matrix or the counts array

[01:56:50.340] so that the loss is minimized in a gradient-based framework.

[01:56:54.340] And we saw that both of them give the same result, and that's it.

[01:56:59.340] Now, the second one of these, the gradient-based framework,

[01:57:02.340] is much more flexible.

[01:57:03.340] And right now, our neural network is super simple.

[01:57:06.340] We're taking a single previous character,

[01:57:09.340] and we're taking it through a single linear layer to calculate the logits.

[01:57:13.340] This is about to complexify.

[01:57:15.340] So in the follow-up videos,

[01:57:16.340] we're going to be taking more and more of these characters,

[01:57:19.340] and we're going to be feeding them into a neural net.

[01:57:22.340] But this neural net will still output the exact same thing.

[01:57:25.340] The neural net will output logits.

[01:57:27.340] And these logits will still be normalized in the exact same way,

[01:57:30.340] and all the loss and everything else in the gradient-based framework,

[01:57:33.340] everything stays identical.

[01:57:34.340] It's just that this neural net will now complexify all the way to transformers.

[01:57:39.340] So that's going to be pretty awesome, and I'm looking forward to it.

[01:57:42.340] For now, bye.

